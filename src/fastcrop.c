#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <getopt.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <inttypes.h>

#include "jpeglib.h"

#define QUEUE_CAPACITY 1024
#define PATH_MAX_LEN 2048

// Custom error manager for libjpeg that recovers gracefully without exit()
struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
    char last_error[JMSG_LENGTH_MAX];
};

static void my_error_exit(j_common_ptr cinfo) {
    struct my_error_mgr *myerr = (struct my_error_mgr *)cinfo->err;
    (*cinfo->err->format_message)(cinfo, myerr->last_error);
    longjmp(myerr->setjmp_buffer, 1);
}

static void my_output_message(j_common_ptr cinfo) {
    (void)cinfo;
    // Suppress non-fatal warning spam to keep terminal clean
}

typedef struct {
    char in_path[PATH_MAX_LEN];
    char out_path[PATH_MAX_LEN];
    int64_t in_size;
} WorkItem;

typedef struct {
    WorkItem items[QUEUE_CAPACITY];
    int head;
    int tail;
    int count;
    bool done;
    int active_workers;
    uint64_t total_enqueued;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} WorkQueue;

typedef struct {
    char in_dir[PATH_MAX_LEN];
    char out_dir[PATH_MAX_LEN];
    int crop_x;
    int crop_y;
    int crop_w;
    int crop_h;
    float pct_x;
    float pct_y;
    float pct_w;
    float pct_h;
    bool use_pct;
    int quality;
    int num_threads;
    bool recursive;
    bool skip_existing;
    bool verbose;
} Config;

typedef struct {
    pthread_mutex_t lock;
    uint64_t processed;
    uint64_t skipped;
    uint64_t failed;
    uint64_t bytes_in;
    uint64_t bytes_out;
} Stats;

static Config g_config = {
    .in_dir = "",
    .out_dir = "",
    .crop_x = 0,
    .crop_y = 0,
    .crop_w = 0,
    .crop_h = 0,
    .pct_x = 0.0f,
    .pct_y = 0.0f,
    .pct_w = 0.0f,
    .pct_h = 0.0f,
    .use_pct = false,
    .quality = 75,
    .num_threads = 4,
    .recursive = true,
    .skip_existing = false,
    .verbose = false
};

static WorkQueue g_queue;
static Stats g_stats;
static volatile sig_atomic_t g_stop_requested = 0;

static void sigint_handler(int sig) {
    (void)sig;
    g_stop_requested = 1;
}

static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static bool is_jpeg_extension(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot) return false;
    return (strcasecmp(dot, ".jpg") == 0 ||
            strcasecmp(dot, ".jpeg") == 0);
}

static int mkdir_p(const char *path) {
    char tmp[PATH_MAX_LEN];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (len == 0) return 0;
    if (tmp[len - 1] == '/') tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int mkdir_parent(const char *filepath) {
    char tmp[PATH_MAX_LEN];
    snprintf(tmp, sizeof(tmp), "%s", filepath);
    char *slash = strrchr(tmp, '/');
    if (slash) {
        *slash = 0;
        return mkdir_p(tmp);
    }
    return 0;
}

static int64_t get_file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return st.st_size;
    }
    return -1;
}

// Scanline crop engine: partially decodes JPEG, saving RAM and CPU cycles
static int process_single_jpeg(const char *in_path, const char *out_path,
                               const Config *cfg, uint64_t *out_size) {
    *out_size = 0;

    if (cfg->skip_existing) {
        int64_t sz = get_file_size(out_path);
        if (sz > 0) {
            return 1; // Skipped
        }
    }

    FILE *infile = fopen(in_path, "rb");
    if (!infile) {
        return -1;
    }

    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    jerr.pub.output_message = my_output_message;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return -2;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    // Fast integer IDCT for maximum performance on ARM Cortex-A53
    cinfo.dct_method = JDCT_IFAST;

    // Calculate crop rectangle
    int cx, cy, cw, ch;
    if (cfg->use_pct) {
        cx = (int)(cfg->pct_x * (float)cinfo.image_width / 100.0f);
        cy = (int)(cfg->pct_y * (float)cinfo.image_height / 100.0f);
        cw = (int)(cfg->pct_w * (float)cinfo.image_width / 100.0f);
        ch = (int)(cfg->pct_h * (float)cinfo.image_height / 100.0f);
    } else {
        cx = cfg->crop_x;
        cy = cfg->crop_y;
        cw = cfg->crop_w;
        ch = cfg->crop_h;
    }

    // Clamp bounds
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx >= (int)cinfo.image_width) cx = cinfo.image_width - 1;
    if (cy >= (int)cinfo.image_height) cy = cinfo.image_height - 1;
    if (cx + cw > (int)cinfo.image_width) cw = cinfo.image_width - cx;
    if (cy + ch > (int)cinfo.image_height) ch = cinfo.image_height - cy;
    if (cw <= 0) cw = 1;
    if (ch <= 0) ch = 1;

    JDIMENSION x_start = cx;
    JDIMENSION crop_w = cw;

    jpeg_start_decompress(&cinfo);

    // Crop scanline to MCU boundary covering [cx, cx + cw]
    jpeg_crop_scanline(&cinfo, &x_start, &crop_w);

    // Calculate pixel offset within the aligned MCU block
    int x_offset_pixels = cx - (int)x_start;
    if (x_offset_pixels < 0) x_offset_pixels = 0;

    // Skip preceding scanlines without IDCT decompression
    if (cy > 0) {
        jpeg_skip_scanlines(&cinfo, cy);
    }

    // Ensure output parent directory exists
    mkdir_parent(out_path);

    // Write to a temporary file, then atomic rename
    char tmp_out_path[PATH_MAX_LEN + 32];
    snprintf(tmp_out_path, sizeof(tmp_out_path), "%s.tmp%lx", out_path, (unsigned long)pthread_self());

    FILE *outfile = fopen(tmp_out_path, "wb");
    if (!outfile) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return -3;
    }

    struct jpeg_compress_struct cinfo_out;
    struct my_error_mgr jerr_out;
    cinfo_out.err = jpeg_std_error(&jerr_out.pub);
    jerr_out.pub.error_exit = my_error_exit;
    jerr_out.pub.output_message = my_output_message;

    if (setjmp(jerr_out.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo_out);
        fclose(outfile);
        unlink(tmp_out_path);
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return -4;
    }

    jpeg_create_compress(&cinfo_out);
    jpeg_stdio_dest(&cinfo_out, outfile);

    // Exact requested width and height
    cinfo_out.image_width = cw;
    cinfo_out.image_height = ch;
    cinfo_out.input_components = cinfo.output_components;
    cinfo_out.in_color_space = cinfo.out_color_space;

    jpeg_set_defaults(&cinfo_out);
    jpeg_set_quality(&cinfo_out, cfg->quality, TRUE);
    cinfo_out.optimize_coding = TRUE; // Minimizes file size
    cinfo_out.dct_method = JDCT_IFAST;

    jpeg_start_compress(&cinfo_out, TRUE);

    // Allocate buffer for the MCU-aligned width
    JSAMPARRAY row_buffer = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr)&cinfo, JPOOL_IMAGE, crop_w * cinfo.output_components, 1);

    int bytes_per_pixel = cinfo.output_components;
    int offset_bytes = x_offset_pixels * bytes_per_pixel;

    int lines_written = 0;
    while (lines_written < ch && cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, row_buffer, 1);
        JSAMPROW out_row_ptr = row_buffer[0] + offset_bytes;
        jpeg_write_scanlines(&cinfo_out, &out_row_ptr, 1);
        lines_written++;
    }

    // Skip the remainder of the file
    if (cinfo.output_scanline < cinfo.output_height) {
        jpeg_skip_scanlines(&cinfo, cinfo.output_height - cinfo.output_scanline);
    }

    jpeg_finish_compress(&cinfo_out);
    jpeg_destroy_compress(&cinfo_out);
    fclose(outfile);

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    // Rename temp file to final output path
    if (rename(tmp_out_path, out_path) != 0) {
        unlink(tmp_out_path);
        return -5;
    }

    int64_t final_size = get_file_size(out_path);
    if (final_size > 0) {
        *out_size = (uint64_t)final_size;
    }
    return 0; // Success
}

static void *worker_thread(void *arg) {
    (void)arg;
    while (!g_stop_requested) {
        WorkItem item;
        pthread_mutex_lock(&g_queue.lock);
        while (g_queue.count == 0 && !g_queue.done && !g_stop_requested) {
            pthread_cond_wait(&g_queue.not_empty, &g_queue.lock);
        }

        if ((g_queue.count == 0 && g_queue.done) || g_stop_requested) {
            pthread_mutex_unlock(&g_queue.lock);
            break;
        }

        item = g_queue.items[g_queue.head];
        g_queue.head = (g_queue.head + 1) % QUEUE_CAPACITY;
        g_queue.count--;
        g_queue.active_workers++;
        pthread_cond_signal(&g_queue.not_full);
        pthread_mutex_unlock(&g_queue.lock);

        uint64_t out_size = 0;
        int rc = process_single_jpeg(item.in_path, item.out_path, &g_config, &out_size);

        pthread_mutex_lock(&g_queue.lock);
        g_queue.active_workers--;
        pthread_mutex_unlock(&g_queue.lock);

        pthread_mutex_lock(&g_stats.lock);
        if (rc == 0) {
            g_stats.processed++;
            g_stats.bytes_in += (item.in_size > 0 ? (uint64_t)item.in_size : 0);
            g_stats.bytes_out += out_size;
        } else if (rc == 1) {
            g_stats.skipped++;
        } else {
            g_stats.failed++;
            if (g_config.verbose) {
                fprintf(stderr, "\n[Warning] Failed to crop %s (code %d)\n", item.in_path, rc);
            }
        }
        pthread_mutex_unlock(&g_stats.lock);
    }
    return NULL;
}

static void queue_push(const char *in_path, const char *out_path, int64_t in_size) {
    pthread_mutex_lock(&g_queue.lock);
    while (g_queue.count == QUEUE_CAPACITY && !g_stop_requested) {
        pthread_cond_wait(&g_queue.not_full, &g_queue.lock);
    }

    if (g_stop_requested) {
        pthread_mutex_unlock(&g_queue.lock);
        return;
    }

    WorkItem *item = &g_queue.items[g_queue.tail];
    snprintf(item->in_path, sizeof(item->in_path), "%s", in_path);
    snprintf(item->out_path, sizeof(item->out_path), "%s", out_path);
    item->in_size = in_size;

    g_queue.tail = (g_queue.tail + 1) % QUEUE_CAPACITY;
    g_queue.count++;
    g_queue.total_enqueued++;
    pthread_cond_signal(&g_queue.not_empty);
    pthread_mutex_unlock(&g_queue.lock);
}

static void join_path(char *dst, size_t dst_sz, const char *p1, const char *p2) {
    if (!p2 || strlen(p2) == 0) {
        snprintf(dst, dst_sz, "%s", p1 ? p1 : "");
    } else if (!p1 || strlen(p1) == 0) {
        snprintf(dst, dst_sz, "%s", p2);
    } else {
        size_t l1 = strlen(p1);
        if (p1[l1 - 1] == '/') {
            snprintf(dst, dst_sz, "%s%s", p1, p2);
        } else {
            snprintf(dst, dst_sz, "%s/%s", p1, p2);
        }
    }
}

static void scan_directory(const char *in_dir, const char *out_dir, const char *rel_subpath) {
    char current_in[PATH_MAX_LEN];
    join_path(current_in, sizeof(current_in), in_dir, rel_subpath);

    DIR *dir = opendir(current_in);
    if (!dir) {
        if (g_config.verbose) {
            fprintf(stderr, "[Warning] Cannot open dir %s: %s\n", current_in, strerror(errno));
        }
        return;
    }

    struct dirent *entry;
    while (!g_stop_requested && (entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        char item_subpath[PATH_MAX_LEN];
        join_path(item_subpath, sizeof(item_subpath), rel_subpath, entry->d_name);

        char full_in[PATH_MAX_LEN];
        join_path(full_in, sizeof(full_in), in_dir, item_subpath);

        struct stat st;
        if (stat(full_in, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            if (g_config.recursive) {
                char *real_sub = realpath(full_in, NULL);
                char *real_out = realpath(out_dir, NULL);
                bool is_out = false;
                if (real_sub && real_out && strcmp(real_sub, real_out) == 0) {
                    is_out = true;
                }
                free(real_sub);
                free(real_out);
                if (is_out) {
                    continue;
                }
                scan_directory(in_dir, out_dir, item_subpath);
            }
        } else if (S_ISREG(st.st_mode) && is_jpeg_extension(entry->d_name)) {
            char full_out[PATH_MAX_LEN];
            join_path(full_out, sizeof(full_out), out_dir, item_subpath);
            queue_push(full_in, full_out, st.st_size);
        }
    }
    closedir(dir);
}

static void print_help(const char *progname) {
    printf("Fast Picture Crop Tool for Raspberry Pi Zero 2\n");
    printf("Ultra-low RAM, multi-core streaming crop engine for large photo batches.\n\n");
    printf("Usage:\n");
    printf("  %s -i <input_dir> -o <output_dir> -c <x,y,w,h> [options]\n", progname);
    printf("  %s -i <input_dir> -o <output_dir> -p <pct_x,pct_y,pct_w,pct_h> [options]\n\n", progname);
    printf("Crop Specification (choose one):\n");
    printf("  -c, --crop <x,y,w,h>          Pixel crop rectangle (e.g. 100,200,800,600)\n");
    printf("  -p, --crop-pct <x,y,w,h>      Percentage crop rectangle (e.g. 10,20,50,40)\n");
    printf("  -b, --bbox <x1,y1,x2,y2>      Bounding box in pixels (e.g. 100,200,900,800)\n\n");
    printf("Options:\n");
    printf("  -i, --input <dir_or_file>     Input directory or single image file\n");
    printf("  -o, --output <dir>            Output directory\n");
    printf("  -q, --quality <1-100>         JPEG output quality (default: 75, compact size)\n");
    printf("  -t, --threads <N>             Worker threads (default: CPU cores, max 4 on Pi Zero 2)\n");
    printf("  -r, --recursive               Process subdirectories (default: ENABLED)\n");
    printf("  -n, --no-recursive, --flat    Disable subdirectories processing (top-level only)\n");
    printf("  -s, --skip-existing           Skip files already present in output (resume)\n");
    printf("  -v, --verbose                 Enable detailed warning messages\n");
    printf("  -h, --help                    Show this help message\n\n");
    printf("Examples:\n");
    printf("  # Crop a 640x480 box across all folders and subfolders:\n");
    printf("  %s -i /media/photos -o /media/crops -c 100,50,640,480\n\n", progname);
    printf("  # Crop the center 50%% with quality 75 and 4 threads:\n");
    printf("  %s -i ./photos -o ./crops -p 25,25,50,50 -q 75 -t 4 -s\n\n", progname);
}

int main(int argc, char **argv) {
    int default_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (default_threads <= 0) default_threads = 4;
    if (default_threads > 4) default_threads = 4; // Pi Zero 2 has 4 cores
    g_config.num_threads = default_threads;

    static struct option long_options[] = {
        {"input",         required_argument, 0, 'i'},
        {"output",        required_argument, 0, 'o'},
        {"crop",          required_argument, 0, 'c'},
        {"crop-pct",      required_argument, 0, 'p'},
        {"bbox",          required_argument, 0, 'b'},
        {"quality",       required_argument, 0, 'q'},
        {"threads",       required_argument, 0, 't'},
        {"recursive",     no_argument,       0, 'r'},
        {"no-recursive",  no_argument,       0, 'n'},
        {"flat",          no_argument,       0, 'n'},
        {"skip-existing", no_argument,       0, 's'},
        {"verbose",       no_argument,       0, 'v'},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    bool has_crop = false;

    while ((opt = getopt_long(argc, argv, "i:o:c:p:b:q:t:rnsvh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'i':
                snprintf(g_config.in_dir, sizeof(g_config.in_dir), "%s", optarg);
                break;
            case 'o':
                snprintf(g_config.out_dir, sizeof(g_config.out_dir), "%s", optarg);
                break;
            case 'c':
                if (sscanf(optarg, "%d,%d,%d,%d",
                           &g_config.crop_x, &g_config.crop_y,
                           &g_config.crop_w, &g_config.crop_h) == 4) {
                    has_crop = true;
                    g_config.use_pct = false;
                } else {
                    fprintf(stderr, "Error: Invalid crop format '%s'. Expected x,y,w,h\n", optarg);
                    return 1;
                }
                break;
            case 'p':
                if (sscanf(optarg, "%f,%f,%f,%f",
                           &g_config.pct_x, &g_config.pct_y,
                           &g_config.pct_w, &g_config.pct_h) == 4) {
                    has_crop = true;
                    g_config.use_pct = true;
                } else {
                    fprintf(stderr, "Error: Invalid crop-pct format '%s'. Expected pct_x,pct_y,pct_w,pct_h\n", optarg);
                    return 1;
                }
                break;
            case 'b': {
                int x1, y1, x2, y2;
                if (sscanf(optarg, "%d,%d,%d,%d", &x1, &y1, &x2, &y2) == 4) {
                    g_config.crop_x = x1;
                    g_config.crop_y = y1;
                    g_config.crop_w = x2 - x1;
                    g_config.crop_h = y2 - y1;
                    has_crop = true;
                    g_config.use_pct = false;
                } else {
                    fprintf(stderr, "Error: Invalid bbox format '%s'. Expected x1,y1,x2,y2\n", optarg);
                    return 1;
                }
                break;
            }
            case 'q':
                g_config.quality = atoi(optarg);
                if (g_config.quality < 1) g_config.quality = 1;
                if (g_config.quality > 100) g_config.quality = 100;
                break;
            case 't':
                g_config.num_threads = atoi(optarg);
                if (g_config.num_threads < 1) g_config.num_threads = 1;
                if (g_config.num_threads > 16) g_config.num_threads = 16;
                break;
            case 'r':
                g_config.recursive = true;
                break;
            case 'n':
                g_config.recursive = false;
                break;
            case 's':
                g_config.skip_existing = true;
                break;
            case 'v':
                g_config.verbose = true;
                break;
            case 'h':
                print_help(argv[0]);
                return 0;
            default:
                print_help(argv[0]);
                return 1;
        }
    }

    if (strlen(g_config.in_dir) == 0 || strlen(g_config.out_dir) == 0 || !has_crop) {
        fprintf(stderr, "Error: --input, --output, and crop area (-c, -p, or -b) are required.\n\n");
        print_help(argv[0]);
        return 1;
    }

    // Strip trailing slashes
    size_t in_len = strlen(g_config.in_dir);
    if (in_len > 1 && g_config.in_dir[in_len - 1] == '/') g_config.in_dir[in_len - 1] = 0;
    size_t out_len = strlen(g_config.out_dir);
    if (out_len > 1 && g_config.out_dir[out_len - 1] == '/') g_config.out_dir[out_len - 1] = 0;

    // Check input path
    struct stat in_st;
    if (stat(g_config.in_dir, &in_st) != 0) {
        fprintf(stderr, "Error: Input path '%s' does not exist.\n", g_config.in_dir);
        return 1;
    }

    // Initialize mutexes and queue
    pthread_mutex_init(&g_queue.lock, NULL);
    pthread_cond_init(&g_queue.not_empty, NULL);
    pthread_cond_init(&g_queue.not_full, NULL);
    g_queue.head = 0;
    g_queue.tail = 0;
    g_queue.count = 0;
    g_queue.done = false;
    g_queue.active_workers = 0;
    g_queue.total_enqueued = 0;

    pthread_mutex_init(&g_stats.lock, NULL);
    memset(&g_stats, 0, sizeof(g_stats));

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    mkdir_p(g_config.out_dir);

    printf("=============================================================\n");
    printf(" Fast Picture Crop Tool (Raspberry Pi Zero 2 Optimized)\n");
    printf("=============================================================\n");
    printf(" Input:        %s\n", g_config.in_dir);
    printf(" Output:       %s\n", g_config.out_dir);
    if (g_config.use_pct) {
        printf(" Crop Region:  %.1f%%, %.1f%%, %.1f%%, %.1f%% (X, Y, W, H)\n",
               g_config.pct_x, g_config.pct_y, g_config.pct_w, g_config.pct_h);
    } else {
        printf(" Crop Region:  %d, %d, %d, %d (X, Y, W, H)\n",
               g_config.crop_x, g_config.crop_y, g_config.crop_w, g_config.crop_h);
    }
    printf(" Quality:      %d (optimized for low file size)\n", g_config.quality);
    printf(" Threads:      %d worker threads\n", g_config.num_threads);
    printf(" Recursive:    %s\n", g_config.recursive ? "Yes" : "No");
    printf(" Skip Exists:  %s\n", g_config.skip_existing ? "Yes" : "No");
    printf("=============================================================\n");

    // Spawn worker threads
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * g_config.num_threads);
    for (int i = 0; i < g_config.num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, NULL);
    }

    double t_start = get_time_sec();
    double t_last_report = t_start;

    // If input is a single file
    if (S_ISREG(in_st.st_mode)) {
        char *base = strrchr(g_config.in_dir, '/');
        const char *filename = base ? base + 1 : g_config.in_dir;
        char out_path[PATH_MAX_LEN];
        snprintf(out_path, sizeof(out_path), "%s/%s", g_config.out_dir, filename);
        queue_push(g_config.in_dir, out_path, in_st.st_size);
    } else {
        // Input is a directory
        scan_directory(g_config.in_dir, g_config.out_dir, "");
    }

    // Signal completion to workers
    pthread_mutex_lock(&g_queue.lock);
    g_queue.done = true;
    pthread_cond_broadcast(&g_queue.not_empty);
    pthread_mutex_unlock(&g_queue.lock);

    // Wait for all queued items to finish while reporting live progress
    while (true) {
        pthread_mutex_lock(&g_queue.lock);
        int remaining = g_queue.count;
        int active = g_queue.active_workers;
        uint64_t enqueued = g_queue.total_enqueued;
        pthread_mutex_unlock(&g_queue.lock);

        double now = get_time_sec();
        if (now - t_last_report >= 0.20) {
            pthread_mutex_lock(&g_stats.lock);
            uint64_t proc = g_stats.processed;
            uint64_t skip = g_stats.skipped;
            uint64_t fail = g_stats.failed;
            uint64_t bin = g_stats.bytes_in;
            uint64_t bout = g_stats.bytes_out;
            pthread_mutex_unlock(&g_stats.lock);

            double elapsed = now - t_start;
            double speed = elapsed > 0 ? (double)(proc + skip) / elapsed : 0;
            double mb_in = (double)bin / (1024.0 * 1024.0);
            double mb_out = (double)bout / (1024.0 * 1024.0);
            double reduction = mb_in > 0 ? (1.0 - (mb_out / mb_in)) * 100.0 : 0;

            if (enqueued > 0) {
                double pct = (double)(proc + skip + fail) / (double)enqueued * 100.0;
                printf("\r[Progress] %" PRIu64 "/%" PRIu64 " (%.1f%%) | %.1f img/s | In: %.1f MB -> Out: %.1f MB (-%.1f%%) | Errs: %" PRIu64 "   ",
                       proc + skip, enqueued, pct, speed, mb_in, mb_out, reduction, fail);
            } else {
                printf("\r[Progress] Done: %" PRIu64 " | %.1f img/s | In: %.1f MB -> Out: %.1f MB (-%.1f%%) | Errs: %" PRIu64 "   ",
                       proc + skip, speed, mb_in, mb_out, reduction, fail);
            }
            fflush(stdout);
            t_last_report = now;
        }

        if (remaining == 0 && active == 0) {
            break;
        }
        usleep(30000); // 30ms sleep
    }

    // Join all threads
    for (int i = 0; i < g_config.num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);

    double t_total = get_time_sec() - t_start;
    double avg_speed = t_total > 0 ? (double)(g_stats.processed + g_stats.skipped) / t_total : 0;
    double total_mb_in = (double)g_stats.bytes_in / (1024.0 * 1024.0);
    double total_mb_out = (double)g_stats.bytes_out / (1024.0 * 1024.0);
    double space_saved_pct = total_mb_in > 0 ? (1.0 - (total_mb_out / total_mb_in)) * 100.0 : 0.0;

    printf("\n\n=============================================================\n");
    printf(" Processing Summary\n");
    printf("=============================================================\n");
    printf(" Successfully Cropped:  %" PRIu64 " images\n", g_stats.processed);
    printf(" Skipped (Pre-existing):%" PRIu64 " images\n", g_stats.skipped);
    printf(" Failed / Corrupt:      %" PRIu64 " images\n", g_stats.failed);
    printf(" Elapsed Time:          %.2f seconds\n", t_total);
    printf(" Average Speed:         %.1f images/second\n", avg_speed);
    printf(" Total Input Size:      %.2f MB\n", total_mb_in);
    printf(" Total Output Size:     %.2f MB\n", total_mb_out);
    printf(" Storage Space Saved:   %.1f%%\n", space_saved_pct);
    printf("=============================================================\n");

    return g_stats.failed > 0 ? 2 : 0;
}
