#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <getopt.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <setjmp.h>
#include <math.h>
#include <ctype.h>

#if defined(__has_include)
#if __has_include(<jpeglib.h>)
#include <jpeglib.h>
#else
#include "jpeglib.h"
#endif
#else
#include "jpeglib.h"
#endif

#define PATH_MAX_LEN 2048
#define MAX_WINDOW_RADIUS 3
#define MAX_BUF_SIZE (2 * MAX_WINDOW_RADIUS + 1)

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
}

typedef struct {
    char path[PATH_MAX_LEN];
    char name[256];
} FileEntry;

typedef struct {
    FileEntry *items;
    size_t count;
    size_t capacity;
} FileList;

typedef struct {
    char path[PATH_MAX_LEN];
    char name[256];
    int width;
    int height;
    float luma;
    float sat_pct;
    float *hi; // size: width * height * 3
} FrameSlot;

typedef struct {
    char path[PATH_MAX_LEN];
    char name[256];
    char prev_path[PATH_MAX_LEN];
    char next_path[PATH_MAX_LEN];
    float score;
    int x;
    int y;
    float luma;
    char crop_file[PATH_MAX_LEN];
    char compare_file[PATH_MAX_LEN];
    char tag[128];
} EvalRecord;

typedef struct {
    char in_dir[PATH_MAX_LEN];
    char out_dir[PATH_MAX_LEN];
    bool recursive;
    int tile_size;
    int window_radius;
    int margin_left;
    int margin_top;
    int margin_right;
    int margin_bottom;
    float dark_threshold;
    float glare_max_sat;
    float min_score;
    bool use_fixed_score;
    float mad_multiplier;
    bool save_full;
    bool verbose;
    int quality;
} DetectConfig;

// Built-in 5x7 bitmap font for ASCII 32..95 (uppercase, digits, punctuation)
static const uint8_t FONT5X7[64][5] = {
    {0x00,0x00,0x00,0x00,0x00}, // ' '
    {0x00,0x00,0x5F,0x00,0x00}, // '!'
    {0x00,0x07,0x00,0x07,0x00}, // '"'
    {0x14,0x7F,0x14,0x7F,0x14}, // '#'
    {0x24,0x2A,0x7F,0x2A,0x12}, // '$'
    {0x23,0x13,0x08,0x64,0x62}, // '%'
    {0x36,0x49,0x55,0x22,0x50}, // '&'
    {0x00,0x05,0x03,0x00,0x00}, // '''
    {0x00,0x1C,0x22,0x41,0x00}, // '('
    {0x00,0x41,0x22,0x1C,0x00}, // ')'
    {0x14,0x08,0x3E,0x08,0x14}, // '*'
    {0x08,0x08,0x3E,0x08,0x08}, // '+'
    {0x00,0x50,0x30,0x00,0x00}, // ','
    {0x08,0x08,0x08,0x08,0x08}, // '-'
    {0x00,0x60,0x60,0x00,0x00}, // '.'
    {0x20,0x10,0x08,0x04,0x02}, // '/'
    {0x3E,0x51,0x49,0x45,0x3E}, // '0'
    {0x00,0x42,0x7F,0x40,0x00}, // '1'
    {0x42,0x61,0x51,0x49,0x46}, // '2'
    {0x21,0x41,0x45,0x4B,0x31}, // '3'
    {0x18,0x14,0x12,0x7F,0x10}, // '4'
    {0x27,0x45,0x45,0x45,0x39}, // '5'
    {0x3C,0x4A,0x49,0x49,0x30}, // '6'
    {0x01,0x71,0x09,0x05,0x03}, // '7'
    {0x36,0x49,0x49,0x49,0x36}, // '8'
    {0x06,0x49,0x49,0x29,0x1E}, // '9'
    {0x00,0x36,0x36,0x00,0x00}, // ':'
    {0x00,0x56,0x36,0x00,0x00}, // ';'
    {0x08,0x14,0x22,0x41,0x00}, // '<'
    {0x14,0x14,0x14,0x14,0x14}, // '='
    {0x00,0x41,0x22,0x14,0x08}, // '>'
    {0x02,0x01,0x51,0x09,0x06}, // '?'
    {0x32,0x49,0x79,0x41,0x3E}, // '@'
    {0x7E,0x11,0x11,0x11,0x7E}, // 'A'
    {0x7F,0x49,0x49,0x49,0x36}, // 'B'
    {0x3E,0x41,0x41,0x41,0x22}, // 'C'
    {0x7F,0x41,0x41,0x22,0x1C}, // 'D'
    {0x7F,0x49,0x49,0x49,0x41}, // 'E'
    {0x7F,0x09,0x09,0x09,0x01}, // 'F'
    {0x3E,0x41,0x49,0x49,0x7A}, // 'G'
    {0x7F,0x08,0x08,0x08,0x7F}, // 'H'
    {0x00,0x41,0x7F,0x41,0x00}, // 'I'
    {0x20,0x40,0x41,0x3F,0x01}, // 'J'
    {0x7F,0x08,0x14,0x22,0x41}, // 'K'
    {0x7F,0x40,0x40,0x40,0x40}, // 'L'
    {0x7F,0x02,0x0C,0x02,0x7F}, // 'M'
    {0x7F,0x04,0x08,0x10,0x7F}, // 'N'
    {0x3E,0x41,0x41,0x41,0x3E}, // 'O'
    {0x7F,0x09,0x09,0x09,0x06}, // 'P'
    {0x3E,0x41,0x51,0x21,0x5E}, // 'Q'
    {0x7F,0x09,0x19,0x29,0x46}, // 'R'
    {0x46,0x49,0x49,0x49,0x31}, // 'S'
    {0x01,0x01,0x7F,0x01,0x01}, // 'T'
    {0x3F,0x40,0x40,0x40,0x3F}, // 'U'
    {0x1F,0x20,0x40,0x20,0x1F}, // 'V'
    {0x3F,0x40,0x38,0x40,0x3F}, // 'W'
    {0x63,0x14,0x08,0x14,0x63}, // 'X'
    {0x07,0x08,0x70,0x08,0x07}, // 'Y'
    {0x61,0x51,0x49,0x45,0x43}, // 'Z'
    {0x00,0x7F,0x41,0x41,0x00}, // '['
    {0x02,0x04,0x08,0x10,0x20}, // '\'
    {0x00,0x41,0x41,0x7F,0x00}, // ']'
    {0x04,0x02,0x01,0x02,0x04}, // '^'
    {0x40,0x40,0x40,0x40,0x40}  // '_'
};

static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static bool has_jpeg_ext(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) return false;
    return (strcasecmp(dot, ".jpg") == 0 || strcasecmp(dot, ".jpeg") == 0);
}

static void filelist_init(FileList *fl) {
    fl->items = NULL;
    fl->count = 0;
    fl->capacity = 0;
}

static void filelist_push(FileList *fl, const char *path, const char *name) {
    if (fl->count == fl->capacity) {
        size_t new_cap = fl->capacity == 0 ? 512 : fl->capacity * 2;
        FileEntry *n = realloc(fl->items, new_cap * sizeof(FileEntry));
        if (!n) {
            fprintf(stderr, "Out of memory allocating file list\n");
            exit(1);
        }
        fl->items = n;
        fl->capacity = new_cap;
    }
    snprintf(fl->items[fl->count].path, PATH_MAX_LEN, "%s", path);
    snprintf(fl->items[fl->count].name, sizeof(fl->items[fl->count].name), "%s", name);
    fl->count++;
}

static int cmp_file_entry(const void *a, const void *b) {
    const FileEntry *fa = (const FileEntry *)a;
    const FileEntry *fb = (const FileEntry *)b;
    int c = strcmp(fa->name, fb->name);
    if (c != 0) return c;
    return strcmp(fa->path, fb->path);
}

static void scan_dir(const char *dir_path, bool recursive, FileList *fl) {
    DIR *d = opendir(dir_path);
    if (!d) return;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        char full[PATH_MAX_LEN];
        snprintf(full, sizeof(full), "%s/%s", dir_path, ent->d_name);

        struct stat st;
        if (stat(full, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            if (recursive) {
                scan_dir(full, recursive, fl);
            }
        } else if (S_ISREG(st.st_mode)) {
            if (has_jpeg_ext(ent->d_name)) {
                filelist_push(fl, full, ent->d_name);
            }
        }
    }
    closedir(d);
}

static int mkdir_p(const char *path) {
    char tmp[PATH_MAX_LEN];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len == 0) return 0;
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

static void box_blur_rgb_float(const float *src, float *dst, float *tmp, int W, int H, int radius) {
    if (radius <= 0) {
        memcpy(dst, src, (size_t)W * H * 3 * sizeof(float));
        return;
    }
    const float inv = 1.0f / (float)(2 * radius + 1);

    for (int y = 0; y < H; y++) {
        const float *row_in = src + (size_t)y * W * 3;
        float *row_out = tmp + (size_t)y * W * 3;
        float sr = row_in[0] * (radius + 1);
        float sg = row_in[1] * (radius + 1);
        float sb = row_in[2] * (radius + 1);
        for (int i = 1; i <= radius; i++) {
            int x = i < W ? i : W - 1;
            sr += row_in[x * 3 + 0];
            sg += row_in[x * 3 + 1];
            sb += row_in[x * 3 + 2];
        }
        for (int x = 0; x < W; x++) {
            row_out[x * 3 + 0] = sr * inv;
            row_out[x * 3 + 1] = sg * inv;
            row_out[x * 3 + 2] = sb * inv;

            int x_rem = x - radius;
            if (x_rem < 0) x_rem = 0;
            int x_add = x + radius + 1;
            if (x_add >= W) x_add = W - 1;

            sr += row_in[x_add * 3 + 0] - row_in[x_rem * 3 + 0];
            sg += row_in[x_add * 3 + 1] - row_in[x_rem * 3 + 1];
            sb += row_in[x_add * 3 + 2] - row_in[x_rem * 3 + 2];
        }
    }

    for (int x = 0; x < W; x++) {
        float sr = tmp[x * 3 + 0] * (radius + 1);
        float sg = tmp[x * 3 + 1] * (radius + 1);
        float sb = tmp[x * 3 + 2] * (radius + 1);
        for (int i = 1; i <= radius; i++) {
            int y = i < H ? i : H - 1;
            const float *p = tmp + ((size_t)y * W + x) * 3;
            sr += p[0];
            sg += p[1];
            sb += p[2];
        }
        for (int y = 0; y < H; y++) {
            float *out_p = dst + ((size_t)y * W + x) * 3;
            out_p[0] = sr * inv;
            out_p[1] = sg * inv;
            out_p[2] = sb * inv;

            int y_rem = y - radius;
            if (y_rem < 0) y_rem = 0;
            int y_add = y + radius + 1;
            if (y_add >= H) y_add = H - 1;

            const float *p_add = tmp + ((size_t)y_add * W + x) * 3;
            const float *p_rem = tmp + ((size_t)y_rem * W + x) * 3;
            sr += p_add[0] - p_rem[0];
            sg += p_add[1] - p_rem[1];
            sb += p_add[2] - p_rem[2];
        }
    }
}

static void gaussian_approx_rgb(float *inout, float *buf1, float *buf2, int W, int H, int radius) {
    box_blur_rgb_float(inout, buf1, buf2, W, H, radius);
    box_blur_rgb_float(buf1, inout, buf2, W, H, radius);
}

static uint8_t *load_jpeg_rgb(const char *path, int *out_w, int *out_h) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    jerr.pub.output_message = my_output_message;

    uint8_t *rgb = NULL;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        free(rgb);
        return NULL;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return NULL;
    }

    cinfo.out_color_space = JCS_RGB;
    cinfo.dct_method = JDCT_IFAST;
    jpeg_start_decompress(&cinfo);

    int W = (int)cinfo.output_width;
    int H = (int)cinfo.output_height;
    *out_w = W;
    *out_h = H;

    size_t row_stride = (size_t)W * 3;
    rgb = malloc(row_stride * H);
    if (!rgb) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return NULL;
    }

    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row_ptr = rgb + (size_t)cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
    return rgb;
}

static bool save_jpeg_rgb(const char *path, const uint8_t *rgb, int W, int H, int quality) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;

    struct jpeg_compress_struct cinfo;
    struct my_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    jerr.pub.output_message = my_output_message;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        fclose(fp);
        return false;
    }

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width = W;
    cinfo.image_height = H;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    cinfo.optimize_coding = TRUE;

    jpeg_start_compress(&cinfo, TRUE);
    size_t row_stride = (size_t)W * 3;
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row_ptr = (JSAMPROW)(rgb + (size_t)cinfo.next_scanline * row_stride);
        jpeg_write_scanlines(&cinfo, &row_ptr, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);
    return true;
}

static bool load_and_preprocess_slot(
    const FileEntry *fe,
    FrameSlot *slot,
    float **scratch1,
    float **scratch2,
    float **scratch3,
    size_t *scratch_cap,
    float dark_threshold,
    float glare_max_sat,
    int *status_out
) {
    int W = 0, H = 0;
    uint8_t *rgb = load_jpeg_rgb(fe->path, &W, &H);
    if (!rgb || W <= 0 || H <= 0) {
        free(rgb);
        *status_out = -1;
        return false;
    }

    size_t npix = (size_t)W * H;
    size_t nvals = npix * 3;

    double sum_all = 0.0;
    size_t sat_count = 0;
    for (size_t i = 0; i < nvals; i++) {
        uint8_t v = rgb[i];
        sum_all += v;
        if (v > 250) sat_count++;
    }

    float luma = (float)(sum_all / (double)nvals);
    float sat_pct = (float)((double)sat_count * 100.0 / (double)nvals);

    snprintf(slot->path, PATH_MAX_LEN, "%s", fe->path);
    snprintf(slot->name, sizeof(slot->name), "%s", fe->name);
    slot->width = W;
    slot->height = H;
    slot->luma = luma;
    slot->sat_pct = sat_pct;

    if (luma < dark_threshold) {
        free(rgb);
        *status_out = 1;
        return false;
    }
    if (glare_max_sat < 100.0f && sat_pct > glare_max_sat) {
        free(rgb);
        *status_out = 2;
        return false;
    }

    if (*scratch_cap < nvals) {
        free(*scratch1);
        free(*scratch2);
        free(*scratch3);
        *scratch1 = malloc(nvals * sizeof(float));
        *scratch2 = malloc(nvals * sizeof(float));
        *scratch3 = malloc(nvals * sizeof(float));
        if (!*scratch1 || !*scratch2 || !*scratch3) {
            fprintf(stderr, "Out of memory allocating scratch buffers\n");
            exit(1);
        }
        *scratch_cap = nvals;
    }

    free(slot->hi);
    slot->hi = malloc(nvals * sizeof(float));
    if (!slot->hi) {
        fprintf(stderr, "Out of memory allocating frame slot\n");
        exit(1);
    }

    for (size_t i = 0; i < nvals; i++) {
        slot->hi[i] = (float)rgb[i];
    }
    free(rgb);

    // 1. Small smoothing blur (radius 1, 2 passes)
    gaussian_approx_rgb(slot->hi, *scratch1, *scratch2, W, H, 1);

    // 2. Per-channel mean & std normalization
    double mean[3] = {0.0, 0.0, 0.0};
    for (size_t p = 0; p < npix; p++) {
        mean[0] += slot->hi[p * 3 + 0];
        mean[1] += slot->hi[p * 3 + 1];
        mean[2] += slot->hi[p * 3 + 2];
    }
    mean[0] /= (double)npix;
    mean[1] /= (double)npix;
    mean[2] /= (double)npix;

    double var[3] = {0.0, 0.0, 0.0};
    for (size_t p = 0; p < npix; p++) {
        double d0 = slot->hi[p * 3 + 0] - mean[0];
        double d1 = slot->hi[p * 3 + 1] - mean[1];
        double d2 = slot->hi[p * 3 + 2] - mean[2];
        var[0] += d0 * d0;
        var[1] += d1 * d1;
        var[2] += d2 * d2;
    }
    float inv_std[3] = {
        (float)(1.0 / (sqrt(var[0] / (double)npix) + 1e-6)),
        (float)(1.0 / (sqrt(var[1] / (double)npix) + 1e-6)),
        (float)(1.0 / (sqrt(var[2] / (double)npix) + 1e-6))
    };

    for (size_t p = 0; p < npix; p++) {
        for (int c = 0; c < 3; c++) {
            float v = (slot->hi[p * 3 + c] - (float)mean[c]) * inv_std[c];
            float scaled = v * 40.0f + 128.0f;
            if (scaled < 0.0f) scaled = 0.0f;
            if (scaled > 255.0f) scaled = 255.0f;
            slot->hi[p * 3 + c] = scaled;
        }
    }

    // 3. Low-frequency background via 2-pass radius-20 box blur (~25px Gaussian)
    float *lo = *scratch1;
    memcpy(lo, slot->hi, nvals * sizeof(float));
    gaussian_approx_rgb(lo, *scratch2, *scratch3, W, H, 20);

    // 4. High-pass residual
    for (size_t i = 0; i < nvals; i++) {
        slot->hi[i] -= lo[i];
    }

    *status_out = 0;
    return true;
}

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float compute_median(float *arr, size_t n) {
    if (n == 0) return 0.0f;
    qsort(arr, n, sizeof(float), cmp_float);
    if (n % 2 == 1) return arr[n / 2];
    return 0.5f * (arr[n / 2 - 1] + arr[n / 2]);
}

static float score_frame_c(
    const FrameSlot *center,
    const FrameSlot **neighbors,
    int num_neighbors,
    const DetectConfig *cfg,
    int *out_x,
    int *out_y
) {
    int W = center->width;
    int H = center->height;
    int T = cfg->tile_size;

    int x0 = cfg->margin_left < W - T ? cfg->margin_left : W - T;
    int y0 = cfg->margin_top < H - T ? cfg->margin_top : H - T;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;

    int x1 = W - cfg->margin_right;
    int y1 = H - cfg->margin_bottom;
    if (x1 < x0 + T) x1 = x0 + T;
    if (y1 < y0 + T) y1 = y0 + T;
    if (x1 > W) x1 = W;
    if (y1 > H) y1 = H;

    int ntx = (x1 - x0) / T;
    int nty = (y1 - y0) / T;
    if (ntx <= 0 || nty <= 0) {
        *out_x = 0;
        *out_y = 0;
        return 0.0f;
    }

    size_t ntiles = (size_t)ntx * nty;
    float *tiles = malloc(ntiles * sizeof(float));
    float *sorted = malloc(ntiles * sizeof(float));
    if (!tiles || !sorted) {
        fprintf(stderr, "Out of memory allocating tile buffers\n");
        exit(1);
    }

    const float inv_tile_area = 1.0f / (float)(T * T);

    for (int ty = 0; ty < nty; ty++) {
        for (int tx = 0; tx < ntx; tx++) {
            float sum_dmin = 0.0f;
            int py_start = y0 + ty * T;
            int px_start = x0 + tx * T;

            for (int dy = 0; dy < T; dy++) {
                int py = py_start + dy;
                size_t row_off = (size_t)py * W;
                for (int dx = 0; dx < T; dx++) {
                    int px = px_start + dx;
                    size_t idx = (row_off + px) * 3;
                    float cr = center->hi[idx + 0];
                    float cg = center->hi[idx + 1];
                    float cb = center->hi[idx + 2];

                    float min_diff = 1e30f;
                    for (int k = 0; k < num_neighbors; k++) {
                        if (neighbors[k]->width != W || neighbors[k]->height != H) continue;
                        const float *nh = neighbors[k]->hi + idx;
                        float d = fabsf(cr - nh[0]) + fabsf(cg - nh[1]) + fabsf(cb - nh[2]);
                        if (d < min_diff) min_diff = d;
                    }
                    if (min_diff < 1e29f) {
                        sum_dmin += min_diff;
                    }
                }
            }
            tiles[ty * ntx + tx] = sum_dmin * inv_tile_area;
        }
    }

    memcpy(sorted, tiles, ntiles * sizeof(float));
    float med = compute_median(sorted, ntiles);

    for (size_t i = 0; i < ntiles; i++) {
        sorted[i] = fabsf(tiles[i] - med);
    }
    float mad = compute_median(sorted, ntiles) + 1e-6f;

    float max_z = -1.0f;
    int best_tx = 0, best_ty = 0;
    for (int ty = 0; ty < nty; ty++) {
        for (int tx = 0; tx < ntx; tx++) {
            float z = (tiles[ty * ntx + tx] - med) / mad;
            if (z > max_z) {
                max_z = z;
                best_tx = tx;
                best_ty = ty;
            }
        }
    }

    free(tiles);
    free(sorted);

    *out_x = x0 + best_tx * T;
    *out_y = y0 + best_ty * T;
    return max_z;
}

// Drawing helpers on RGB buffers
static void draw_pixel(uint8_t *rgb, int W, int H, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    size_t idx = ((size_t)y * W + x) * 3;
    rgb[idx + 0] = r;
    rgb[idx + 1] = g;
    rgb[idx + 2] = b;
}

static void draw_rect_outline(uint8_t *rgb, int W, int H, int x0, int y0, int x1, int y1, int thickness, uint8_t r, uint8_t g, uint8_t b) {
    for (int t = 0; t < thickness; t++) {
        for (int x = x0 - t; x <= x1 + t; x++) {
            draw_pixel(rgb, W, H, x, y0 - t, r, g, b);
            draw_pixel(rgb, W, H, x, y1 + t, r, g, b);
        }
        for (int y = y0 - t; y <= y1 + t; y++) {
            draw_pixel(rgb, W, H, x0 - t, y, r, g, b);
            draw_pixel(rgb, W, H, x1 + t, y, r, g, b);
        }
    }
}

static void draw_text_5x7(uint8_t *rgb, int W, int H, int x, int y, const char *text, uint8_t r, uint8_t g, uint8_t b) {
    int cx = x;
    for (const char *p = text; *p; p++) {
        unsigned char c = (unsigned char)toupper((unsigned char)*p);
        if (c < 32 || c > 95) c = ' ';
        const uint8_t *glyph = FONT5X7[c - 32];
        for (int col = 0; col < 5; col++) {
            uint8_t bits = glyph[col];
            for (int row = 0; row < 7; row++) {
                if (bits & (1 << row)) {
                    draw_pixel(rgb, W, H, cx + col, y + row, r, g, b);
                }
            }
        }
        cx += 6;
    }
}

// Bilinear crop & upscale from source RGB image
static void crop_and_resize_bilinear(
    const uint8_t *src, int src_w, int src_h,
    int bx0, int by0, int bw, int bh,
    uint8_t *dst, int dst_w, int dst_h
) {
    for (int dy = 0; dy < dst_h; dy++) {
        float sy = by0 + ((dy + 0.5f) * bh / (float)dst_h) - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y1 < 0) y1 = 0;
        if (y0 >= src_h) y0 = src_h - 1;
        if (y1 >= src_h) y1 = src_h - 1;

        for (int dx = 0; dx < dst_w; dx++) {
            float sx = bx0 + ((dx + 0.5f) * bw / (float)dst_w) - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x1 < 0) x1 = 0;
            if (x0 >= src_w) x0 = src_w - 1;
            if (x1 >= src_w) x1 = src_w - 1;

            const uint8_t *p00 = src + ((size_t)y0 * src_w + x0) * 3;
            const uint8_t *p01 = src + ((size_t)y0 * src_w + x1) * 3;
            const uint8_t *p10 = src + ((size_t)y1 * src_w + x0) * 3;
            const uint8_t *p11 = src + ((size_t)y1 * src_w + x1) * 3;

            uint8_t *out = dst + ((size_t)dy * dst_w + dx) * 3;
            for (int c = 0; c < 3; c++) {
                float v0 = p00[c] * (1.0f - fx) + p01[c] * fx;
                float v1 = p10[c] * (1.0f - fx) + p11[c] * fx;
                int val = (int)(v0 * (1.0f - fy) + v1 * fy + 0.5f);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                out[c] = (uint8_t)val;
            }
        }
    }
}

static void extract_tag(const char *filename, char *out_tag, size_t max_len) {
    char stem[256];
    snprintf(stem, sizeof(stem), "%s", filename);
    char *dot = strrchr(stem, '.');
    if (dot) *dot = '\0';

    // Check if formatted like image_YYYYMMDD_HHMMSS_...
    if (strncmp(stem, "image_", 6) == 0 && strlen(stem) >= 21 && stem[14] == '_') {
        snprintf(out_tag, max_len, "%.15s", stem + 6);
    } else {
        snprintf(out_tag, max_len, "%s", stem);
    }
}

static void save_detection_visuals_c(EvalRecord *rec, const DetectConfig *cfg) {
    extract_tag(rec->name, rec->tag, sizeof(rec->tag));

    int W = 0, H = 0;
    uint8_t *rgb_c = load_jpeg_rgb(rec->path, &W, &H);
    if (!rgb_c) return;

    int crop_w = 176;
    int crop_h = 116;
    int zoom = 3;

    if (crop_w > W) crop_w = W;
    if (crop_h > H) crop_h = H;

    int cx = rec->x + cfg->tile_size / 2;
    int cy = rec->y + cfg->tile_size / 2;
    int bx0 = cx - crop_w / 2;
    int by0 = cy - crop_h / 2;
    if (bx0 < 0) bx0 = 0;
    if (by0 < 0) by0 = 0;
    if (bx0 + crop_w > W) bx0 = W - crop_w;
    if (by0 + crop_h > H) by0 = H - crop_h;

    int zw = crop_w * zoom;
    int zh = crop_h * zoom;

    uint8_t *crop_buf = malloc((size_t)zw * zh * 3);
    if (!crop_buf) {
        free(rgb_c);
        return;
    }

    // 1. Save single zoomed crop
    crop_and_resize_bilinear(rgb_c, W, H, bx0, by0, crop_w, crop_h, crop_buf, zw, zh);
    snprintf(rec->crop_file, PATH_MAX_LEN, "%.1800s/flag_%.120s.jpg", cfg->out_dir, rec->tag);
    save_jpeg_rgb(rec->crop_file, crop_buf, zw, zh, cfg->quality);

    // 2. Save 3-panel comparison strip [PREV | FLAGGED | NEXT]
    int header_h = 20;
    int strip_w = zw * 3 + 8;
    int strip_h = zh + header_h;
    uint8_t *strip = malloc((size_t)strip_w * strip_h * 3);
    if (strip) {
        memset(strip, 18, (size_t)strip_w * strip_h * 3);

        const char *paths[3] = { rec->prev_path, rec->path, rec->next_path };
        const char *prefixes[3] = { "PREV: ", "FLAGGED: ", "NEXT: " };

        for (int p_i = 0; p_i < 3; p_i++) {
            int px_off = p_i * (zw + 4);
            uint8_t *pan_rgb = NULL;
            int pw = 0, ph = 0;

            if (p_i == 1) {
                pan_rgb = rgb_c;
                pw = W;
                ph = H;
            } else if (paths[p_i][0] != '\0') {
                pan_rgb = load_jpeg_rgb(paths[p_i], &pw, &ph);
            }

            if (pan_rgb && pw == W && ph == H) {
                crop_and_resize_bilinear(pan_rgb, W, H, bx0, by0, crop_w, crop_h, crop_buf, zw, zh);
                for (int r = 0; r < zh; r++) {
                    memcpy(
                        strip + ((size_t)(header_h + r) * strip_w + px_off) * 3,
                        crop_buf + (size_t)r * zw * 3,
                        (size_t)zw * 3
                    );
                }
            }

            if (p_i != 1 && pan_rgb) {
                free(pan_rgb);
            }

            char lbl[128];
            const char *base = strrchr(paths[p_i], '/');
            base = base ? base + 1 : paths[p_i];
            char tag_p[64];
            if (base[0] != '\0') {
                extract_tag(base, tag_p, sizeof(tag_p));
            } else {
                snprintf(tag_p, sizeof(tag_p), "NONE");
            }
            snprintf(lbl, sizeof(lbl), "%s%s", prefixes[p_i], tag_p);

            if (p_i == 1) {
                draw_text_5x7(strip, strip_w, strip_h, px_off + 6, 6, lbl, 255, 90, 90);
                int rx0 = px_off + (rec->x - bx0) * zoom;
                int ry0 = header_h + (rec->y - by0) * zoom;
                int rx1 = rx0 + cfg->tile_size * zoom;
                int ry1 = ry0 + cfg->tile_size * zoom;
                draw_rect_outline(strip, strip_w, strip_h, rx0 - 4, ry0 - 4, rx1 + 4, ry1 + 4, 2, 255, 60, 60);

                char score_lbl[128];
                snprintf(score_lbl, sizeof(score_lbl), "SCORE=%.1f AT=(%d,%d)", rec->score, rec->x, rec->y);
                draw_text_5x7(strip, strip_w, strip_h, px_off + 8, strip_h - 12, score_lbl, 255, 235, 50);
            } else {
                draw_text_5x7(strip, strip_w, strip_h, px_off + 6, 6, lbl, 200, 200, 200);
            }
        }

        snprintf(rec->compare_file, PATH_MAX_LEN, "%.1800s/compare_%.120s.jpg", cfg->out_dir, rec->tag);
        save_jpeg_rgb(rec->compare_file, strip, strip_w, strip_h, cfg->quality);
        free(strip);
    }

    // 3. Optional full annotated image
    if (cfg->save_full) {
        draw_rect_outline(rgb_c, W, H, bx0, by0, bx0 + crop_w - 1, by0 + crop_h - 1, 2, 255, 50, 50);
        char full_path[PATH_MAX_LEN];
        snprintf(full_path, sizeof(full_path), "%.1800s/full_%.120s.jpg", cfg->out_dir, rec->tag);
        save_jpeg_rgb(full_path, rgb_c, W, H, cfg->quality);
    }

    free(crop_buf);
    free(rgb_c);
}

static bool build_contact_sheet_c(const EvalRecord *flagged, size_t n_flagged, const DetectConfig *cfg, char *out_sheet_path) {
    if (n_flagged == 0) return false;

    int cols = 5;
    int thumb_w = 352;
    int thumb_h = 232;
    int header_h = 20;
    int rows = (int)((n_flagged + cols - 1) / cols);

    int sheet_w = cols * thumb_w;
    int sheet_h = rows * (thumb_h + header_h);

    uint8_t *sheet = malloc((size_t)sheet_w * sheet_h * 3);
    uint8_t *thumb = malloc((size_t)thumb_w * thumb_h * 3);
    if (!sheet || !thumb) {
        free(sheet);
        free(thumb);
        return false;
    }
    memset(sheet, 12, (size_t)sheet_w * sheet_h * 3);

    for (size_t i = 0; i < n_flagged; i++) {
        int cw = 0, ch = 0;
        uint8_t *c_rgb = load_jpeg_rgb(flagged[i].crop_file, &cw, &ch);
        if (!c_rgb) continue;

        crop_and_resize_bilinear(c_rgb, cw, ch, 0, 0, cw, ch, thumb, thumb_w, thumb_h);
        free(c_rgb);

        int gx = (int)(i % cols) * thumb_w;
        int gy = (int)(i / cols) * (thumb_h + header_h);

        for (int r = 0; r < thumb_h; r++) {
            memcpy(
                sheet + ((size_t)(gy + header_h + r) * sheet_w + gx) * 3,
                thumb + (size_t)r * thumb_w * 3,
                (size_t)thumb_w * 3
            );
        }

        char lbl[128];
        snprintf(lbl, sizeof(lbl), "%s S=%.1f (%d,%d)", flagged[i].tag, flagged[i].score, flagged[i].x, flagged[i].y);
        draw_text_5x7(sheet, sheet_w, sheet_h, gx + 6, gy + 6, lbl, 255, 235, 60);
    }

    snprintf(out_sheet_path, PATH_MAX_LEN, "%.1900s/contact_sheet.jpg", cfg->out_dir);
    bool ok = save_jpeg_rgb(out_sheet_path, sheet, sheet_w, sheet_h, cfg->quality);
    free(thumb);
    free(sheet);
    return ok;
}

static int cmp_eval_desc(const void *a, const void *b) {
    const EvalRecord *ra = (const EvalRecord *)a;
    const EvalRecord *rb = (const EvalRecord *)b;
    if (rb->score > ra->score) return 1;
    if (rb->score < ra->score) return -1;
    return 0;
}

static void evaluate_center_slot(
    FrameSlot **ring_ptrs,
    int ring_len,
    int center_idx,
    const DetectConfig *cfg,
    EvalRecord **records,
    size_t *rec_count,
    size_t *rec_cap
) {
    const FrameSlot *center = ring_ptrs[center_idx];
    const FrameSlot *neighbors[MAX_BUF_SIZE];
    int n_nb = 0;
    for (int j = 0; j < ring_len; j++) {
        if (j != center_idx) {
            neighbors[n_nb++] = ring_ptrs[j];
        }
    }
    if (n_nb == 0) return;

    int bx = 0, by = 0;
    float score = score_frame_c(center, neighbors, n_nb, cfg, &bx, &by);

    if (*rec_count == *rec_cap) {
        size_t new_cap = *rec_cap == 0 ? 512 : (*rec_cap * 2);
        EvalRecord *n = realloc(*records, new_cap * sizeof(EvalRecord));
        if (!n) {
            fprintf(stderr, "Out of memory allocating evaluation records\n");
            exit(1);
        }
        *records = n;
        *rec_cap = new_cap;
    }

    EvalRecord *rec = &(*records)[*rec_count];
    memset(rec, 0, sizeof(EvalRecord));
    snprintf(rec->path, PATH_MAX_LEN, "%s", center->path);
    snprintf(rec->name, sizeof(rec->name), "%s", center->name);
    if (center_idx > 0) {
        snprintf(rec->prev_path, PATH_MAX_LEN, "%s", ring_ptrs[center_idx - 1]->path);
    }
    if (center_idx + 1 < ring_len) {
        snprintf(rec->next_path, PATH_MAX_LEN, "%s", ring_ptrs[center_idx + 1]->path);
    }
    rec->score = score;
    rec->x = bx;
    rec->y = by;
    rec->luma = center->luma;
    (*rec_count)++;

    if (cfg->verbose) {
        printf("  [Eval] %s  score=%7.1f  at=(%d,%d)  luma=%.1f\n",
               center->name, score, bx, by, center->luma);
    }
}

static void print_usage(const char *prog) {
    printf("Fast Transient Object / Anomaly Detector (Native C - Raspberry Pi Optimized)\n");
    printf("Usage: %s -i <input_dir> -o <output_dir> [options]\n\n", prog);
    printf("Required:\n");
    printf("  -i, --input <path>           Input directory of cropped JPEG images\n");
    printf("  -o, --output <path>          Output directory for flagged crops & reports\n\n");
    printf("Options:\n");
    printf("  -r, --recursive              Scan subdirectories recursively (default: enabled)\n");
    printf("  -n, --no-recursive, --flat   Disable recursive subdirectory scan\n");
    printf("      --tile-size <px>         Tile size in pixels for local scoring (default: 16)\n");
    printf("      --window <1..3>          Temporal neighbor radius (default: 2 -> 5-frame buffer)\n");
    printf("      --ignore-margins <L,T,R,B> Edge margins in px to ignore (default: 0,0,50,0)\n");
    printf("      --dark-threshold <val>   Skip night frames with mean luma < val (default: 40.0)\n");
    printf("      --glare-max-sat <pct>    Skip sun-glare frames with >pct%% pixels >250 (default: 15.0)\n");
    printf("      --min-score <val>        Fixed anomaly score cutoff (overrides adaptive MAD)\n");
    printf("      --mad-multiplier <k>     Adaptive cutoff: median + k * MAD (default: 4.0)\n");
    printf("  -q, --quality <1-100>        JPEG output quality for crops/sheets (default: 85)\n");
    printf("      --save-full              Also save full annotated frame for flagged detections\n");
    printf("  -v, --verbose                Print score and coordinates for every evaluated frame\n");
    printf("  -h, --help                   Display this help message\n");
}

int main(int argc, char **argv) {
    DetectConfig cfg = {
        .in_dir = "",
        .out_dir = "",
        .recursive = true,
        .tile_size = 16,
        .window_radius = 2,
        .margin_left = 0,
        .margin_top = 0,
        .margin_right = 50,
        .margin_bottom = 0,
        .dark_threshold = 40.0f,
        .glare_max_sat = 15.0f,
        .min_score = 0.0f,
        .use_fixed_score = false,
        .mad_multiplier = 4.0f,
        .save_full = false,
        .verbose = false,
        .quality = 85
    };

    static struct option long_opts[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"recursive", no_argument, 0, 'r'},
        {"no-recursive", no_argument, 0, 'n'},
        {"flat", no_argument, 0, 'n'},
        {"tile-size", required_argument, 0, 1001},
        {"window", required_argument, 0, 1002},
        {"ignore-margins", required_argument, 0, 1003},
        {"dark-threshold", required_argument, 0, 1004},
        {"glare-max-sat", required_argument, 0, 1005},
        {"min-score", required_argument, 0, 1006},
        {"mad-multiplier", required_argument, 0, 1007},
        {"save-full", no_argument, 0, 1008},
        {"quality", required_argument, 0, 'q'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:rnq:vh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'i':
                snprintf(cfg.in_dir, sizeof(cfg.in_dir), "%s", optarg);
                break;
            case 'o':
                snprintf(cfg.out_dir, sizeof(cfg.out_dir), "%s", optarg);
                break;
            case 'r':
                cfg.recursive = true;
                break;
            case 'n':
                cfg.recursive = false;
                break;
            case 'q':
                cfg.quality = atoi(optarg);
                if (cfg.quality < 1 || cfg.quality > 100) cfg.quality = 85;
                break;
            case 1001:
                cfg.tile_size = atoi(optarg);
                if (cfg.tile_size < 4) cfg.tile_size = 4;
                break;
            case 1002:
                cfg.window_radius = atoi(optarg);
                if (cfg.window_radius < 1) cfg.window_radius = 1;
                if (cfg.window_radius > MAX_WINDOW_RADIUS) cfg.window_radius = MAX_WINDOW_RADIUS;
                break;
            case 1003:
                if (sscanf(optarg, "%d,%d,%d,%d",
                           &cfg.margin_left, &cfg.margin_top, &cfg.margin_right, &cfg.margin_bottom) != 4) {
                    fprintf(stderr, "Error: --ignore-margins expects 4 integers: left,top,right,bottom\n");
                    return 1;
                }
                break;
            case 1004:
                cfg.dark_threshold = strtof(optarg, NULL);
                break;
            case 1005:
                cfg.glare_max_sat = strtof(optarg, NULL);
                break;
            case 1006:
                cfg.min_score = strtof(optarg, NULL);
                cfg.use_fixed_score = true;
                break;
            case 1007:
                cfg.mad_multiplier = strtof(optarg, NULL);
                break;
            case 1008:
                cfg.save_full = true;
                break;
            case 'v':
                cfg.verbose = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (cfg.in_dir[0] == '\0' || cfg.out_dir[0] == '\0') {
        fprintf(stderr, "Error: Both -i (--input) and -o (--output) are required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (mkdir_p(cfg.out_dir) != 0) {
        fprintf(stderr, "Error: Could not create output directory '%s'\n", cfg.out_dir);
        return 1;
    }

    double t_start = get_time_sec();

    FileList fl;
    filelist_init(&fl);

    struct stat st_in;
    if (stat(cfg.in_dir, &st_in) != 0) {
        fprintf(stderr, "Error: Input path '%s' does not exist.\n", cfg.in_dir);
        return 1;
    }
    if (S_ISREG(st_in.st_mode)) {
        const char *b = strrchr(cfg.in_dir, '/');
        filelist_push(&fl, cfg.in_dir, b ? b + 1 : cfg.in_dir);
    } else {
        scan_dir(cfg.in_dir, cfg.recursive, &fl);
    }

    if (fl.count == 0) {
        fprintf(stderr, "No supported JPEG images found in '%s'.\n", cfg.in_dir);
        free(fl.items);
        return 1;
    }

    qsort(fl.items, fl.count, sizeof(FileEntry), cmp_file_entry);

    int max_buf = 2 * cfg.window_radius + 1;

    printf("=============================================================\n");
    printf(" Fast Transient Object Detector (Native C Engine)\n");
    printf("=============================================================\n");
    printf(" Input Path:            %s\n", cfg.in_dir);
    printf(" Output Directory:      %s\n", cfg.out_dir);
    printf(" Total Images Found:    %zu\n", fl.count);
    printf(" Tile Size:             %dx%d px\n", cfg.tile_size, cfg.tile_size);
    printf(" Temporal Window:       +/- %d frames (%d-frame sliding buffer)\n", cfg.window_radius, max_buf);
    printf(" Ignored Margins:       L=%d, T=%d, R=%d, B=%d px\n",
           cfg.margin_left, cfg.margin_top, cfg.margin_right, cfg.margin_bottom);
    printf(" Dark Frame Cutoff:     mean luma < %.1f\n", cfg.dark_threshold);
    printf(" Glare Frame Cutoff:    saturated pixels > %.1f%%\n", cfg.glare_max_sat);
    printf("=============================================================\n\n");

    FrameSlot slots[MAX_BUF_SIZE];
    FrameSlot *ring_ptrs[MAX_BUF_SIZE];
    memset(slots, 0, sizeof(slots));
    for (int i = 0; i < max_buf; i++) {
        ring_ptrs[i] = &slots[i];
    }
    int ring_len = 0;

    float *scratch1 = NULL, *scratch2 = NULL, *scratch3 = NULL;
    size_t scratch_cap = 0;

    size_t dark_skipped = 0;
    size_t glare_skipped = 0;
    size_t corrupt_failed = 0;
    size_t valid_loaded = 0;

    EvalRecord *records = NULL;
    size_t rec_count = 0;
    size_t rec_cap = 0;

    double t_last = get_time_sec();

    for (size_t i = 0; i < fl.count; i++) {
        FrameSlot *target_slot;
        if (ring_len < max_buf) {
            target_slot = ring_ptrs[ring_len];
        } else {
            target_slot = ring_ptrs[0];
        }

        int status = 0;
        bool ok = load_and_preprocess_slot(
            &fl.items[i],
            target_slot,
            &scratch1, &scratch2, &scratch3, &scratch_cap,
            cfg.dark_threshold,
            cfg.glare_max_sat,
            &status
        );

        if (!ok) {
            if (status == 1) dark_skipped++;
            else if (status == 2) glare_skipped++;
            else corrupt_failed++;
        } else {
            if (ring_len < max_buf) {
                ring_len++;
            } else {
                // Rotate ring_ptrs left by 1 so target_slot moves to the end
                FrameSlot *first = ring_ptrs[0];
                for (int k = 0; k < max_buf - 1; k++) {
                    ring_ptrs[k] = ring_ptrs[k + 1];
                }
                ring_ptrs[max_buf - 1] = first;
            }
            valid_loaded++;

            if (valid_loaded == (size_t)(cfg.window_radius + 1)) {
                for (int c_i = 0; c_i < cfg.window_radius; c_i++) {
                    evaluate_center_slot(ring_ptrs, ring_len, c_i, &cfg, &records, &rec_count, &rec_cap);
                }
            }
            if (ring_len == max_buf) {
                evaluate_center_slot(ring_ptrs, ring_len, cfg.window_radius, &cfg, &records, &rec_count, &rec_cap);
            }
        }

        double now = get_time_sec();
        if (!cfg.verbose && (now - t_last >= 0.20 || i + 1 == fl.count)) {
            double elapsed = now - t_start;
            double spd = elapsed > 0.0 ? (double)(i + 1) / elapsed : 0.0;
            printf("\r[Scanning] %zu/%zu (%.1f img/s) | Valid: %zu | Dark: %zu | Glare: %zu   ",
                   i + 1, fl.count, spd, valid_loaded, dark_skipped, glare_skipped);
            fflush(stdout);
            t_last = now;
        }
    }

    // Flush trailing frames in buffer
    if (ring_len > 1) {
        int start_tail = ring_len - cfg.window_radius;
        if (start_tail < 0 || valid_loaded <= (size_t)cfg.window_radius) {
            start_tail = 0;
        }
        for (int c_i = start_tail; c_i < ring_len; c_i++) {
            evaluate_center_slot(ring_ptrs, ring_len, c_i, &cfg, &records, &rec_count, &rec_cap);
        }
    }
    printf("\n");

    for (int i = 0; i < max_buf; i++) {
        free(slots[i].hi);
    }
    free(scratch1);
    free(scratch2);
    free(scratch3);

    if (rec_count == 0) {
        printf("\nNo valid daytime frames remained to evaluate.\n");
        free(fl.items);
        free(records);
        return 0;
    }

    // Compute baseline median & MAD across frame scores
    float *score_arr = malloc(rec_count * sizeof(float));
    for (size_t i = 0; i < rec_count; i++) score_arr[i] = records[i].score;
    float med_s = compute_median(score_arr, rec_count);
    for (size_t i = 0; i < rec_count; i++) score_arr[i] = fabsf(records[i].score - med_s);
    float mad_s = compute_median(score_arr, rec_count);
    free(score_arr);

    float adaptive_thr = med_s + cfg.mad_multiplier * mad_s;
    if (adaptive_thr < 12.0f) adaptive_thr = 12.0f;
    float threshold = cfg.use_fixed_score ? cfg.min_score : adaptive_thr;

    // Write scores.csv
    char csv_path[PATH_MAX_LEN];
    snprintf(csv_path, sizeof(csv_path), "%.1900s/scores.csv", cfg.out_dir);
    FILE *f_csv = fopen(csv_path, "w");
    if (f_csv) {
        fprintf(f_csv, "filename,path,score,x,y,luma,flagged\n");
        for (size_t i = 0; i < rec_count; i++) {
            bool is_flag = records[i].score > threshold;
            fprintf(f_csv, "%s,\"%s\",%.2f,%d,%d,%.1f,%s\n",
                    records[i].name, records[i].path, records[i].score,
                    records[i].x, records[i].y, records[i].luma,
                    is_flag ? "YES" : "NO");
        }
        fclose(f_csv);
    }

    // Collect and sort flagged frames descending by score
    EvalRecord *flagged = malloc(rec_count * sizeof(EvalRecord));
    size_t n_flagged = 0;
    for (size_t i = 0; i < rec_count; i++) {
        if (records[i].score > threshold) {
            flagged[n_flagged++] = records[i];
        }
    }
    qsort(flagged, n_flagged, sizeof(EvalRecord), cmp_eval_desc);

    for (size_t i = 0; i < n_flagged; i++) {
        save_detection_visuals_c(&flagged[i], &cfg);
    }

    char sheet_path[PATH_MAX_LEN] = "";
    bool has_sheet = build_contact_sheet_c(flagged, n_flagged, &cfg, sheet_path);

    double t_total = get_time_sec() - t_start;
    double avg_speed = t_total > 0.0 ? (double)fl.count / t_total : 0.0;

    printf("\n=============================================================\n");
    printf(" Detection Summary\n");
    printf("=============================================================\n");
    printf(" Total Images Scanned:  %zu\n", fl.count);
    printf(" Evaluated (Daylight):  %zu\n", rec_count);
    printf(" Skipped (Dark/Night):  %zu\n", dark_skipped);
    printf(" Skipped (Sun Glare):   %zu\n", glare_skipped);
    printf(" Failed / Corrupt:      %zu\n", corrupt_failed);
    printf(" Baseline Score Median: %.1f (MAD: %.1f)\n", med_s, mad_s);
    printf(" Anomaly Threshold:     %.1f\n", threshold);
    printf(" Flagged Anomalies:     %zu frames\n", n_flagged);
    printf(" Elapsed Time:          %.2f seconds (%.1f img/s)\n", t_total, avg_speed);
    printf(" Scores CSV Saved:      %s\n", csv_path);
    if (has_sheet) {
        printf(" Contact Sheet Saved:   %s\n", sheet_path);
    }
    printf("=============================================================\n");

    if (n_flagged > 0) {
        printf("\nFlagged Frames (sorted by anomaly score):\n");
        printf("  %-28s %8s   %-12s %s\n", "Timestamp / File", "Score", "Location", "Comparison Strip");
        printf("  ------------------------------------------------------------------------------\n");
        for (size_t i = 0; i < n_flagged; i++) {
            char loc[32];
            snprintf(loc, sizeof(loc), "(%d,%d)", flagged[i].x, flagged[i].y);
            const char *cbase = strrchr(flagged[i].compare_file, '/');
            cbase = cbase ? cbase + 1 : flagged[i].compare_file;
            printf("  %-28s %8.1f   %-12s %s\n",
                   flagged[i].tag, flagged[i].score, loc, cbase);
        }
    } else {
        printf("\nNo transient objects or anomalies exceeded the threshold.\n");
    }

    free(flagged);
    free(records);
    free(fl.items);
    return 0;
}
