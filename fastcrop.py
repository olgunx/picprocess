#!/usr/bin/env python3
"""
Fast Picture Crop Tool (Raspberry Pi Zero 2 Optimized)
======================================================
High-throughput, low-memory batch image cropping tool designed specifically
for hardware-constrained environments like the Raspberry Pi Zero 2 (512MB RAM, quad-core).

Features:
- Multi-threaded processing using ThreadPoolExecutor (Pillow's C codecs release the GIL).
- Minimal memory footprint: single process (~25MB RSS), safe on 512MB RAM.
- Strips EXIF metadata, ICC profiles, and thumbnails to minimize output file size.
- Configurable quality (default 75) + Huffman table optimization.
- Supports WebP output (--format webp) for maximum compression ratio.
- Supports pixel coordinates, percentage coordinates, and bounding boxes.
- Resumable (--skip-existing) and recursive directory support.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from PIL import Image, ImageFile

# Tolerate truncated images from slow camera writes/crashes
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def parse_crop_spec(crop_str, is_pct=False, is_bbox=False):
    """Parse comma-separated crop coordinates."""
    parts = [p.strip().rstrip('%') for p in crop_str.split(',')]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 values, got {len(parts)}: '{crop_str}'")

    if is_pct:
        vals = [float(p) for p in parts]
        return ('pct', vals[0], vals[1], vals[2], vals[3])
    elif is_bbox:
        vals = [int(p) for p in parts]
        # x1, y1, x2, y2 -> x, y, w, h
        return ('pixel', vals[0], vals[1], vals[2] - vals[0], vals[3] - vals[1])
    else:
        vals = [int(p) for p in parts]
        return ('pixel', vals[0], vals[1], vals[2], vals[3])


def calculate_pixel_box(im_width, im_height, crop_spec):
    """Convert crop specification to PIL bounding box (left, upper, right, lower)."""
    mode = crop_spec[0]
    if mode == 'pct':
        _, px, py, pw, ph = crop_spec
        x = int(px * im_width / 100.0)
        y = int(py * im_height / 100.0)
        w = int(pw * im_width / 100.0)
        h = int(ph * im_height / 100.0)
    else:
        _, x, y, w, h = crop_spec

    # Clamp bounds to image dimensions
    x = max(0, min(x, im_width - 1))
    y = max(0, min(y, im_height - 1))
    w = max(1, min(w, im_width - x))
    h = max(1, min(h, im_height - y))

    return (x, y, x + w, y + h)


def process_image(in_path, out_path, crop_spec, quality, out_format, skip_existing):
    """Worker function to crop and save a single image with size optimizations."""
    try:
        if skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return ('skipped', os.path.getsize(in_path), os.path.getsize(out_path))

        in_size = os.path.getsize(in_path)

        with Image.open(in_path) as im:
            # Calculate crop coordinates based on image dimensions
            box = calculate_pixel_box(im.width, im.height, crop_spec)
            cropped = im.crop(box)

            # Determine output format
            target_fmt = out_format
            if not target_fmt:
                target_fmt = 'JPEG' if im.format in ('JPEG', 'MPO') else im.format
                if not target_fmt:
                    target_fmt = 'JPEG'

            # Ensure output directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Save with maximum file size optimizations
            # Converting RGBA to RGB for JPEG if needed
            save_kwargs = {}
            if target_fmt.upper() in ('JPEG', 'JPG'):
                target_fmt = 'JPEG'
                if cropped.mode in ('RGBA', 'LA', 'P'):
                    cropped = cropped.convert('RGB')
                save_kwargs = {
                    'quality': quality,
                    'optimize': True,
                    'subsampling': 2  # 4:2:0 subsampling for smallest size
                }
            elif target_fmt.upper() == 'WEBP':
                save_kwargs = {
                    'quality': quality,
                    'method': 4,      # Good compression vs speed trade-off
                }
            elif target_fmt.upper() == 'PNG':
                save_kwargs = {
                    'optimize': True,
                    'compress_level': 6
                }

            # Strip all metadata/EXIF by saving clean image object
            cropped.save(out_path, format=target_fmt, **save_kwargs)

        out_size = os.path.getsize(out_path)
        return ('success', in_size, out_size)
    except Exception as exc:
        return ('error', 0, 0, str(exc))


def find_images(input_path, output_path=None, recursive=True):
    """Yield image file paths lazily to keep memory minimal."""
    path = Path(input_path).resolve()
    out = Path(output_path).resolve() if output_path else None
    if path.is_file():
        yield path
        return

    if recursive:
        for root, dirs, files in os.walk(path):
            # Avoid scanning output directory if placed inside input
            if out:
                dirs[:] = [d for d in dirs if (Path(root) / d).resolve() != out]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    yield Path(root) / f
    else:
        for entry in os.scandir(path):
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    yield Path(entry.path)


def main():
    cpu_count = os.cpu_count() or 4
    default_threads = min(cpu_count, 4)  # Safe cap for Pi Zero 2

    parser = argparse.ArgumentParser(
        description="Fast Picture Crop Tool for Raspberry Pi Zero 2 (Low-RAM, Multi-core)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop 640x480 box across all folders and subfolders:
  %(prog)s -i /media/photos -o /media/crops -c 100,50,640,480

  # Crop center 50%% as WebP (super low file size) using 4 threads:
  %(prog)s -i ./photos -o ./crops -p 25,25,50,50 --format webp -q 75 -t 4 -s
        """
    )
    parser.add_argument('-i', '--input', required=True, help="Input directory or single image file")
    parser.add_argument('-o', '--output', required=True, help="Output directory")

    crop_group = parser.add_mutually_exclusive_group(required=True)
    crop_group.add_argument('-c', '--crop', help="Crop rectangle in pixels: X,Y,W,H")
    crop_group.add_argument('-p', '--crop-pct', help="Crop rectangle in percentages: X,Y,W,H (e.g. 10,20,50,40)")
    crop_group.add_argument('-b', '--bbox', help="Bounding box in pixels: X1,Y1,X2,Y2")

    parser.add_argument('-q', '--quality', type=int, default=75, help="Output quality 1-100 (default: 75, compact size)")
    parser.add_argument('-t', '--threads', type=int, default=default_threads, help=f"Worker threads (default: {default_threads})")
    parser.add_argument('-f', '--format', choices=['jpg', 'jpeg', 'webp', 'png', 'same'], default=None,
                        help="Output image format (webp yields smallest file sizes)")
    parser.add_argument('-r', '--recursive', action='store_true', default=True, help="Process subdirectories (default: True)")
    parser.add_argument('-n', '--no-recursive', '--flat', action='store_false', dest='recursive',
                        help="Disable subdirectories (top-level only)")
    parser.add_argument('-s', '--skip-existing', action='store_true', help="Skip files already present in output (resume)")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print verbose error messages")

    args = parser.parse_args()

    # Parse crop coordinates
    if args.crop:
        crop_spec = parse_crop_spec(args.crop, is_pct=False, is_bbox=False)
    elif args.crop_pct:
        crop_spec = parse_crop_spec(args.crop_pct, is_pct=True, is_bbox=False)
    elif args.bbox:
        crop_spec = parse_crop_spec(args.bbox, is_pct=False, is_bbox=True)

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_format = None
    if args.format and args.format != 'same':
        out_format = 'JPEG' if args.format in ('jpg', 'jpeg') else args.format.upper()

    print("=============================================================")
    print(" Fast Picture Crop Tool (Python Engine)")
    print("=============================================================")
    print(f" Input:        {in_root}")
    print(f" Output:       {out_root}")
    if crop_spec[0] == 'pct':
        print(f" Crop Region:  {crop_spec[1]}%, {crop_spec[2]}%, {crop_spec[3]}%, {crop_spec[4]}% (X, Y, W, H)")
    else:
        print(f" Crop Region:  {crop_spec[1]}, {crop_spec[2]}, {crop_spec[3]}, {crop_spec[4]} (X, Y, W, H)")
    print(f" Quality:      {args.quality} (low-size optimized)")
    print(f" Output Format:{out_format or 'Source format'}")
    print(f" Threads:      {args.threads} worker threads")
    print(f" Recursive:    {'Yes' if args.recursive else 'No'}")
    print(f" Skip Exists:  {'Yes' if args.skip_existing else 'No'}")
    print("=============================================================")

    # Stream tasks to ThreadPoolExecutor
    t_start = time.time()
    t_last_report = t_start

    processed = 0
    skipped = 0
    failed = 0
    bytes_in = 0
    bytes_out = 0

    # Build work generator
    def task_generator():
        for in_file in find_images(in_root, output_path=out_root, recursive=args.recursive):
            if in_root.is_file():
                rel = in_file.name
            else:
                rel = in_file.relative_to(in_root)

            out_file = out_root / rel
            if out_format:
                ext_map = {'JPEG': '.jpg', 'WEBP': '.webp', 'PNG': '.png'}
                out_file = out_file.with_suffix(ext_map.get(out_format, out_file.suffix))

            yield (str(in_file), str(out_file))

    # Bounded submission to keep memory strictly under control
    max_in_flight = args.threads * 64

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = set()
        task_gen = task_generator()

        exhausted = False
        while not exhausted or futures:
            # Top up futures queue
            while not exhausted and len(futures) < max_in_flight:
                try:
                    src, dst = next(task_gen)
                    f = executor.submit(process_image, src, dst, crop_spec,
                                        args.quality, out_format, args.skip_existing)
                    futures.add(f)
                except StopIteration:
                    exhausted = True
                    break

            if not futures:
                break

            # Wait for at least one future to complete
            done, not_done = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)

            for f in done:
                futures.remove(f)
                res = f.result()
                status = res[0]
                if status == 'success':
                    processed += 1
                    bytes_in += res[1]
                    bytes_out += res[2]
                elif status == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    if args.verbose and len(res) > 3:
                        sys.stderr.write(f"\n[Warning] {res[3]}\n")

            now = time.time()
            if now - t_last_report >= 0.25:
                elapsed = now - t_start
                speed = (processed + skipped) / elapsed if elapsed > 0 else 0
                mb_in = bytes_in / (1024 * 1024)
                mb_out = bytes_out / (1024 * 1024)
                reduction = (1.0 - (mb_out / mb_in)) * 100.0 if mb_in > 0 else 0.0
                sys.stdout.write(
                    f"\r[Progress] Done: {processed + skipped} | {speed:.1f} img/s | "
                    f"In: {mb_in:.1f} MB -> Out: {mb_out:.1f} MB (-{reduction:.1f}%) | Errs: {failed}   "
                )
                sys.stdout.flush()
                t_last_report = now

    t_total = time.time() - t_start
    avg_speed = (processed + skipped) / t_total if t_total > 0 else 0
    total_mb_in = bytes_in / (1024 * 1024)
    total_mb_out = bytes_out / (1024 * 1024)
    space_saved = (1.0 - (total_mb_out / total_mb_in)) * 100.0 if total_mb_in > 0 else 0.0

    print("\n\n=============================================================")
    print(" Processing Summary")
    print("=============================================================")
    print(f" Successfully Cropped:  {processed} images")
    print(f" Skipped (Pre-existing):{skipped} images")
    print(f" Failed / Corrupt:      {failed} images")
    print(f" Elapsed Time:          {t_total:.2f} seconds")
    print(f" Average Speed:         {avg_speed:.1f} images/second")
    print(f" Total Input Size:      {total_mb_in:.2f} MB")
    print(f" Total Output Size:     {total_mb_out:.2f} MB")
    print(f" Storage Space Saved:   {space_saved:.1f}%")
    print("=============================================================")


if __name__ == '__main__':
    main()
