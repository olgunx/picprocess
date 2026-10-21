#!/usr/bin/env python3
"""
Fast Transient Anomaly / Object Detector (`fastdetect.py`)
==========================================================
Ultra-lightweight, sliding-window anomaly and transient object detector
designed specifically for Raspberry Pi Zero / Zero 2 W (512MB RAM).

Detects humans, animals, vehicles, or stranger objects appearing across
time-lapse / surveillance image sequences without heavy ML frameworks.

Key Features:
- Sliding 5-frame ring buffer: O(1) memory (< 30MB RSS even on 100,000+ photos).
- Lighting & exposure normalization: immune to sunrise/sunset & cloud shifts.
- High-pass spatial filtering: removes broad lighting gradients across the scene.
- Temporal neighbor minimum-diff: suppresses slow drift & persistent wind motion
  while isolating transient objects that appear for 1-2 frames.
- Tile-based MAD z-score (16x16 tiles): catches tiny objects (e.g. 10x35 px human)
  that would otherwise vanish in whole-image averages.
- Automatic night/dark frame & sun-glare filtering.
- Saves zoomed side-by-side comparison strips (prev / flagged / next), highlighted
  crops, and summary contact sheets.
"""

import argparse
from collections import deque
import csv
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFilter

# Tolerate truncated images from slow camera writes/crashes
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def collect_images(input_path, recursive=True):
    """Collect and chronologically sort image files by filename."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = []
    if recursive:
        for root, _, filenames in os.walk(p):
            for fname in filenames:
                if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(Path(root) / fname)
    else:
        for item in p.iterdir():
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(item)

    # Sort primarily by filename (which embeds timestamp YYYYMMDD_HHMMSS), then full path
    files.sort(key=lambda x: (x.name, str(x)))
    return files


def parse_ignore_margins(margin_str):
    """Parse L,T,R,B pixel margins to ignore (e.g. '0,0,50,0')."""
    parts = [int(x.strip()) for x in margin_str.split(',')]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 margin values (left,top,right,bottom), got: '{margin_str}'")
    if any(v < 0 for v in parts):
        raise ValueError("Margins must be non-negative integers")
    return tuple(parts)  # (left, top, right, bottom)


def load_and_preprocess(filepath, blur_radius=1.5, bg_radius=25.0):
    """
    Load an image and compute lighting-normalized high-pass spatial features.

    Returns:
        dict with:
            'path': Path
            'luma': float (0-255 mean brightness)
            'sat_pct': float (percentage of near-saturated pixels > 250)
            'width': int
            'height': int
            'hi': np.ndarray (H, W, 3) float32 high-pass normalized array, or None if filtered
    """
    with Image.open(filepath) as im:
        im_rgb = im.convert('RGB')
        width, height = im_rgb.size
        raw = np.asarray(im_rgb, dtype=np.float32)

        luma = float(raw.mean())
        sat_pct = float((raw > 250.0).mean() * 100.0)

        # Smooth sensor/JPEG block noise
        if blur_radius > 0:
            im_blur = im_rgb.filter(ImageFilter.GaussianBlur(blur_radius))
            arr = np.asarray(im_blur, dtype=np.float32)
        else:
            arr = raw

    # Per-channel normalization removes global brightness & color cast changes
    mean = arr.mean(axis=(0, 1), keepdims=True)
    std = arr.std(axis=(0, 1), keepdims=True) + 1e-6
    norm = (arr - mean) / std

    # Convert to uint8 range to use PIL's fast C GaussianBlur for low-frequency background
    scaled = np.clip(norm * 40.0 + 128.0, 0.0, 255.0)
    u8 = scaled.astype(np.uint8)
    lo_im = Image.fromarray(u8).filter(ImageFilter.GaussianBlur(bg_radius))
    lo = np.asarray(lo_im, dtype=np.float32)

    # High-pass residual isolates local objects from broad lighting gradients
    hi = (scaled - lo).astype(np.float32)

    return {
        'path': filepath,
        'luma': luma,
        'sat_pct': sat_pct,
        'width': width,
        'height': height,
        'hi': hi,
    }


def score_frame(center_entry, neighbor_entries, tile_size=16, margins=(0, 0, 50, 0)):
    """
    Compare center_entry against temporal neighbors and compute tile MAD z-score.

    Returns:
        (max_z, best_x, best_y, tile_z_map)
    """
    hi_c = center_entry['hi']
    H, W, _ = hi_c.shape
    m_left, m_top, m_right, m_bottom = margins

    x0 = min(m_left, W - tile_size)
    y0 = min(m_top, H - tile_size)
    x1 = max(x0 + tile_size, W - m_right)
    y1 = max(y0 + tile_size, H - m_bottom)

    # Pixel must differ from ALL temporal neighbors -> isolates transient objects
    # and suppresses slow shadow movement or single-neighbor jitter
    diffs = [np.abs(hi_c[y0:y1, x0:x1] - nb['hi'][y0:y1, x0:x1]).sum(axis=2) for nb in neighbor_entries]
    dmin = np.min(diffs, axis=0)

    sub_h, sub_w = dmin.shape
    th = (sub_h // tile_size) * tile_size
    tw = (sub_w // tile_size) * tile_size
    if th < tile_size or tw < tile_size:
        raise ValueError(
            f"Active region ({sub_w}x{sub_h}) is smaller than tile size ({tile_size}x{tile_size}). "
            f"Reduce --ignore-margins or --tile-size."
        )

    tiles = dmin[:th, :tw].reshape(th // tile_size, tile_size, tw // tile_size, tile_size).mean(axis=(1, 3))
    med = float(np.median(tiles))
    mad = float(np.median(np.abs(tiles - med))) + 1e-6
    z_map = (tiles - med) / mad

    idx = int(z_map.argmax())
    ty, tx = divmod(idx, z_map.shape[1])
    max_z = float(z_map[ty, tx])

    best_x = x0 + tx * tile_size
    best_y = y0 + ty * tile_size
    return max_z, best_x, best_y


def save_detection_visuals(
    output_dir,
    center_path,
    prev_path,
    next_path,
    x,
    y,
    score,
    tile_size=16,
    crop_w=176,
    crop_h=116,
    zoom_scale=3,
    save_annotated_full=False,
):
    """
    Save zoomed crop and 3-panel temporal comparison (prev / flagged / next)
    for quick visual verification.
    """
    stem = center_path.stem
    # Extract compact timestamp if formatted like image_YYYYMMDD_HHMMSS_...
    parts = stem.split('_')
    if len(parts) >= 3 and len(parts[1]) == 8 and len(parts[2]) == 6:
        tag = f"{parts[1]}_{parts[2]}"
    else:
        tag = stem

    with Image.open(center_path) as im_c:
        im_c = im_c.convert('RGB')
        W, H = im_c.size

        # Center crop box around (x + tile_size/2, y + tile_size/2)
        cx = x + tile_size // 2
        cy = y + tile_size // 2
        bx0 = max(0, min(W - crop_w, cx - crop_w // 2))
        by0 = max(0, min(H - crop_h, cy - crop_h // 2))
        bx1 = min(W, bx0 + crop_w)
        by1 = min(H, by0 + crop_h)
        box = (bx0, by0, bx1, by1)

        zw = (bx1 - bx0) * zoom_scale
        zh = (by1 - by0) * zoom_scale

        # 1. Single zoomed crop (for contact sheet and quick view)
        crop_c = im_c.crop(box).resize((zw, zh), Image.LANCZOS)
        crop_path = output_dir / f"flag_{tag}.png"
        crop_c.save(crop_path)

        # 2. 3-panel temporal comparison: [prev | center (flagged) | next]
        panels = []
        labels = []
        for lbl, p in [('PREV', prev_path), ('FLAGGED', center_path), ('NEXT', next_path)]:
            if p is not None and p.exists():
                with Image.open(p) as im_p:
                    pan = im_p.convert('RGB').crop(box).resize((zw, zh), Image.LANCZOS)
                name_lbl = f"{lbl}: {p.stem}"
            else:
                pan = Image.new('RGB', (zw, zh), (30, 30, 30))
                name_lbl = f"{lbl}: (none)"
            panels.append(pan)
            labels.append(name_lbl)

        header_h = 22
        strip = Image.new('RGB', (zw * 3 + 8, zh + header_h), (15, 15, 15))
        draw = ImageDraw.Draw(strip)

        for idx, (pan, lbl_text) in enumerate(zip(panels, labels)):
            px = idx * (zw + 4)
            strip.paste(pan, (px, header_h))
            color = (255, 80, 80) if idx == 1 else (200, 200, 200)
            draw.text((px + 6, 4), lbl_text[:42], fill=color)
            if idx == 1:
                # Draw subtle target marker around the winning tile in center panel
                rx0 = (x - bx0) * zoom_scale
                ry0 = (y - by0) * zoom_scale
                rx1 = rx0 + tile_size * zoom_scale
                ry1 = ry0 + tile_size * zoom_scale
                draw.rectangle(
                    [px + rx0 - 4, header_h + ry0 - 4, px + rx1 + 4, header_h + ry1 + 4],
                    outline=(255, 60, 60),
                    width=2,
                )

        draw.text((zw + 10, zh + header_h - 16), f"score={score:.1f} at=({x},{y})", fill=(255, 230, 0))
        compare_path = output_dir / f"compare_{tag}.png"
        strip.save(compare_path)

        # 3. Optional full annotated image
        if save_annotated_full:
            full_annot = im_c.copy()
            d_full = ImageDraw.Draw(full_annot)
            d_full.rectangle([bx0, by0, bx1, by1], outline=(255, 50, 50), width=2)
            full_annot.save(output_dir / f"full_{tag}.jpg", quality=85, optimize=True)

    return crop_path, compare_path, tag


def build_contact_sheet(flagged_records, output_path, cols=5, thumb_w=352, thumb_h=232):
    """Build a contact sheet grid of all flagged crops sorted by score descending."""
    if not flagged_records:
        return None

    rows = (len(flagged_records) + cols - 1) // cols
    header_h = 20
    sheet = Image.new('RGB', (cols * thumb_w, rows * (thumb_h + header_h)), (10, 10, 10))
    draw = ImageDraw.Draw(sheet)

    for i, rec in enumerate(flagged_records):
        crop_path = rec['crop_path']
        if not crop_path or not crop_path.exists():
            continue
        gx = (i % cols) * thumb_w
        gy = (i // cols) * (thumb_h + header_h)
        with Image.open(crop_path) as im:
            thumb = im.convert('RGB').resize((thumb_w, thumb_h), Image.LANCZOS)
        sheet.paste(thumb, (gx, gy + header_h))
        label = f"{rec['tag']}  s={rec['score']:.1f} ({rec['x']},{rec['y']})"
        draw.text((gx + 6, gy + 3), label, fill=(255, 235, 60))

    sheet.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fast Transient Object / Anomaly Detector (Raspberry Pi Zero / Zero 2 Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a day folder or hour folder and save flagged crops + contact sheet:
  python3 fastdetect.py -i /path/to/crop_1013 -o ./detections_1013

  # Use a fixed score cutoff (e.g. 25.0) for high-confidence human/object alerts:
  python3 fastdetect.py -i /path/to/crop_1013/08 -o ./out_08 --min-score 25.0

  # Customize ignored edge margins (left,top,right,bottom) for moving bushes:
  python3 fastdetect.py -i /path/to/crop_1018 -o ./out_1018 --ignore-margins 0,0,50,0
        """,
    )

    parser.add_argument('-i', '--input', required=True, help='Input directory or image sequence root')
    parser.add_argument('-o', '--output', required=True, help='Output directory for flagged crops and reports')
    parser.add_argument(
        '-r', '--recursive', action='store_true', default=True,
        help='Recursively scan subdirectories in chronological order (default: True)'
    )
    parser.add_argument(
        '-n', '--no-recursive', '--flat', dest='recursive', action='store_false',
        help='Do not scan subdirectories recursively'
    )
    parser.add_argument(
        '--tile-size', type=int, default=16,
        help='Tile size in pixels for local anomaly scoring (default: 16)'
    )
    parser.add_argument(
        '--window', type=int, default=2, choices=[1, 2, 3],
        help='Temporal neighbor radius before & after each frame (default: 2 -> 5-frame window)'
    )
    parser.add_argument(
        '--ignore-margins', type=str, default='0,0,50,0',
        help='Pixel margins to ignore as left,top,right,bottom (default: 0,0,50,0 to mask right-edge foliage)'
    )
    parser.add_argument(
        '--dark-threshold', type=float, default=40.0,
        help='Skip dark/night frames with mean luminance below this value 0-255 (default: 40.0)'
    )
    parser.add_argument(
        '--glare-max-sat', type=float, default=15.0,
        help='Skip sun-glare washed-out frames where >N%% of pixels are saturated >250 (default: 15.0, 100 to disable)'
    )
    parser.add_argument(
        '--min-score', type=float, default=None,
        help='Fixed minimum anomaly score threshold (if omitted, uses adaptive median + k*MAD, min 15.0)'
    )
    parser.add_argument(
        '--mad-multiplier', type=float, default=4.0,
        help='Multiplier k for adaptive threshold: median + k * MAD (default: 4.0)'
    )
    parser.add_argument(
        '--save-full', action='store_true',
        help='Also save full-frame JPEG with bounding box for each flagged image'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print score for every evaluated frame'
    )

    args = parser.parse_args()

    try:
        margins = parse_ignore_margins(args.ignore_margins)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    all_files = collect_images(args.input, recursive=args.recursive)
    if not all_files:
        print(f"No supported images found in '{args.input}'.", file=sys.stderr)
        sys.exit(1)

    print("=============================================================")
    print(" Fast Transient Object Detector (Raspberry Pi Optimized)")
    print("=============================================================")
    print(f" Input Path:            {args.input}")
    print(f" Output Directory:      {args.output}")
    print(f" Total Images Found:    {len(all_files)}")
    print(f" Tile Size:             {args.tile_size}x{args.tile_size} px")
    print(f" Temporal Window:       +/- {args.window} frames ({2 * args.window + 1}-frame sliding buffer)")
    print(f" Ignored Margins:       L={margins[0]}, T={margins[1]}, R={margins[2]}, B={margins[3]} px")
    print(f" Dark Frame Cutoff:     mean luma < {args.dark_threshold:.1f}")
    print(f" Glare Frame Cutoff:    saturated pixels > {args.glare_max_sat:.1f}%")
    print("=============================================================\n")

    # Sliding window buffer of valid daytime frames: max size = 2 * window + 1
    win_radius = args.window
    max_buf = 2 * win_radius + 1
    ring = deque(maxlen=max_buf)

    dark_skipped = 0
    glare_skipped = 0
    corrupt_failed = 0
    evaluated_results = []  # list of dicts: path, prev_path, next_path, score, x, y, luma

    t_last = time.time()

    def evaluate_center(buf_list, center_idx):
        center = buf_list[center_idx]
        neighbors = [buf_list[j] for j in range(len(buf_list)) if j != center_idx]
        if not neighbors:
            return
        score, bx, by = score_frame(
            center, neighbors, tile_size=args.tile_size, margins=margins
        )
        prev_p = buf_list[center_idx - 1]['path'] if center_idx > 0 else None
        next_p = buf_list[center_idx + 1]['path'] if center_idx + 1 < len(buf_list) else None
        rec = {
            'path': center['path'],
            'prev_path': prev_p,
            'next_path': next_p,
            'score': score,
            'x': bx,
            'y': by,
            'luma': center['luma'],
        }
        evaluated_results.append(rec)
        if args.verbose:
            print(f"  [Eval] {center['path'].name}  score={score:7.1f}  at=({bx},{by})  luma={center['luma']:.1f}")

    valid_loaded = 0

    for idx, fpath in enumerate(all_files):
        try:
            entry = load_and_preprocess(fpath)
        except Exception as e:
            corrupt_failed += 1
            if args.verbose:
                sys.stderr.write(f"\n[Warning] Failed to load {fpath}: {e}\n")
            continue

        if entry['luma'] < args.dark_threshold:
            dark_skipped += 1
            continue

        if args.glare_max_sat < 100.0 and entry['sat_pct'] > args.glare_max_sat:
            glare_skipped += 1
            continue

        ring.append(entry)
        valid_loaded += 1

        buf_list = list(ring)
        # Handle leading boundary frames once we have at least win_radius + 1 frames
        if valid_loaded == win_radius + 1:
            for c_i in range(win_radius):
                evaluate_center(buf_list, c_i)

        # Normal steady-state: when buffer is full, evaluate the center element
        if len(buf_list) == max_buf:
            evaluate_center(buf_list, win_radius)

        now = time.time()
        if not args.verbose and (now - t_last >= 0.25 or idx + 1 == len(all_files)):
            elapsed = now - t_start
            spd = (idx + 1) / elapsed if elapsed > 0 else 0.0
            sys.stdout.write(
                f"\r[Scanning] {idx + 1}/{len(all_files)} ({spd:.1f} img/s) | "
                f"Valid: {valid_loaded} | Dark: {dark_skipped} | Glare: {glare_skipped}   "
            )
            sys.stdout.flush()
            t_last = now

    # Flush remaining trailing frames in the buffer
    buf_list = list(ring)
    if len(buf_list) > 1:
        start_tail = max(0, len(buf_list) - win_radius)
        if valid_loaded <= win_radius:
            start_tail = 0
        for c_i in range(start_tail, len(buf_list)):
            evaluate_center(buf_list, c_i)

    ring.clear()
    print()

    if not evaluated_results:
        print("\nNo valid daytime frames remained to evaluate.")
        sys.exit(0)

    scores = np.array([r['score'] for r in evaluated_results], dtype=np.float64)
    med_s = float(np.median(scores))
    mad_s = float(np.median(np.abs(scores - med_s)))
    adaptive_thr = max(12.0, med_s + args.mad_multiplier * mad_s)
    threshold = args.min_score if args.min_score is not None else adaptive_thr

    # Sort flagged results by score descending
    flagged = [r for r in evaluated_results if r['score'] > threshold]
    flagged.sort(key=lambda r: r['score'], reverse=True)

    # Save CSV of all evaluated frames
    csv_path = out_dir / "scores.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['filename', 'path', 'score', 'x', 'y', 'luma', 'flagged'])
        for r in evaluated_results:
            is_flag = r['score'] > threshold
            writer.writerow([
                r['path'].name,
                str(r['path']),
                f"{r['score']:.2f}",
                r['x'],
                r['y'],
                f"{r['luma']:.1f}",
                'YES' if is_flag else 'NO',
            ])

    # Generate visual crops and comparison strips for flagged frames
    for r in flagged:
        crop_p, comp_p, tag = save_detection_visuals(
            out_dir,
            r['path'],
            r['prev_path'],
            r['next_path'],
            r['x'],
            r['y'],
            r['score'],
            tile_size=args.tile_size,
            save_annotated_full=args.save_full,
        )
        r['crop_path'] = crop_p
        r['compare_path'] = comp_p
        r['tag'] = tag

    sheet_path = build_contact_sheet(flagged, out_dir / "contact_sheet.png") if flagged else None

    t_total = time.time() - t_start
    avg_speed = len(all_files) / t_total if t_total > 0 else 0.0

    print("\n=============================================================")
    print(" Detection Summary")
    print("=============================================================")
    print(f" Total Images Scanned:  {len(all_files)}")
    print(f" Evaluated (Daylight):  {len(evaluated_results)}")
    print(f" Skipped (Dark/Night):  {dark_skipped}")
    print(f" Skipped (Sun Glare):   {glare_skipped}")
    print(f" Failed / Corrupt:      {corrupt_failed}")
    print(f" Baseline Score Median: {med_s:.1f} (MAD: {mad_s:.1f})")
    print(f" Anomaly Threshold:     {threshold:.1f}")
    print(f" Flagged Anomalies:     {len(flagged)} frames")
    print(f" Elapsed Time:          {t_total:.2f} seconds ({avg_speed:.1f} img/s)")
    print(f" Scores CSV Saved:      {csv_path}")
    if sheet_path:
        print(f" Contact Sheet Saved:   {sheet_path}")
    print("=============================================================")

    if flagged:
        print("\nFlagged Frames (sorted by anomaly score):")
        print(f"  {'Timestamp / File':<28} {'Score':>8}   {'Location':<12} {'Comparison Strip'}")
        print("  " + "-" * 78)
        for r in flagged:
            loc = f"({r['x']},{r['y']})"
            print(f"  {r['tag']:<28} {r['score']:8.1f}   {loc:<12} {r['compare_path'].name}")
    else:
        print("\nNo transient objects or anomalies exceeded the threshold.")


if __name__ == '__main__':
    main()
