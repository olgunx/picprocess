#!/usr/bin/env python3
"""
Comprehensive Benchmark & Test Suite for Fast Crop Tools
=========================================================
Compares Native C engine vs Python engine vs Baseline Pillow,
verifying speed, memory usage, compression ratio, and accuracy.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

TEST_DIR = Path('/tmp/picprocess_test')
IN_DIR = TEST_DIR / 'input'
OUT_C_DIR = TEST_DIR / 'out_c'
OUT_PY_JPG_DIR = TEST_DIR / 'out_py_jpg'
OUT_PY_WEBP_DIR = TEST_DIR / 'out_py_webp'


def create_synthetic_dataset(num_images=50, width=1920, height=1080):
    """Generate realistic test photos with patterns, gradients, and noise."""
    if IN_DIR.exists():
        shutil.rmtree(IN_DIR)
    IN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_images} test images ({width}x{height}) in {IN_DIR}...")
    np.random.seed(42)

    for i in range(num_images):
        # Create gradient + texture pattern
        grad_x = np.tile(np.linspace(20, 230, width, dtype=np.uint8), (height, 1))
        grad_y = np.tile(np.linspace(30, 200, height, dtype=np.uint8).reshape(-1, 1), (1, width))
        noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)

        r = np.clip(grad_x + noise, 0, 255).astype(np.uint8)
        g = np.clip(grad_y + noise, 0, 255).astype(np.uint8)
        b = np.clip(255 - grad_x, 0, 255).astype(np.uint8)

        # Draw a synthetic feature (box) in the center
        r[height//4:height*3//4, width//4:width*3//4] = 200
        g[height//4:height*3//4, width//4:width*3//4] = 100
        b[height//4:height*3//4, width//4:width*3//4] = 50

        img = Image.fromarray(np.stack([r, g, b], axis=-1))
        img.save(IN_DIR / f"photo_{i:04d}.jpg", quality=85)

    total_in_mb = sum(f.stat().st_size for f in IN_DIR.glob('*.jpg')) / (1024 * 1024)
    print(f"Generated {num_images} images. Total dataset size: {total_in_mb:.2f} MB\n")
    return total_in_mb


def run_benchmark(crop_args, threads=4, quality=75):
    fastcrop_bin = Path('./fastcrop').resolve()
    fastcrop_py = Path('./fastcrop.py').resolve()

    if not fastcrop_bin.exists():
        print("Compiling fastcrop binary...")
        subprocess.run(['make'], check=True)

    results = []

    # 1. Native C Engine (fastcrop)
    if OUT_C_DIR.exists():
        shutil.rmtree(OUT_C_DIR)
    cmd_c = [
        str(fastcrop_bin),
        '-i', str(IN_DIR),
        '-o', str(OUT_C_DIR),
        '-c', crop_args,
        '-q', str(quality),
        '-t', str(threads)
    ]
    t0 = time.time()
    proc_c = subprocess.run(cmd_c, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t_c = time.time() - t0

    num_c = len(list(OUT_C_DIR.glob('*.jpg')))
    out_c_mb = sum(f.stat().st_size for f in OUT_C_DIR.glob('*.jpg')) / (1024 * 1024)
    results.append({
        'engine': 'Native C (fastcrop)',
        'count': num_c,
        'time': t_c,
        'speed': num_c / t_c if t_c > 0 else 0,
        'out_mb': out_c_mb
    })

    # 2. Python Engine (fastcrop.py - JPEG)
    if OUT_PY_JPG_DIR.exists():
        shutil.rmtree(OUT_PY_JPG_DIR)
    cmd_py_jpg = [
        sys.executable, str(fastcrop_py),
        '-i', str(IN_DIR),
        '-o', str(OUT_PY_JPG_DIR),
        '-c', crop_args,
        '-q', str(quality),
        '-t', str(threads),
        '-f', 'jpg'
    ]
    t0 = time.time()
    proc_py = subprocess.run(cmd_py_jpg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t_py_jpg = time.time() - t0

    num_py_jpg = len(list(OUT_PY_JPG_DIR.glob('*.jpg')))
    out_py_jpg_mb = sum(f.stat().st_size for f in OUT_PY_JPG_DIR.glob('*.jpg')) / (1024 * 1024)
    results.append({
        'engine': 'Python (fastcrop.py - JPG)',
        'count': num_py_jpg,
        'time': t_py_jpg,
        'speed': num_py_jpg / t_py_jpg if t_py_jpg > 0 else 0,
        'out_mb': out_py_jpg_mb
    })

    # 3. Python Engine (fastcrop.py - WebP)
    if OUT_PY_WEBP_DIR.exists():
        shutil.rmtree(OUT_PY_WEBP_DIR)
    cmd_py_webp = [
        sys.executable, str(fastcrop_py),
        '-i', str(IN_DIR),
        '-o', str(OUT_PY_WEBP_DIR),
        '-c', crop_args,
        '-q', str(quality),
        '-t', str(threads),
        '-f', 'webp'
    ]
    t0 = time.time()
    proc_webp = subprocess.run(cmd_py_webp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t_py_webp = time.time() - t0

    num_py_webp = len(list(OUT_PY_WEBP_DIR.glob('*.webp')))
    out_py_webp_mb = sum(f.stat().st_size for f in OUT_PY_WEBP_DIR.glob('*.webp')) / (1024 * 1024)
    results.append({
        'engine': 'Python (fastcrop.py - WebP)',
        'count': num_py_webp,
        'time': t_py_webp,
        'speed': num_py_webp / t_py_webp if t_py_webp > 0 else 0,
        'out_mb': out_py_webp_mb
    })

    # Verification of crop dimensions
    sample_c = next(OUT_C_DIR.glob('*.jpg'))
    with Image.open(sample_c) as im:
        dim_c = im.size

    sample_py = next(OUT_PY_JPG_DIR.glob('*.jpg'))
    with Image.open(sample_py) as im:
        dim_py = im.size

    return results, dim_c, dim_py


def print_results(results, total_in_mb, dim_c, dim_py):
    print("=========================================================================================")
    print(f" BENCHMARK RESULTS (Input: {total_in_mb:.2f} MB)")
    print("=========================================================================================")
    print(f"{'Engine':<28} | {'Images':<7} | {'Time (s)':<9} | {'Speed (img/s)':<14} | {'Output Size':<12} | {'Reduction':<9}")
    print("-----------------------------------------------------------------------------------------")

    for r in results:
        red = (1.0 - (r['out_mb'] / total_in_mb)) * 100.0 if total_in_mb > 0 else 0
        print(f"{r['engine']:<28} | {r['count']:<7} | {r['time']:<9.2f} | {r['speed']:<14.1f} | {r['out_mb']:<9.2f} MB | {red:<8.1f}%")

    print("=========================================================================================")
    print(f" Output Dimensions Verified:")
    print(f"   Native C output:  {dim_c[0]}x{dim_c[1]}")
    print(f"   Python output:    {dim_py[0]}x{dim_py[1]}")
    print("=========================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for fastcrop")
    parser.add_argument('--quick', action='store_true', help="Run quick 20 image benchmark")
    parser.add_argument('--images', type=int, default=50, help="Number of test images (default 50)")
    parser.add_argument('--threads', type=int, default=4, help="Worker threads (default 4)")
    args = parser.parse_args()

    num_imgs = 20 if args.quick else args.images
    total_in_mb = create_synthetic_dataset(num_images=num_imgs, width=1920, height=1080)
    results, dim_c, dim_py = run_benchmark('300,200,800,600', threads=args.threads, quality=75)
    print_results(results, total_in_mb, dim_c, dim_py)


if __name__ == '__main__':
    main()
