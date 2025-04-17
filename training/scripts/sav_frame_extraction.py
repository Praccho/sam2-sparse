# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

def get_args_parser():
    p = argparse.ArgumentParser(
        "[SA‑V Preprocessing] Extracting JPEG frames (local, no Slurm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # DATA
    p.add_argument("--sav-vid-dir", type=str, required=True,
                   help="Directory that contains SA‑V videos (recurses one level)")
    p.add_argument("--sav-frame-sample-rate", type=int, default=4,
                   help="Keep every N‑th frame")

    # OUTPUT
    p.add_argument("--output-dir", type=str, required=True,
                   help="Where to write the extracted JPEG frames")

    # PARALLELISM
    p.add_argument("--n-jobs", type=int, default=os.cpu_count(),
                   help="How many worker processes to use (≤ CPU cores)")

    return p

def decode_video(video_path: str):
    assert os.path.exists(video_path), f"{video_path} does not exist"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_frames(video_path: str, sample_rate: int):
    frames = decode_video(video_path)
    return frames[::sample_rate]


def process_single_video(video_path: str, sample_rate: int, save_root: str):
    frames = extract_frames(video_path, sample_rate)
    dst_dir = os.path.join(save_root, Path(video_path).stem)
    os.makedirs(dst_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        out_path = os.path.join(dst_dir, f"{idx * sample_rate:05d}.jpg")
        cv2.imwrite(out_path, frame)


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    mp4_files = sorted([str(p) for p in Path(args.sav_vid_dir).glob("*/*.mp4")])
    if not mp4_files:
        raise RuntimeError(f"No .mp4 files found under {args.sav_vid_dir}")

    print(f"Found {len(mp4_files)} videos. Processing with {args.n_jobs} workers.")

    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = [
            pool.submit(process_single_video, vid, args.sav_frame_sample_rate, args.output_dir)
            for vid in mp4_files
        ]

        for _ in tqdm.tqdm(as_completed(futures), total=len(futures), unit="video"):
            pass

    print(f"Finished. Extracted frames are in: {args.output_dir}")