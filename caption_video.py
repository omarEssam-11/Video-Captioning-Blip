"""
Video captioning pipeline using the BLIP model (CLI).

Workflow:
  1. Get the video (download from a link, or use a local file)
  2. Cut the video into frames
  3. Caption the frames with BLIP
  4. Merge the frames + captions back into a video

Usage:
  python caption_video.py path_or_url -o output.mp4
  python caption_video.py https://www.youtube.com/watch?v=... -o output.mp4 --fps 1
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch

from app.pipeline import (DEFAULT_MODEL, extract_frames, caption_frames,
                         get_video, load_captioner, merge_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption video frames with BLIP.")
    parser.add_argument("source", help="Video file path or URL (YouTube, direct link, ...)")
    parser.add_argument("-o", "--output", default="captioned_video.mp4",
                        help="Output video path (default: captioned_video.mp4)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to sample (default: 1)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"BLIP model on Hugging Face (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference (default: cuda if available, else cpu)")
    args = parser.parse_args()

    output_path = Path(args.output)
    work_dir = Path(tempfile.mkdtemp(prefix="video_caption_"))

    # 1. Get the video
    video_path = get_video(args.source, work_dir)
    print(f"[1/4] Video ready: {video_path}")

    # 2. Cut into frames
    frame_paths, _, size = extract_frames(video_path, work_dir / "frames", args.fps)
    print(f"[2/4] Extracted {len(frame_paths)} frames")

    # 3. Caption frames
    processor, model = load_captioner(args.model, args.device)
    print(f"[3/4] Captioning {len(frame_paths)} frames on {args.device}...")
    captions = caption_frames(frame_paths, processor, model, args.device)

    # 4. Merge back
    merge_frames(frame_paths, captions, output_path, args.fps, size)
    print(f"[4/4] Wrote captioned video to {output_path}")

    # Timestamped caption log
    sidecar = output_path.with_suffix(".txt")
    with open(sidecar, "w", encoding="utf-8") as f:
        for i, cap in enumerate(captions):
            f.write(f"{i / args.fps:8.2f}s  {cap}\n")
    print(f"\nDone! Captioned video: {output_path}")
    print(f"         Caption log:   {sidecar}")


if __name__ == "__main__":
    main()
