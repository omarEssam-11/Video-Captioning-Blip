"""
Shared video-captioning pipeline: download -> frames -> BLIP captions -> merge.

Used by both the CLI (caption_video.py) and the deployed app (app/main.py).
"""

import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipForConditionalGeneration, BlipProcessor

DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_VIDEO_SECONDS = 30 * 60  # safety limit to protect disk space


# ---------------------------------------------------------------------------
# Step 1: get the video (download from link, or verify local file)
# ---------------------------------------------------------------------------

def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def looks_like_video_url(source: str) -> bool:
    path = urllib.parse.urlparse(source).path.lower()
    return Path(path).suffix in VIDEO_EXTS


def download_with_ytdlp(url: str, dest_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is required for this link. Install it: pip install yt-dlp")

    out_template = str(dest_dir / "source.%(ext)s")
    ydl_opts = {
        "outtmpl": out_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded = list(dest_dir.glob("source.*"))
    if not downloaded:
        raise RuntimeError("yt-dlp finished but no video file was produced.")
    return downloaded[0]


def download_direct(url: str, dest_dir: Path, progress=None) -> Path:
    filename = Path(urllib.parse.urlparse(url).path).name or "source.mp4"
    dest = dest_dir / filename

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and progress:
                        progress("download", downloaded * 100 // total)
    return dest


def get_video(source: str, work_dir: Path, progress=None) -> Path:
    """Return a local path to the video, downloading it first if needed."""
    if not is_url(source):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {source}")
        return path

    if looks_like_video_url(source):
        return download_direct(source, work_dir, progress)
    return download_with_ytdlp(source, work_dir)


# ---------------------------------------------------------------------------
# Step 2: cut the video into frames
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path, fps: float, progress=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps else 0

    if duration > MAX_VIDEO_SECONDS:
        cap.release()
        raise ValueError(
            f"Video is {duration / 60:.1f} min long; the limit is {MAX_VIDEO_SECONDS // 60} min."
        )

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    frame_idx = 0
    stride = max(1, round(video_fps / fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            frame_path = frames_dir / f"frame_{len(frame_paths):06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            if progress and total_frames:
                progress("extract", frame_idx * 100 // total_frames)
        frame_idx += 1

    cap.release()
    if not frame_paths:
        raise RuntimeError("No frames were extracted from the video.")
    return frame_paths, video_fps, (width, height)


# ---------------------------------------------------------------------------
# Step 3: caption the frames with BLIP
# ---------------------------------------------------------------------------

def load_captioner(model_name: str, device: str):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def caption_frames(frame_paths, processor, model, device: str,
                   max_new_tokens: int = 50, progress=None) -> list:
    captions = []
    for i, frame_path in enumerate(frame_paths, 1):
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append(caption)
        if progress:
            progress("caption", i * 100 // len(frame_paths), caption)
    return captions


# ---------------------------------------------------------------------------
# Step 4: merge frames + captions back into a video
# ---------------------------------------------------------------------------

def find_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ["arial.ttf", "segoeui.ttf", "calibri.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def burn_caption(frame: np.ndarray, caption: str) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_size = max(16, w // 40)
    font = find_font(font_size)

    words = caption.split()
    lines, line = [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font)[2] <= w - 20:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    line_height = font_size + 6
    box_height = line_height * len(lines) + 16
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([(0, h - box_height), (w, h)], fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    y = h - box_height + 8
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        x = (w - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), ln, font=font, fill=(255, 255, 255))
        y += line_height

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def merge_frames(frame_paths, captions, output_path: Path, fps: float,
                 size, progress=None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    for i, (frame_path, caption) in enumerate(zip(frame_paths, captions), 1):
        frame = cv2.imread(str(frame_path))
        writer.write(burn_caption(frame, caption))
        if progress:
            progress("merge", i * 100 // len(frame_paths))
    writer.release()


# ---------------------------------------------------------------------------
# Full pipeline (used by the job worker)
# ---------------------------------------------------------------------------

def run_pipeline(source: str, output_path: Path, fps: float, model_name: str,
                 device: str, processor, model, progress=None) -> dict:
    """Run all 4 steps. progress(stage, percent, caption=None) is called on updates."""
    work_dir = Path(tempfile.mkdtemp(prefix="video_caption_"))
    try:
        def cb(stage, percent, caption=None):
            if progress:
                progress(stage, percent, caption)

        video_path = get_video(source, work_dir, cb)
        frame_paths, video_fps, size = extract_frames(video_path, work_dir / "frames", fps, cb)
        captions = caption_frames(frame_paths, processor, model, device, progress=cb)
        merge_frames(frame_paths, captions, output_path, fps, size, cb)

        return {
            "captions": captions,
            "timestamps": [i / fps for i in range(len(captions))],
            "output_path": str(output_path),
            "frames": len(frame_paths),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
