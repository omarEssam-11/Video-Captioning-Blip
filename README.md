# Video Captioning with BLIP

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-BLIP-red.svg)](https://huggingface.co/Salesforce/blip-image-captioning-base)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automatically caption every frame of a video using the [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) vision-language model, then burn the captions onto the frames and re-encode a captioned video.

![Demo](https://img.shields.io/badge/Demo-Web%20UI-orange.svg)

## Features

- **Multiple input sources** — local video files, direct video URLs, or page links (YouTube, etc. via `yt-dlp`)
- **Web UI** — upload a video or paste a link, watch live progress with per-frame captions, download the result
- **REST API** — submit jobs, poll progress, download results programmatically
- **CLI** — simple command-line interface for scripting
- **Jupyter notebook** — interactive step-by-step notebook
- **GPU/CPU support** — uses CUDA if available, falls back to CPU
- **Docker** — containerized for easy deployment

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. Get     │────▶│  2. Cut     │────▶│  3. Caption │────▶│  4. Merge   │
│  Video      │     │  Frames     │     │  with BLIP  │     │  & Burn     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **Get** — the video is downloaded (if a link) or read from disk
2. **Cut** — frames are sampled at the chosen rate (e.g. 1 fps)
3. **Caption** — each frame is described by the BLIP vision-language model
4. **Merge** — captions are burned onto the frames and re-encoded to MP4

## Quick Start

### Prerequisites

- Python 3.10+
- ~2 GB disk space (for the BLIP model, downloaded on first run)

### Installation

```bash
git clone https://github.com/omarEssam-11/Video-Captioning-Blip.git
cd Video-Captioning-Blip
pip install -r requirements.txt
```

### Run the Web UI

```bash
# Option A: double-click run.bat (Windows)
# Option B:
py -m uvicorn app.main:app --host 127.0.0.1 --port 7860
```

Open **http://127.0.0.1:7860** in your browser.

### Run the CLI

```bash
# Local video
python caption_video.py my_video.mp4 -o captioned.mp4

# Direct link to a video file
python caption_video.py https://example.com/clip.mp4 -o captioned.mp4

# Page link (YouTube, etc.)
python caption_video.py https://www.youtube.com/watch?v=xxxx -o captioned.mp4

# Sample 2 frames per second
python caption_video.py my_video.mp4 --fps 2
```

### Run the Notebook

```bash
jupyter notebook caption_video_notebook.ipynb
```

## Web UI

![UI Screenshot](https://img.shields.io/badge/UI-Gradio-purple.svg)

The Gradio interface provides:
- Video file upload or URL input
- Adjustable sampling rate (0.5–5 fps)
- Live progress bar with current caption display
- Captioned video player
- Timestamped caption log
- Download buttons for video and log

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Model status, device, queue length |
| `POST` | `/api/caption?source=<path_or_url>&fps=1` | Submit a job → `{job_id}` |
| `GET` | `/api/caption/{job_id}` | Poll status (state, stage, percent, current caption) |
| `GET` | `/api/caption/{job_id}/result` | Download the captioned MP4 |
| `GET` | `/api/caption/{job_id}/captions` | Get timestamped captions as JSON |

### API Example

```bash
# Submit a job
curl -X POST "http://127.0.0.1:7860/api/caption?source=my_video.mp4&fps=1"
# → {"job_id": "abc123", "state": "queued"}

# Poll status
curl "http://127.0.0.1:7860/api/caption/abc123"
# → {"job_id": "abc123", "state": "running", "stage": "caption", "percent": 42, ...}

# Download result
curl -OJ "http://127.0.0.1:7860/api/caption/abc123/result"

# Get captions as JSON
curl "http://127.0.0.1:7860/api/caption/abc123/captions"
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `source` | — | Video file path or URL |
| `-o`, `--output` | `captioned_video.mp4` | Output video path |
| `--fps` | `1` | Frames per second to sample |
| `--model` | `Salesforce/blip-image-captioning-base` | Hugging Face BLIP model |
| `--device` | `cuda` if available, else `cpu` | Inference device |

## Project Structure

```
Video-Captioning-Blip/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI + Gradio app with job queue
│   └── pipeline.py        # Shared pipeline (download → frames → caption → merge)
├── caption_video.py       # CLI script
├── caption_video_notebook.ipynb  # Jupyter notebook version
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── run.bat                # Windows launcher for the web UI
└── README.md
```

## Docker

```bash
# Build
docker build -t video-captioning .

# Run
docker run -p 7860:7860 video-captioning
```

## API Reference

### Job States

| State | Description |
|-------|-------------|
| `queued` | Waiting in queue |
| `running` | Processing (stage: download / extract / caption / merge) |
| `done` | Finished, result available |
| `error` | Failed (see `error` field) |

### Response Format

```json
{
  "job_id": "abc123",
  "state": "running",
  "stage": "caption",
  "percent": 42,
  "current_caption": "a dog running through the grass",
  "error": "",
  "frames": 10,
  "created_at": 1695600000.0
}
```

## Notes

- The BLIP model (~1 GB) downloads from Hugging Face on first run, then is cached locally
- Videos longer than 30 minutes are rejected to protect disk space
- Results are kept for 1 hour, then auto-deleted
- Free-tier cloud hosting (Render, etc.) requires a credit card; this app is designed to run locally

## License

MIT
