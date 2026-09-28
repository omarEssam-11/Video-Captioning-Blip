# Video Captioning with BLIP

Captions the frames of a video using the [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) image-captioning model, then burns the captions onto the frames and re-encodes a captioned video.

## Workflow

1. **Get the video** — pass a local file path, a direct video-file URL, or a page URL (e.g. YouTube, downloaded via `yt-dlp`).
2. **Cut into frames** — samples frames with OpenCV at the rate given by `--fps`.
3. **Caption frames** — each frame is captioned with BLIP on the CPU or GPU.
4. **Merge back** — captions are burned onto the bottom of each frame and the frames are written to an MP4. A `.txt` sidecar with timestamps is also produced.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Local video
python caption_video.py my_video.mp4 -o captioned.mp4

# Direct link to a video file
python caption_video.py https://example.com/clip.mp4 -o captioned.mp4

# Page link (YouTube, etc. — needs yt-dlp)
python caption_video.py https://www.youtube.com/watch?v=xxxx -o captioned.mp4

# Sample 2 frames per second, keep extracted frames
python caption_video.py my_video.mp4 --fps 2 --keep-frames
```

### Options

| Flag | Default | Description |
|---|---|---|
| `source` | — | Video file path or URL |
| `-o`, `--output` | `captioned_video.mp4` | Output video path |
| `--fps` | `1` | Frames per second to sample |
| `--model` | `Salesforce/blip-image-captioning-base` | Hugging Face BLIP model |
| `--device` | `cuda` if available, else `cpu` | Inference device |
| `--keep-frames` | off | Keep the extracted frames directory |

## Output

- `captioned.mp4` — video with captions burned in
- `captioned.txt` — timestamped caption log, e.g. `    0.00s  a dog running through the grass`
