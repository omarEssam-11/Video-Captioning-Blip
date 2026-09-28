"""
Video captioning app: FastAPI REST API + Gradio web UI.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 7860
"""

import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.pipeline import DEFAULT_MODEL, load_captioner, run_pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = DEFAULT_MODEL
RESULT_TTL_SECONDS = 3600  # results are deleted 1 hour after creation

# ---------------------------------------------------------------------------
# Job manager (in-memory, sequential worker)
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, source: str, fps: float):
        self.id = uuid.uuid4().hex[:12]
        self.source = source
        self.fps = fps
        self.state = "queued"        # queued | running | done | error
        self.stage = ""              # download | extract | caption | merge
        self.percent = 0
        self.current_caption = ""
        self.error = ""
        self.result_path = None
        self.captions = None
        self.timestamps = None
        self.frames = 0
        self.created_at = time.time()
        self.done_at = None

    def to_dict(self):
        return {
            "job_id": self.id,
            "state": self.state,
            "stage": self.stage,
            "percent": self.percent,
            "current_caption": self.current_caption,
            "error": self.error,
            "frames": self.frames,
            "created_at": self.created_at,
        }


jobs: dict[str, Job] = {}
job_queue: deque[Job] = deque()
jobs_lock = threading.Lock()
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Model (loaded once at startup in the background)
# ---------------------------------------------------------------------------

model_state = {"status": "loading", "processor": None, "model": None}


def load_model():
    try:
        processor, model = load_captioner(MODEL_NAME, DEVICE)
        model_state.update(status="ready", processor=processor, model=model)
        print(f"[app] BLIP model ready on {DEVICE}")
    except Exception as e:
        model_state["status"] = "error"
        print(f"[app] Model loading failed: {e}")


threading.Thread(target=load_model, daemon=True).start()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def process_job(job: Job):
    job.state = "running"

    def progress(stage, percent, caption=None):
        job.stage = stage
        job.percent = percent
        if caption:
            job.current_caption = caption

    output_path = results_dir / f"{job.id}.mp4"
    try:
        result = run_pipeline(
            source=job.source,
            output_path=output_path,
            fps=job.fps,
            model_name=MODEL_NAME,
            device=DEVICE,
            processor=model_state["processor"],
            model=model_state["model"],
            progress=progress,
        )
        job.result_path = str(result["output_path"])
        job.captions = result["captions"]
        job.timestamps = result["timestamps"]
        job.frames = result["frames"]
        job.state = "done"
        job.percent = 100
    except Exception as e:
        job.state = "error"
        job.error = str(e)
        print(f"[app] job {job.id} failed: {e}")
    finally:
        job.done_at = time.time()


def worker_loop():
    while True:
        job = None
        with jobs_lock:
            # expire old results
            now = time.time()
            for j in list(jobs.values()):
                if j.state in ("done", "error") and j.done_at and now - j.done_at > RESULT_TTL_SECONDS:
                    if j.result_path and Path(j.result_path).exists():
                        Path(j.result_path).unlink()
                    del jobs[j.id]
            if job_queue:
                job = job_queue.popleft()
        if job:
            process_job(job)
        else:
            time.sleep(0.2)


threading.Thread(target=worker_loop, daemon=True).start()


def submit_job(source: str, fps: float) -> Job:
    job = Job(source=source, fps=fps)
    with jobs_lock:
        jobs[job.id] = job
        job_queue.append(job)
    return job


def get_job(job_id: str) -> Job:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found (it may have expired).")
    return job


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Video Captioning API", version="1.0.0")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_status": model_state["status"],
        "device": DEVICE,
        "queued_jobs": len(job_queue),
    }


@app.post("/api/caption")
def create_caption_job(source: str, fps: float = 1.0):
    """Submit a captioning job. `source` is a video file path or URL."""
    if not source:
        raise HTTPException(status_code=400, detail="source is required")
    job = submit_job(source, fps)
    return {"job_id": job.id, "state": job.state}


@app.get("/api/caption/{job_id}")
def caption_status(job_id: str):
    return get_job(job_id).to_dict()


@app.get("/api/caption/{job_id}/result")
def caption_result(job_id: str):
    job = get_job(job_id)
    if job.state != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job.state}, not done yet.")
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="Result file has expired.")
    return FileResponse(path, media_type="video/mp4", filename=f"captioned_{job.id}.mp4")


@app.get("/api/caption/{job_id}/captions")
def caption_list(job_id: str):
    job = get_job(job_id)
    if job.state != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job.state}, not done yet.")
    return {
        "job_id": job.id,
        "frames": job.frames,
        "captions": [
            {"time": t, "caption": c}
            for t, c in zip(job.timestamps, job.captions)
        ],
    }


# ---------------------------------------------------------------------------
# Gradio UI (mounted at /)
# ---------------------------------------------------------------------------

def build_ui():
    def submit(source_url, source_file, fps):
        source = source_file.name if source_file else source_url
        if not source:
            return "Please provide a video file or URL.", None, None, None, None
        job = submit_job(source, fps)
        return job.id, None, None, None, None

    def watch(job_id):
        while True:
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                yield "Job expired.", None, None, None, 0
                return
            if job.state == "error":
                yield f"Error: {job.error}", None, None, None, 0
                return
            if job.state == "done":
                captions = "\n".join(
                    f"{t:8.2f}s  {c}" for t, c in zip(job.timestamps, job.captions)
                )
                yield "Done!", job.result_path, captions, None, 100
                return
            stage = job.stage or "queued"
            yield f"{stage}... {job.percent}%", None, None, job.current_caption or None, job.percent
            time.sleep(1)

    with gr.Blocks(title="Video Captioning with BLIP", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Video Captioning with BLIP\n"
            "Upload a video or paste a link (YouTube, direct video URL). "
            "Each frame is captioned with the BLIP model and the captions are burned onto the output video."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                url_input = gr.Textbox(
                    label="Video URL",
                    placeholder="https://www.youtube.com/watch?v=... or direct .mp4 link",
                )
                file_input = gr.File(label="Or upload a video file", file_types=["video"])
                fps_input = gr.Slider(
                    minimum=0.5, maximum=5, value=1, step=0.5,
                    label="Frames per second to caption",
                    info="Higher = smoother captions but slower processing",
                )
                submit_btn = gr.Button("Caption video", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Progress")
                status_box = gr.Textbox(
                    label="Status", interactive=False,
                    value="Waiting for a video...",
                )
                progress_bar = gr.Slider(
                    minimum=0, maximum=100, value=0, interactive=False,
                    label="Progress",
                )
                caption_box = gr.Textbox(
                    label="Current caption", interactive=False,
                    placeholder="Captions will appear here as frames are processed...",
                )

        gr.Markdown("### Result")
        with gr.Row():
            with gr.Column(scale=2):
                video_out = gr.Video(label="Captioned video", interactive=False)
            with gr.Column(scale=1):
                log_out = gr.Textbox(
                    label="Caption log", interactive=False, lines=10,
                    placeholder="Timestamped captions will appear here...",
                )
                with gr.Row():
                    download_video_btn = gr.Button("Download video", variant="secondary")
                    download_log_btn = gr.Button("Download log", variant="secondary")

        gr.Markdown("### How it works")
        gr.Markdown(
            "1. **Get** — the video is downloaded (if a link) or read from disk\n"
            "2. **Cut** — frames are sampled at the chosen rate\n"
            "3. **Caption** — each frame is described by the BLIP vision-language model\n"
            "4. **Merge** — captions are burned onto the frames and re-encoded to MP4\n\n"
            "The REST API is available at `/api/caption` (POST to submit, GET to poll)."
        )

        job_id_state = gr.State()

        submit_btn.click(
            submit, inputs=[url_input, file_input, fps_input],
            outputs=[job_id_state, status_box, video_out, log_out, progress_bar],
        ).then(
            watch, inputs=[job_id_state],
            outputs=[status_box, video_out, log_out, caption_box, progress_bar],
        )

    return demo


import gradio as gr

app = gr.mount_gradio_app(app, build_ui(), path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
