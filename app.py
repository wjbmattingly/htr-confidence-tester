"""HTR Confidence Tester — run a Qwen 3.5 HTR finetune N times over a page image
and use the divergence between runs as a word/character-level confidence signal.

    uv run app.py            # http://127.0.0.1:8765
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from analysis import analyze
from model_runner import ModelManager

ROOT = Path(__file__).parent
IMAGES_DIR = ROOT / "images"
UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MAX_SIDE = 1600  # pre-resize uploads; the processor caps visual tokens anyway

PRELOADED = [
    {
        "id": "gottschalk_antiphonary.jpg",
        "title": "Gottschalk Antiphonary",
        "detail": "Carolingian minuscule · Latin · 12th c.",
    },
    {
        "id": "kells_text.jpg",
        "title": "Book of Kells, fol. 309r",
        "detail": "Insular majuscule · Latin · c. 800",
    },
    {
        "id": "malmesbury_bible.jpg",
        "title": "Malmesbury Bible",
        "detail": "Gothic textura · Latin · 1407",
    },
    {
        "id": "peterborough.jpg",
        "title": "Peterborough Chronicle",
        "detail": "English vernacular minuscule · Old English · 12th c.",
    },
    {
        "id": "beowulf.jpg",
        "title": "Beowulf, fol. 132r",
        "detail": "Insular minuscule · Old English · c. 1000",
    },
]

app = FastAPI(title="HTR Confidence Tester")
manager = ModelManager()

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
run_queue_lock = threading.Lock()  # one generation at a time on CPU


class JobRequest(BaseModel):
    image_id: str
    num_runs: int = Field(5, ge=2, le=25)
    temperature: float = Field(0.7, ge=0.05, le=2.0)
    top_p: float = Field(0.95, ge=0.1, le=1.0)
    max_new_tokens: int = Field(3072, ge=64, le=8192)
    greedy_first: bool = False


def _image_path(image_id: str) -> Path:
    for base in (IMAGES_DIR, UPLOADS_DIR):
        p = base / Path(image_id).name
        if p.exists():
            return p
    raise HTTPException(404, f"unknown image: {image_id}")


def _worker(job_id: str, req: JobRequest):
    job = jobs[job_id]
    try:
        job["state"] = "loading_model"
        manager.ensure_loaded()

        image = Image.open(_image_path(req.image_id)).convert("RGB")
        transcripts: list[str] = []
        job["state"] = "running"

        with run_queue_lock:
            for i in range(req.num_runs):
                if job["cancelled"]:
                    break
                job["current_run"] = i + 1
                job["partial"] = ""
                job["partial_tokens"] = 0
                started = time.time()

                def on_token(text, n):
                    job["partial"] = text
                    job["partial_tokens"] = n

                out = manager.transcribe_once(
                    image,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    greedy=(req.greedy_first and i == 0),
                    max_new_tokens=req.max_new_tokens,
                    on_token=on_token,
                    should_stop=lambda: job["cancelled"],
                )
                if job["cancelled"]:
                    break
                transcripts.append(out)
                job["runs"].append(
                    {
                        "text": out,
                        "seconds": round(time.time() - started, 1),
                        "greedy": req.greedy_first and i == 0,
                    }
                )

        if job["cancelled"]:
            job["state"] = "cancelled"
            return
        job["state"] = "analyzing"
        job["result"] = analyze(transcripts)
        job["state"] = "done"
    except Exception as e:
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/images")
def list_images():
    uploads = [
        {"id": p.name, "title": p.stem, "detail": "uploaded", "uploaded": True}
        for p in sorted(UPLOADS_DIR.iterdir())
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ]
    return {
        "images": PRELOADED + uploads,
        "model": manager.repo,
        "model_state": manager.state,
        "device": manager.device,
    }


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    return FileResponse(_image_path(image_id))


@app.post("/api/upload")
async def upload(file: UploadFile):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "not a readable image")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    name = f"{Path(file.filename or 'upload').stem[:40]}_{uuid.uuid4().hex[:6]}.jpg"
    img.save(UPLOADS_DIR / name, quality=92)
    return {"id": name}


@app.post("/api/jobs")
def create_job(req: JobRequest):
    _image_path(req.image_id)  # validate early
    job_id = uuid.uuid4().hex[:10]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "image_id": req.image_id,
            "num_runs": req.num_runs,
            "current_run": 0,
            "partial": "",
            "partial_tokens": 0,
            "runs": [],
            "result": None,
            "error": None,
            "cancelled": False,
            "created": time.time(),
        }
    threading.Thread(target=_worker, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return JSONResponse(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    job["cancelled"] = True
    return {"ok": True}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
