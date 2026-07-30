from __future__ import annotations

import asyncio
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from server.config import AUDIO_SUFFIXES, SAFE_FILENAME, Settings
from server.manager import JobManager
from server.mix import render_mix
from server.store import JobNotFound, JobStore
from worker.separate_v4 import (
    STEM_NAMES,
    audio_separator_executable,
    env_flag,
    ffmpeg_executable,
    missing_model_files,
    model_directory,
)

logging.basicConfig(level=logging.INFO)
settings = Settings.load()
store = JobStore(settings.storage_root, settings.retention_hours)
manager = JobManager(store, settings.worker_count)
ALLOWED_SUFFIXES = AUDIO_SUFFIXES


async def cleanup_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        await asyncio.to_thread(store.cleanup_expired)


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.recover()
    cleanup_task = asyncio.create_task(cleanup_loop())
    yield
    cleanup_task.cancel()
    manager.shutdown()


app = FastAPI(
    title="StemFlow API",
    version="1.0.0",
    description="Local GPU stem separation service",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def allow_configured_private_network_access(request: Request, call_next):
    """Allow the private hosted UI to reach the user's loopback GPU service."""
    response = await call_next(request)
    origin = request.headers.get("origin")
    if (
        origin in settings.allowed_origins
        and request.headers.get("access-control-request-private-network") == "true"
    ):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


class MixRequest(BaseModel):
    levels: dict[str, float] = Field(default_factory=dict)


def public_job(metadata: dict, request: Request) -> dict:
    job_id = metadata["jobId"]
    base = str(request.base_url).rstrip("/")
    result = {
        key: value
        for key, value in metadata.items()
        if key not in {"sourceSuffix", "files", "previews", "archive", "errorDetail"}
    }
    if metadata.get("status") == "completed":
        result["stems"] = [
            {
                "id": name,
                "label": name.title(),
                "previewUrl": f"{base}/api/jobs/{job_id}/preview/{name}",
                "downloadUrl": f"{base}/api/jobs/{job_id}/stem/{name}",
            }
            for name in STEM_NAMES
        ]
        result["archiveUrl"] = f"{base}/api/jobs/{job_id}/archive"
        result["mixUrl"] = f"{base}/api/jobs/{job_id}/mix"
    return result


def get_job(job_id: str) -> dict:
    try:
        return store.read(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.exception_handler(JobNotFound)
async def job_not_found(_: Request, __: JobNotFound):
    return JSONResponse({"detail": "Job not found"}, status_code=404)


@app.get("/api/health")
def health() -> dict:
    separator_ready = False
    encoder_ready = False
    try:
        separator_ready = Path(audio_separator_executable()).is_file()
        encoder_ready = Path(ffmpeg_executable()).is_file()
    except FileNotFoundError:
        pass
    missing = missing_model_files()
    ready = separator_ready and encoder_ready and not missing
    return {
        "status": "ok" if ready else "degraded",
        "service": "StemFlow API",
        "gpuWorker": ready,
        "separatorReady": separator_ready,
        "encoderReady": encoder_ready,
        "modelsReady": not missing,
        "modelDir": str(model_directory()),
        "missingModelFiles": list(missing),
        "offline": env_flag("STEMFLOW_OFFLINE", True),
        "retentionHours": settings.retention_hours,
    }


@app.get("/api/jobs")
def list_jobs(request: Request, limit: int = 20) -> dict:
    return {
        "jobs": [
            public_job(item, request)
            for item in store.list(limit=max(1, min(limit, 100)))
        ]
    }


@app.post("/api/jobs", status_code=202)
async def create_job(request: Request, file: UploadFile = File(...)) -> dict:
    original_name = Path(file.filename or "audio").name[:180]
    suffix = Path(original_name).suffix.casefold()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported audio format")
    job, metadata = store.create(
        SAFE_FILENAME.sub("_", original_name) or f"audio{suffix}", suffix
    )
    destination = job / "original" / f"source{suffix}"
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="File exceeds upload limit")
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        store.update(
            metadata["jobId"],
            sizeBytes=size,
            status="queued",
            stage="Waiting for GPU",
            progress=2,
        )
        manager.enqueue(metadata["jobId"])
        return public_job(store.read(metadata["jobId"]), request)
    except Exception:
        if job.exists():
            store.delete(metadata["jobId"])
        raise
    finally:
        await file.close()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> dict:
    return public_job(get_job(job_id), request)


def completed_job(job_id: str) -> tuple[dict, Path]:
    metadata = get_job(job_id)
    if metadata.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not complete")
    return metadata, store.job_path(job_id)


@app.get("/api/jobs/{job_id}/preview/{stem}")
def preview(job_id: str, stem: str):
    _, job = completed_job(job_id)
    if stem not in STEM_NAMES:
        raise HTTPException(status_code=404, detail="Stem not found")
    path = job / "preview" / f"{stem}.mp3"
    return FileResponse(path, media_type="audio/mpeg", filename=f"{stem}.mp3")


@app.get("/api/jobs/{job_id}/stem/{stem}")
def stem(job_id: str, stem: str):
    _, job = completed_job(job_id)
    if stem not in STEM_NAMES:
        raise HTTPException(status_code=404, detail="Stem not found")
    path = job / "stems" / f"{stem}.wav"
    return FileResponse(path, media_type="audio/wav", filename=f"{stem}.wav")


@app.get("/api/jobs/{job_id}/archive")
def archive(job_id: str):
    metadata, job = completed_job(job_id)
    filename = f"{Path(metadata['originalFilename']).stem}-stems.zip"
    return FileResponse(
        job / "exports" / "stems.zip",
        media_type="application/zip",
        filename=filename,
    )


@app.post("/api/jobs/{job_id}/mix")
def mix(job_id: str, payload: MixRequest):
    metadata, job = completed_job(job_id)
    try:
        output = render_mix(job / "stems", payload.levels)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    filename = f"{Path(metadata['originalFilename']).stem}-mix.wav"
    return StreamingResponse(
        output,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    metadata = get_job(job_id)
    if metadata.get("status") == "processing":
        manager.cancel(job_id, delete_after=True)
        raise HTTPException(
            status_code=202, detail="Cancellation and deletion requested"
        )
    manager.cancel(job_id)
    store.delete(job_id)
    return None
