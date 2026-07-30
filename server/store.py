from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from worker.separate_v4 import utc_now, write_metadata


class JobNotFound(KeyError):
    pass


class JobStore:
    def __init__(self, root: Path, retention_hours: int) -> None:
        self.root = root
        self.jobs_root = root / "jobs"
        self.retention_hours = retention_hours
        self.lock = threading.RLock()
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def job_path(self, job_id: str) -> Path:
        try:
            normalized = str(uuid.UUID(job_id))
        except ValueError as exc:
            raise JobNotFound(job_id) from exc
        path = (self.jobs_root / normalized).resolve()
        if path.parent != self.jobs_root.resolve():
            raise JobNotFound(job_id)
        return path

    def metadata_path(self, job_id: str) -> Path:
        return self.job_path(job_id) / "metadata.json"

    def read(self, job_id: str) -> dict:
        path = self.metadata_path(job_id)
        if not path.is_file():
            raise JobNotFound(job_id)
        with self.lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def create(self, original_filename: str, suffix: str) -> tuple[Path, dict]:
        job_id = str(uuid.uuid4())
        job = self.job_path(job_id)
        (job / "original").mkdir(parents=True)
        # Retention <= 0 keeps jobs indefinitely: cleanup skips a null expiry.
        expires = (
            datetime.now(timezone.utc) + timedelta(hours=self.retention_hours)
            if self.retention_hours > 0
            else None
        )
        metadata = {
            "jobId": job_id,
            "originalFilename": original_filename,
            "status": "uploading",
            "progress": 0,
            "stage": "Uploading audio",
            "model": "BS-Roformer ep_317 → BS-Roformer-SW",
            "stemType": "6stems",
            "createdAt": utc_now(),
            "expiresAt": expires.isoformat() if expires else None,
            "sourceSuffix": suffix,
        }
        write_metadata(job, metadata)
        return job, metadata

    def update(self, job_id: str, **values: object) -> dict:
        with self.lock:
            metadata = self.read(job_id)
            metadata.update(values)
            write_metadata(self.job_path(job_id), metadata)
            return metadata

    def list(self, limit: int = 20) -> list[dict]:
        items: list[dict] = []
        for path in self.jobs_root.glob("*/metadata.json"):
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        items.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
        return items[:limit]

    def delete(self, job_id: str) -> None:
        job = self.job_path(job_id)
        if not job.is_dir():
            raise JobNotFound(job_id)
        shutil.rmtree(job)

    def recoverable(self) -> Iterable[dict]:
        return (
            item
            for item in self.list(limit=10_000)
            if item.get("status") in {"queued", "processing"}
        )

    def cleanup_expired(self) -> int:
        # Disabling retention also protects jobs created while it was enabled,
        # which still carry an expiresAt from that time.
        if self.retention_hours <= 0:
            return 0
        now = datetime.now(timezone.utc)
        removed = 0
        for item in self.list(limit=10_000):
            value = item.get("expiresAt")
            if not value or item.get("status") == "processing":
                continue
            try:
                expires = datetime.fromisoformat(value)
            except ValueError:
                continue
            if expires <= now:
                try:
                    self.delete(item["jobId"])
                    removed += 1
                except JobNotFound:
                    pass
        return removed
