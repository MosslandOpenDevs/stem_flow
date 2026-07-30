from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from server.store import JobNotFound, JobStore
from worker.separate_v4 import SeparationCancelled, separate_job, utc_now

logger = logging.getLogger("stemflow.jobs")


class JobManager:
    def __init__(self, store: JobStore, worker_count: int = 1) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="stemflow-gpu"
        )
        self.futures: dict[str, Future] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.delete_after_cancel: set[str] = set()
        self.lock = threading.RLock()

    def enqueue(self, job_id: str) -> None:
        with self.lock:
            active = self.futures.get(job_id)
            if active and not active.done():
                return
            cancel_event = threading.Event()
            self.cancel_events[job_id] = cancel_event
            self.store.update(
                job_id, status="queued", progress=2, stage="Waiting for GPU"
            )
            self.futures[job_id] = self.executor.submit(
                self._run, job_id, cancel_event
            )

    def _run(self, job_id: str, cancel_event: threading.Event) -> None:
        try:
            metadata = self.store.update(
                job_id,
                status="processing",
                progress=5,
                stage="Starting separation",
                startedAt=utc_now(),
                error=None,
            )

            def progress(value: int, stage: str) -> None:
                self.store.update(job_id, progress=value, stage=stage)

            separate_job(
                self.store.job_path(job_id), metadata, progress, cancel_event
            )
        except SeparationCancelled:
            try:
                self.store.update(
                    job_id,
                    status="cancelled",
                    stage="Cancelled",
                    cancelledAt=utc_now(),
                )
            except JobNotFound:
                pass
        except Exception as exc:  # noqa: BLE001 - convert worker failure to job state
            logger.exception("Job %s failed", job_id)
            try:
                self.store.update(
                    job_id,
                    status="failed",
                    stage="Separation failed",
                    error="The GPU worker could not complete this track.",
                    errorDetail=str(exc)[-8000:],
                    failedAt=utc_now(),
                )
            except JobNotFound:
                pass
        finally:
            with self.lock:
                self.cancel_events.pop(job_id, None)
                delete_after = job_id in self.delete_after_cancel
                self.delete_after_cancel.discard(job_id)
            if delete_after:
                try:
                    self.store.delete(job_id)
                except JobNotFound:
                    pass

    def cancel(self, job_id: str, delete_after: bool = False) -> bool:
        with self.lock:
            event = self.cancel_events.get(job_id)
            if event:
                if delete_after:
                    self.delete_after_cancel.add(job_id)
                event.set()
                self.store.update(job_id, stage="Cancelling")
                return True
            future = self.futures.get(job_id)
            return bool(future and future.cancel())

    def recover(self) -> None:
        for item in self.store.recoverable():
            self.enqueue(item["jobId"])

    def shutdown(self) -> None:
        for event in tuple(self.cancel_events.values()):
            event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
