"""In-memory job registry persisted to a JSON file. Background work runs
in threads — fine for the IO/compute mix we have (ffmpeg + cv2 release the
GIL). Move to a queue (Celery/RQ) if we ever need cross-process workers.
"""
import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from . import settings


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"          # queued | running | done | failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    output_file_id: Optional[str] = None
    error: Optional[str] = None
    params: dict = field(default_factory=dict)
    log_tail: list = field(default_factory=list)


_lock = threading.Lock()
_jobs: dict[str, Job] = {}


def _persist() -> None:
    tmp = settings.JOBS_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump({jid: asdict(j) for jid, j in _jobs.items()}, f)
    tmp.replace(settings.JOBS_FILE)


def _restore() -> None:
    if not settings.JOBS_FILE.exists():
        return
    raw = json.loads(settings.JOBS_FILE.read_text())
    for jid, d in raw.items():
        _jobs[jid] = Job(**d)


_restore()


def create(kind: str, params: dict, runner: Callable[[Job], None]) -> Job:
    job = Job(id=uuid.uuid4().hex, kind=kind, params=params)
    with _lock:
        _jobs[job.id] = job
        _persist()
    t = threading.Thread(target=_run, args=(job, runner), daemon=True)
    t.start()
    return job


def _run(job: Job, runner: Callable[[Job], None]) -> None:
    with _lock:
        job.status = "running"
        job.started_at = time.time()
        _persist()
    try:
        runner(job)
        with _lock:
            job.status = "done"
            job.finished_at = time.time()
            _persist()
    except Exception as e:
        with _lock:
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            job.finished_at = time.time()
            _persist()


def get(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def all_jobs(limit: int = 100) -> list[Job]:
    return sorted(_jobs.values(), key=lambda j: -j.created_at)[:limit]


def append_log(job: Job, line: str) -> None:
    with _lock:
        job.log_tail.append(line)
        if len(job.log_tail) > 200:
            job.log_tail = job.log_tail[-200:]
        _persist()
