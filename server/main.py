"""FastAPI surface. Keep endpoints minimal — one for upload, one to fire a
job, one to poll status, one to download an output. Add new job kinds by
extending runners.RUNNERS."""
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import jobs, runners, settings
from .auth import require_bearer

app = FastAPI(title="stabl-server")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/v1/upload", dependencies=[Depends(require_bearer)])
async def upload(
    file: UploadFile = File(...),
    kind: str = Form("clip"),  # "clip" (mp4) or "csv"
) -> dict:
    """Multipart upload. OK for csv tracks and small clips only — Cloudflare's
    free tier caps the body at 100MB. For bigger files use /v1/upload/reserve
    + rsync over Tailscale (see CLI `stabl upload-rsync`)."""
    suffix = ".mp4" if kind == "clip" else ".csv"
    file_id = uuid.uuid4().hex
    dest = settings.UPLOADS_DIR / f"{file_id}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file_id": file_id, "size": dest.stat().st_size, "kind": kind}


@app.post("/v1/upload/reserve", dependencies=[Depends(require_bearer)])
async def reserve_upload(body: dict) -> dict:
    """Allocate a file_id for an out-of-band upload (rsync). Returns the
    target path the client should rsync to. Used for clips > Cloudflare's
    proxy body limit."""
    kind = body.get("kind", "clip")
    suffix = ".mp4" if kind == "clip" else ".csv"
    file_id = uuid.uuid4().hex
    dest = settings.UPLOADS_DIR / f"{file_id}{suffix}"
    return {"file_id": file_id, "rsync_target": str(dest), "kind": kind}


@app.get("/v1/files/{file_id}/info", dependencies=[Depends(require_bearer)])
async def file_info(file_id: str) -> dict:
    """Verify a file landed and report its size. Used after an rsync upload."""
    for d in (settings.UPLOADS_DIR, settings.OUTPUTS_DIR):
        for p in d.glob(f"{file_id}.*"):
            return {"file_id": file_id, "path": str(p), "size": p.stat().st_size}
    raise HTTPException(404, "no such file")


@app.post("/v1/upload/r2/presign", dependencies=[Depends(require_bearer)])
async def r2_presign(body: dict) -> dict:
    """Return a presigned PUT URL for direct upload to R2. Client uses it
    with curl -T (or anything that does HTTP PUT). Then calls /finalize."""
    from . import r2
    if not r2.configured():
        raise HTTPException(503, "R2 not configured on server")
    kind = body.get("kind", "clip")
    suffix = ".mp4" if kind == "clip" else ".csv"
    file_id = uuid.uuid4().hex
    key = f"uploads/{file_id}{suffix}"
    url = r2.presign_put(key, expires=body.get("expires_in", 3600))
    return {"file_id": file_id, "kind": kind, "r2_key": key, "presigned_url": url}


@app.get("/v1/version")
def version() -> dict:
    """Public — return the running commit. No auth so the CLI can check
    parity before sending a write. Includes dirty flag if there are uncommitted
    changes (which should never happen in production)."""
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=settings.REPO_ROOT, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=settings.REPO_ROOT, text=True,
        ).strip())
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=settings.REPO_ROOT, text=True,
        ).strip()
        return {"sha": sha, "branch": branch, "dirty": dirty}
    except Exception as e:
        return {"sha": None, "error": str(e)}


@app.post("/v1/sync", dependencies=[Depends(require_bearer)])
def sync(request: Request, body: dict | None = None) -> dict:
    """Git pull origin/main + reinstall + restart. Hardcoded to main —
    add a release/tag path later if branch deploys are needed.

    Loopback-only: trusted X-Forwarded-For chain (uvicorn is started
    with --forwarded-allow-ips=127.0.0.1) means request.client.host
    is the original client IP when behind cloudflared, or 127.0.0.1
    when called directly on localhost. Restricting to loopback
    effectively makes the deploy require tailnet access — you SSH to
    olares (which needs Tailscale) and call localhost:8000 from there.
    Stops a leaked bearer token from being deploy-capable on its own.
    """
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(403,
            "sync only accepts loopback connections; "
            "ssh into the host and call localhost:8000")
    import subprocess, os, signal, threading
    branch = "main"
    repo = settings.REPO_ROOT
    out = []
    def run(*cmd):
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        out.append(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
        if r.returncode != 0:
            raise HTTPException(500, f"{' '.join(cmd)} failed:\n{r.stdout}{r.stderr}")
    run("git", "fetch", "origin", branch)
    run("git", "reset", "--hard", f"origin/{branch}")
    # Reinstall in case pyproject.toml changed (cheap no-op otherwise).
    pip = str(Path(os.sys.executable).parent / "pip")
    run(pip, "install", "-e", f"{repo}[api]")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    def kick():
        import time; time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=kick, daemon=True).start()
    return {"sha": sha, "branch": branch, "log": out, "restarting": True}


@app.post("/v1/restart", dependencies=[Depends(require_bearer)])
async def restart() -> dict:
    """Schedule a server restart shortly after responding so the systemd
    unit picks up newly deployed code."""
    import os, signal, threading
    def kick():
        import time; time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)  # systemd Restart=on-failure will bring us back
    threading.Thread(target=kick, daemon=True).start()
    return {"restarting": True}


# ----- TEMP DEBUG ENDPOINTS REMOVED HERE -----
# /v1/diag, /v1/proc/{pid}, /v1/pods, /v1/kick-pod were added 2026-06-09
# during an Olares outage when SSH and FRP were both unreachable, and
# removed 2026-06-12 once host access was restored. They were a wider
# blast radius than the rest of the stabl API (kubectl + /proc + fixed
# shell commands as the olares user), so they don't live past the
# incident. If similar tooling is needed again, see git history at
# commits 07c78fd / e8fa542 / a2754cc and re-introduce intentionally.
# --------------------------------------------


@app.post("/v1/upload/r2/finalize", dependencies=[Depends(require_bearer)])
async def r2_finalize(body: dict) -> dict:
    """After client PUTs to R2, call this with {file_id, r2_key}. Server
    pulls the object from R2 into UPLOADS_DIR so runners can use it locally."""
    from . import r2
    if not r2.configured():
        raise HTTPException(503, "R2 not configured on server")
    file_id = body["file_id"]
    key = body["r2_key"]
    info = r2.head(key)
    if not info:
        raise HTTPException(404, f"no object at r2://{r2.bucket()}/{key}")
    suffix = Path(key).suffix or ".bin"
    dest = settings.UPLOADS_DIR / f"{file_id}{suffix}"
    r2.download(key, str(dest))
    return {"file_id": file_id, "path": str(dest), "size": dest.stat().st_size}


@app.post("/v1/jobs/{kind}", dependencies=[Depends(require_bearer)])
async def create_job(kind: str, params: dict) -> dict:
    if kind not in runners.RUNNERS:
        raise HTTPException(404, f"unknown job kind: {kind}")
    job = jobs.create(kind, params, runners.RUNNERS[kind])
    return {"job_id": job.id, "status": job.status}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_bearer)])
async def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    from dataclasses import asdict
    return asdict(job)


@app.get("/v1/jobs", dependencies=[Depends(require_bearer)])
async def list_jobs(limit: int = 50) -> dict:
    from dataclasses import asdict
    return {"jobs": [asdict(j) for j in jobs.all_jobs(limit=limit)]}


@app.get("/v1/files/{file_id}", dependencies=[Depends(require_bearer)])
async def get_file(file_id: str) -> FileResponse:
    for d in (settings.OUTPUTS_DIR, settings.UPLOADS_DIR):
        for p in d.glob(f"{file_id}.*"):
            return FileResponse(p, filename=p.name)
    raise HTTPException(404, "no such file")
