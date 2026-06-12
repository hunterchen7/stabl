"""FastAPI surface. Keep endpoints minimal — one for upload, one to fire a
job, one to poll status, one to download an output. Add new job kinds by
extending runners.RUNNERS."""
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
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
def sync(body: dict | None = None) -> dict:
    """Git pull origin/main + reinstall + restart. Hardcoded to main —
    add a release/tag path later if branch deploys are needed."""
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


@app.get("/v1/diag", dependencies=[Depends(require_bearer)])
def diag() -> dict:
    """Read-only host diagnostics. Used to investigate Olares-side memory
    pressure / k3s pod state when SSH-via-Tailscale and Olares Space FRP
    are both unavailable. Runs as the `olares` host user — only sees what
    that user can read (which is enough for ps / kubectl if the k3s
    kubeconfig is group-readable, which is the default on Olares)."""
    import subprocess
    def run(*cmd, timeout=10):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return {"rc": r.returncode, "out": r.stdout[-8000:], "err": r.stderr[-2000:]}
        except Exception as e:
            return {"rc": -1, "err": str(e)}
    out: dict = {}
    out["uptime"] = run("uptime")
    try:
        out["loadavg"] = {"out": open("/proc/loadavg").read()}
    except Exception as e:
        out["loadavg"] = {"err": str(e)}
    try:
        out["meminfo"] = {"out": open("/proc/meminfo").read()}
    except Exception as e:
        out["meminfo"] = {"err": str(e)}
    out["top_rss"] = run("bash", "-c",
        "ps axo pid,rss,user,comm --sort=-rss | head -30")
    out["top_swap"] = run("bash", "-c",
        "for f in /proc/*/status; do "
        "  awk '/^Name:/{n=$2} /^VmSwap:/{print $2,n}' \"$f\" 2>/dev/null; "
        "done | sort -nr | head -20")
    out["kubectl_top_pods"] = run("bash", "-c",
        "for kc in /etc/rancher/k3s/k3s.yaml $HOME/.kube/config; do "
        "  [ -r \"$kc\" ] && { "
        "    KUBECONFIG=\"$kc\" kubectl top pod -A --sort-by=memory 2>&1 | head -40; "
        "    exit 0; "
        "  }; "
        "done; echo '(no readable kubeconfig)'")
    out["kubectl_bad_pods"] = run("bash", "-c",
        "for kc in /etc/rancher/k3s/k3s.yaml $HOME/.kube/config; do "
        "  [ -r \"$kc\" ] && { "
        "    KUBECONFIG=\"$kc\" kubectl get pods -A --no-headers 2>&1 "
        "      | awk '$4!=\"Running\" && $4!=\"Completed\"' | head -40; "
        "    exit 0; "
        "  }; "
        "done; echo '(no readable kubeconfig)'")
    out["systemctl_failed"] = run("bash", "-c",
        "systemctl list-units --state=failed --no-pager 2>&1 | head -20")
    out["dmesg_tail"] = run("bash", "-c",
        "dmesg 2>/dev/null | tail -30 || echo '(no dmesg access)'")
    out["runners_ps"] = run("bash", "-c",
        "ps -eo pid,ppid,user,etime,rss,comm,args 2>/dev/null "
        "| grep -iE 'runner|actions|github' | grep -v grep | head -20")
    out["runner_homes"] = run("bash", "-c",
        "for d in /home/*/actions-runner /opt/actions-runner "
        "         /home/*/_work /opt/_work; do "
        "  [ -d \"$d\" ] && echo \"$d\"; "
        "done")
    out["docker_containers"] = run("bash", "-c",
        "{ docker ps --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}' 2>&1 "
        "  || podman ps --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}' 2>&1 "
        "  || echo '(no docker/podman socket access)'; } | head -20")
    out["docker_ps_all"] = run("bash", "-c",
        "docker ps -a --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}' 2>&1 | head -40")
    out["containerd_ctrs"] = run("bash", "-c",
        "for ns in $(sudo -n /usr/local/bin/ctr ns list -q 2>/dev/null "
        "             || /usr/local/bin/ctr ns list -q 2>/dev/null "
        "             || echo default k8s.io moby); do "
        "  echo \"--- ns:$ns ---\"; "
        "  /usr/local/bin/ctr -n \"$ns\" containers ls 2>&1 | head -10; "
        "done 2>&1 | head -40")
    out["runner_search"] = run("bash", "-c",
        "ls -la /home/olares/actions-runner /home/olares/_work "
        "       /opt/actions-runner /opt/_work "
        "       /etc/systemd/system/actions-runner*.service 2>&1 "
        "| head -20")
    out["systemd_runner_units"] = run("bash", "-c",
        "systemctl list-units --all --no-pager --type=service 2>&1 "
        "| grep -iE 'runner|actions|github' | head -10")
    # Deeper hunt — runners may be in a privileged k8s pod via docker socket,
    # or named differently in some user-space namespace.
    out["pods_by_image"] = run("bash", "-c",
        "for kc in /etc/rancher/k3s/k3s.yaml $HOME/.kube/config; do "
        "  [ -r \"$kc\" ] && { "
        "    KUBECONFIG=\"$kc\" kubectl get pods -A "
        "      -o custom-columns=NS:.metadata.namespace,POD:.metadata.name,IMAGES:'.spec.containers[*].image' "
        "      --no-headers 2>&1 "
        "      | grep -iE 'runner|actions|github|myoung34|summerwind|advocate' "
        "      | head -10; "
        "    exit 0; "
        "  }; "
        "done; echo '(no readable kubeconfig)'")
    out["all_namespaces"] = run("bash", "-c",
        "for kc in /etc/rancher/k3s/k3s.yaml $HOME/.kube/config; do "
        "  [ -r \"$kc\" ] && { "
        "    KUBECONFIG=\"$kc\" kubectl get ns --no-headers 2>&1; "
        "    exit 0; "
        "  }; "
        "done")
    out["search_disk_for_runner"] = run("bash", "-c",
        "find /home/olares /root /opt /etc -maxdepth 5 "
        "  -iname '*actions-runner*' -o -iname '*github-runner*' "
        "  -o -iname '_diag' 2>/dev/null | head -20")
    out["all_listening_ports"] = run("bash", "-c",
        "ss -tlnp 2>/dev/null | head -30 || netstat -tlnp 2>/dev/null | head -30")
    # Dump POD+IMAGE for namespaces that could plausibly host a runner.
    out["pods_in_dev_ns"] = run("bash", "-c",
        "for kc in /etc/rancher/k3s/k3s.yaml $HOME/.kube/config; do "
        "  [ -r \"$kc\" ] && { "
        "    for ns in devworkspace-dev-hunterchen studio-hunterchen studioserver-shared "
        "              windows-hunterchen user-system-hunterchen user-space-hunterchen "
        "              default os-platform; do "
        "      echo \"=== ns:$ns ===\"; "
        "      KUBECONFIG=\"$kc\" kubectl -n \"$ns\" get pods "
        "        -o custom-columns=POD:.metadata.name,IMAGES:'.spec.containers[*].image' "
        "        --no-headers 2>&1 | head -20; "
        "    done; "
        "    exit 0; "
        "  }; "
        "done")
    return out


@app.get("/v1/proc/{pid}", dependencies=[Depends(require_bearer)])
def proc_info(pid: int) -> dict:
    """Read-only inspection of a single host process via /proc: cmdline,
    cwd, exe, status. No shell execution. Used to identify what a
    specific high-RSS process actually is during incident response."""
    base = f"/proc/{pid}"
    out: dict = {"pid": pid}
    def read(p):
        try:
            with open(p, "rb") as f:
                return f.read().decode("utf-8", "replace")
        except Exception as e:
            return f"<error: {e}>"
    def readlink(p):
        import os
        try:
            return os.readlink(p)
        except Exception as e:
            return f"<error: {e}>"
    out["cmdline"] = read(f"{base}/cmdline").replace("\x00", " ").strip()
    out["exe"] = readlink(f"{base}/exe")
    out["cwd"] = readlink(f"{base}/cwd")
    status = read(f"{base}/status")
    out["status"] = "\n".join(
        ln for ln in status.splitlines()
        if ln.startswith(("Name:", "Pid:", "PPid:", "Uid:", "VmRSS:",
                          "VmSize:", "VmSwap:", "Threads:", "State:"))
    )
    return out


@app.get("/v1/pods", dependencies=[Depends(require_bearer)])
def list_pods(only_unhealthy: bool = False) -> dict:
    """List k3s pods (all or only non-Running/Completed). kubectl
    against the system kubeconfig — read-only."""
    import subprocess
    filt = (" | awk '$4!=\"Running\" && $4!=\"Completed\"'"
            if only_unhealthy else "")
    cmd = (
        "for kc in /etc/rancher/k3s/k3s.yaml $HOME/.kube/config; do "
        "  [ -r \"$kc\" ] && { "
        f"    KUBECONFIG=\"$kc\" kubectl get pods -A -o wide --no-headers 2>&1{filt}; "
        "    exit 0; "
        "  }; "
        "done; echo '(no readable kubeconfig)'"
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True,
                        text=True, timeout=20)
    return {"rc": r.returncode, "out": r.stdout, "err": r.stderr[-2000:]}


@app.post("/v1/kick-pod", dependencies=[Depends(require_bearer)])
def kick_pod(body: dict) -> dict:
    """Delete one pod by namespace+name so k8s reschedules it. Used to
    revive a specific pod that has crashed but isn't auto-recovering
    (e.g. OOMKilled control-plane pod). Uses k8s default grace period
    — no --force. Names validated as DNS-1123 to prevent shell injection."""
    import subprocess, re
    ns = body.get("namespace", "")
    name = body.get("name", "")
    if not re.fullmatch(r"[a-z0-9.-]{1,253}", ns) \
            or not re.fullmatch(r"[a-z0-9.-]{1,253}", name):
        raise HTTPException(400,
            "namespace/name must be lowercase alphanumerics/./- only")
    cmd = (
        "for kc in /etc/rancher/k3s/k3s.yaml $HOME/.kube/config; do "
        "  [ -r \"$kc\" ] && { "
        f"    KUBECONFIG=\"$kc\" kubectl -n {ns} delete pod {name} 2>&1; "
        "    exit 0; "
        "  }; "
        "done; echo '(no readable kubeconfig)'"
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True,
                        text=True, timeout=30)
    return {"rc": r.returncode, "out": r.stdout, "err": r.stderr[-2000:]}


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
