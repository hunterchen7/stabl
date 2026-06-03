"""Job runners — thin shells around the existing stabl scripts. Each runner
takes a Job, reads its params, writes its output to OUTPUTS_DIR, and stores
the resulting file's id on the job."""
import subprocess
import sys
import uuid
from pathlib import Path

from . import jobs, settings


def _run_subprocess(job: jobs.Job, cmd: list[str]) -> None:
    jobs.append_log(job, f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout
    for line in proc.stdout:
        jobs.append_log(job, line.rstrip())
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"command exited {code}")


def _output_path(suffix: str = ".mp4") -> tuple[str, Path]:
    file_id = uuid.uuid4().hex
    return file_id, settings.OUTPUTS_DIR / f"{file_id}{suffix}"


def _resolve_clip(p: dict) -> Path:
    """Accept either `clip_id` (uploads) or `pictures_path` (under PICTURES_ROOT)."""
    if p.get("pictures_path"):
        if settings.PICTURES_ROOT is None:
            raise RuntimeError("pictures_path used but STABL_PICTURES_ROOT not set")
        rel = Path(p["pictures_path"])
        if rel.is_absolute() or any(part == ".." for part in rel.parts):
            raise ValueError("pictures_path must be relative, no ..")
        clip = settings.PICTURES_ROOT / rel
        if not clip.exists():
            raise FileNotFoundError(f"not in Pictures: {rel}")
        return clip
    if p.get("clip_id"):
        clip = settings.UPLOADS_DIR / f"{p['clip_id']}.mp4"
        if not clip.exists():
            raise FileNotFoundError(f"clip {p['clip_id']} not in uploads")
        return clip
    raise ValueError("need either clip_id or pictures_path")


def _maybe_trim(job: jobs.Job, clip_path: Path, p: dict) -> Path:
    """If start_sec/duration_sec are set, stream-copy a trimmed segment into
    OUTPUTS_DIR and return that path. Otherwise return clip_path unchanged."""
    if "start_sec" not in p and "duration_sec" not in p:
        return clip_path
    trim_path = settings.OUTPUTS_DIR / f"{job.id}_trim.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    if "start_sec" in p:
        cmd += ["-ss", str(p["start_sec"])]
    cmd += ["-i", str(clip_path)]
    if "duration_sec" in p:
        cmd += ["-t", str(p["duration_sec"])]
    cmd += ["-c", "copy", str(trim_path)]
    _run_subprocess(job, cmd)
    return trim_path


def run_klt_affine(job: jobs.Job) -> None:
    """Multi-feature KLT + rigid affine warp."""
    p = job.params
    clip_path = _maybe_trim(job, _resolve_clip(p), p)
    file_id, out_path = _output_path()
    script = settings.REPO_ROOT / "klt_affine.py"
    cmd = [
        sys.executable, str(script),
        "--input", str(clip_path),
        "--output", str(out_path),
        "--n_features", str(p.get("n_features", 15)),
        "--crop_w", str(p.get("crop_w", 0)),  # 0 = auto
        "--crop_h", str(p.get("crop_h", 0)),
        "--offset_x", str(p.get("offset_x", 0)),
        "--offset_y", str(p.get("offset_y", 0)),
    ]
    if p.get("dlc_h5"):
        cmd += ["--dlc_h5", p["dlc_h5"]]
    if p.get("feature_bbox"):
        cmd += ["--feature_bbox", p["feature_bbox"]]
    if p.get("no_rotation"):
        cmd.append("--no_rotation")
    if p.get("initial_points"):
        cmd += ["--initial_points", p["initial_points"]]
    if p.get("err_thresh") is not None:
        cmd += ["--err_thresh", str(p["err_thresh"])]
    if p.get("ransac_thresh") is not None:
        cmd += ["--ransac_thresh", str(p["ransac_thresh"])]
    if p.get("auto_pick"):
        cmd.append("--auto_pick")
    if p.get("auto_pick_pool") is not None:
        cmd += ["--auto_pick_pool", str(p["auto_pick_pool"])]
    _run_subprocess(job, cmd)
    job.output_file_id = file_id


def run_stabl_track_csv(job: jobs.Job) -> None:
    """Run stabl.py with --track_csv. Source clip via clip_id or pictures_path."""
    p = job.params
    clip_path = _resolve_clip(p)
    csv_path = settings.UPLOADS_DIR / f"{p['csv_id']}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    file_id, out_path = _output_path()
    script = settings.REPO_ROOT / "stabl.py"
    cmd = [
        sys.executable, str(script),
        str(clip_path), str(out_path),
        "--track_csv", str(csv_path),
        "--smoothing_window", str(p.get("smoothing_window", 1)),
        "--max_pixel_shift", str(p.get("max_pixel_shift", 200)),
        "--bitrate", p.get("bitrate", "50M"),
    ]
    if p.get("auto_crop"):
        cmd += ["--auto_crop", "--auto_crop_percentile", str(p.get("auto_crop_percentile", 2))]
    if p.get("width"):
        cmd += ["--width", str(p["width"]), "--height", str(p["height"])]
    cmd += ["--color_offset_x", str(p.get("color_offset_x", 0)),
            "--color_offset_y", str(p.get("color_offset_y", 0))]
    if p.get("visualize"):
        cmd.append("--visualize")
    _run_subprocess(job, cmd)
    job.output_file_id = file_id


def run_dlc_stabl(job: jobs.Job) -> None:
    """DLC SuperAnimal-Bird inference + body-cluster CSV + stabl.py with that
    CSV. Default tracks the body center; pass `preset` to switch ("head",
    "torso_center", or "custom" with `parts`).

    All the usual stabl knobs from run_stabl_track_csv are supported.
    """
    p = job.params
    clip_path = _maybe_trim(job, _resolve_clip(p), p)

    # Step 1: DLC inference -> H5 in DATA_DIR/jobs/<job_id>/
    work = settings.DATA_DIR / "jobs" / job.id
    work.mkdir(parents=True, exist_ok=True)
    h5_path = work / "track.h5"
    if p.get("reuse_h5_job_id"):
        src = settings.DATA_DIR / "jobs" / p["reuse_h5_job_id"] / "track.h5"
        if not src.exists():
            raise FileNotFoundError(f"no H5 from job {p['reuse_h5_job_id']}")
        jobs.append_log(job, f"reusing H5 from job {p['reuse_h5_job_id']}")
        h5_path.symlink_to(src)
    else:
        cmd = [
            sys.executable, str(settings.REPO_ROOT / "dlc_infer.py"),
            "--input", str(clip_path),
            "--output_h5", str(h5_path),
            "--device", p.get("dlc_device", "cuda"),
        ]
        _run_subprocess(job, cmd)

    # Step 2: H5 -> body-cluster CSV
    csv_path = work / "track.csv"
    cmd = [
        sys.executable, str(settings.REPO_ROOT / "dlc_track.py"),
        "--h5", str(h5_path),
        "--out_csv", str(csv_path),
        "--preset", p.get("preset", "body"),
        "--min_likelihood", str(p.get("min_likelihood", 0.5)),
    ]
    if p.get("parts"):
        cmd += ["--parts", p["parts"]]
    _run_subprocess(job, cmd)

    # Step 3: stabl with track CSV
    file_id, out_path = _output_path()
    script = settings.REPO_ROOT / "stabl.py"
    cmd = [
        sys.executable, str(script),
        str(clip_path), str(out_path),
        "--track_csv", str(csv_path),
        "--smoothing_window", str(p.get("smoothing_window", 1)),
        "--max_pixel_shift", str(p.get("max_pixel_shift", 200)),
        "--bitrate", p.get("bitrate", "50M"),
    ]
    if p.get("auto_crop", True):
        cmd += ["--auto_crop", "--auto_crop_percentile", str(p.get("auto_crop_percentile", 2))]
    if p.get("width"):
        cmd += ["--width", str(p["width"]), "--height", str(p["height"])]
    cmd += ["--color_offset_x", str(p.get("color_offset_x", 0)),
            "--color_offset_y", str(p.get("color_offset_y", 0))]
    if p.get("visualize"):
        cmd.append("--visualize")
    _run_subprocess(job, cmd)
    job.output_file_id = file_id


def _add_klt_encode_flags(cmd: list, p: dict) -> None:
    """Shared encode-side flags for klt_affine.py runs."""
    if p.get("preview"):
        cmd.append("--preview")
    if p.get("bitrate"):
        cmd += ["--bitrate", p["bitrate"]]
    if p.get("no_rotation"):
        cmd.append("--no_rotation")
    if p.get("ransac_thresh") is not None:
        cmd += ["--ransac_thresh", str(p["ransac_thresh"])]


def run_cotracker_stabl(job: jobs.Job) -> None:
    """CoTracker3 tracking + affine warp. Tracks N points jointly via the
    transformer-based tracker (survives occlusion/wing flap), then runs the
    same affine RANSAC + warp as klt-affine.
    """
    p = job.params
    clip_path = _maybe_trim(job, _resolve_clip(p), p)
    work = settings.DATA_DIR / "jobs" / job.id
    work.mkdir(parents=True, exist_ok=True)
    tracks_json = work / "tracks.json"

    cmd = [
        sys.executable, str(settings.REPO_ROOT / "cotracker_track.py"),
        "--input", str(clip_path),
        "--output_json", str(tracks_json),
        "--n_points", str(p.get("n_points", 80)),
        "--max_track_dim", str(p.get("max_track_dim", 640)),
        "--mode", p.get("cotracker_mode", "offline"),
    ]
    if p.get("mask_circle"):
        cmd += ["--mask_circle", p["mask_circle"]]
    if p.get("query_points"):
        cmd += ["--query_points", p["query_points"]]
    if p.get("expand_patch") is not None:
        cmd += ["--expand_patch", str(p["expand_patch"])]
    _run_subprocess(job, cmd)

    file_id, out_path = _output_path()
    script = settings.REPO_ROOT / "klt_affine.py"
    cmd = [
        sys.executable, str(script),
        "--input", str(clip_path),
        "--output", str(out_path),
        "--tracks_json", str(tracks_json),
        "--vis_thresh", str(p.get("vis_thresh", 0.7)),
        "--crop_w", str(p.get("crop_w", 0)),
        "--crop_h", str(p.get("crop_h", 0)),
        "--offset_x", str(p.get("offset_x", 0)),
        "--offset_y", str(p.get("offset_y", 0)),
        "--bitrate", p.get("bitrate", "50M"),
    ]
    if p.get("no_rotation"):
        cmd.append("--no_rotation")
    if p.get("no_consensus_filter"):
        cmd.append("--no_consensus_filter")
    if p.get("draw_debug"):
        cmd.append("--debug_overlay")
    if p.get("ransac_thresh") is not None:
        cmd += ["--ransac_thresh", str(p["ransac_thresh"])]
    if p.get("auto_crop"):
        cmd.append("--auto_crop")
        cmd += ["--auto_crop_pct", str(p.get("auto_crop_pct", 2.0))]
        cmd += ["--aspect", p.get("aspect", "16:9")]
        cmd += ["--bias_x", str(p.get("bias_x", 0)), "--bias_y", str(p.get("bias_y", 0))]
    _run_subprocess(job, cmd)
    job.output_file_id = file_id


def run_cotracker_track(job: jobs.Job) -> None:
    """CoTracker3 tracking only — produces a tracks.json that the Mac can
    download and warp+encode locally. Much smaller transfer than the full
    stabilized video. ~3-8 MB JSON vs 100+ MB MP4."""
    p = job.params
    clip_path = _maybe_trim(job, _resolve_clip(p), p)
    file_id = uuid.uuid4().hex if False else None
    # Use _output_path's id but with a .json suffix
    import uuid as _uuid
    file_id = _uuid.uuid4().hex
    out_path = settings.OUTPUTS_DIR / f"{file_id}.json"
    cmd = [
        sys.executable, str(settings.REPO_ROOT / "cotracker_track.py"),
        "--input", str(clip_path),
        "--output_json", str(out_path),
        "--n_points", str(p.get("n_points", 80)),
        "--max_track_dim", str(p.get("max_track_dim", 480)),
        "--mode", p.get("cotracker_mode", "online"),
    ]
    if p.get("mask_circle"):
        cmd += ["--mask_circle", p["mask_circle"]]
    if p.get("query_points"):
        cmd += ["--query_points", p["query_points"]]
    if p.get("expand_patch") is not None:
        cmd += ["--expand_patch", str(p["expand_patch"])]
    _run_subprocess(job, cmd)
    job.output_file_id = file_id


RUNNERS = {
    "klt-affine": run_klt_affine,
    "stabl-track-csv": run_stabl_track_csv,
    "dlc-stabl": run_dlc_stabl,
    "cotracker-stabl": run_cotracker_stabl,
    "cotracker-track": run_cotracker_track,
}
