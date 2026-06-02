"""Server config — paths and the API-key hash, all from env."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("STABL_DATA_DIR", Path.home() / ".stabl-server"))
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
JOBS_FILE = DATA_DIR / "jobs.json"

# Root of the user's Pictures library on this box. When STABL_PICTURES_ROOT is
# set, jobs can accept a `pictures_path` (relative to this root) and skip the
# upload step entirely — useful when Pictures is already mounted from another
# system (e.g. Olares user volumes).
PICTURES_ROOT = Path(os.environ["STABL_PICTURES_ROOT"]) if os.environ.get("STABL_PICTURES_ROOT") else None

for d in (UPLOADS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

API_KEY_HASH = os.environ.get("STABL_API_KEY_HASH", "")
PORT = int(os.environ.get("STABL_PORT", "8000"))

REPO_ROOT = Path(__file__).resolve().parent.parent
