from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def sync_index_from_gcs(*, gcs_uri: str, local_dir: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    dst = Path(local_dir)
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("gcloud"):
        cmd = ["gcloud", "storage", "rsync", "-r", gcs_uri, str(dst)]
        subprocess.run(cmd, check=True)
        return
    if shutil.which("gsutil"):
        cmd = ["gsutil", "-m", "rsync", "-r", gcs_uri, str(dst)]
        subprocess.run(cmd, check=True)
        return
    raise RuntimeError("missing dependency: gcloud or gsutil")
