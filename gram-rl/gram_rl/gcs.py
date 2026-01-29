from __future__ import annotations

import subprocess
from pathlib import Path


def sync_index_from_gcs(*, gcs_uri: str, local_dir: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    dst = Path(local_dir)
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["gcloud", "storage", "rsync", "-r", gcs_uri, str(dst)]
    subprocess.run(cmd, check=True)

