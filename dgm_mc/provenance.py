"""Portable manifests; never collect credentials or personal filesystem paths."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p)
            for folder in ("dgm_mc", "configs") for p in sorted((ROOT / folder).rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json"}}


def manifest(config, command):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), config=config,
                command=command, source_sha256=source_hashes(),
                git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                environment=dict(python=platform.python_version(), platform=platform.system(),
                                 machine=platform.machine(), numpy=np.__version__, scipy=scipy.__version__,
                                 torch=torch.__version__, torch_threads=torch.get_num_threads(), device="cpu",
                                 dtype="float64"))


def seed_for(*labels):
    """Disjoint deterministic namespaces, stable across process hash randomization."""
    digest = hashlib.sha256(json.dumps(labels).encode()).digest()
    return int.from_bytes(digest[:8], "little")
