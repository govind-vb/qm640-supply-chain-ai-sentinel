"""Shared paths for notebooks and scripts."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    # src/ -> project root
    return Path(__file__).resolve().parents[1]


def load_env() -> Path:
    root = project_root()
    load_dotenv(root / ".env")
    return root


def ensure_output_dirs(root: Path | None = None) -> dict[str, Path]:
    root = root or project_root()
    iter_name = os.getenv("OUTPUT_ITER", "latest")
    paths = {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "outputs": root / "notebooks" / "outputs" / iter_name,
        "figures": root / "notebooks" / "outputs" / iter_name / "figures",
        "tables": root / "notebooks" / "outputs" / iter_name / "tables",
    }
    for key in ("raw", "processed", "outputs", "figures", "tables"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths
