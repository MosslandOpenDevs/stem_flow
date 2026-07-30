from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path | None = None) -> bool:
    """Load .env so the paths a user fills in actually take effect.

    Real environment variables win, so a shell export or a CLI override is never
    silently replaced by the file. Returns whether a file was found.
    """
    target = path or PROJECT_ROOT / ".env"
    if not target.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - declared in requirements.txt
        return False
    load_dotenv(target, override=False)
    return True


# Import-time so every entry point (API, batch runner, tests) sees the same
# configuration without having to remember to call it.
load_env_file()

AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"})
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_name(name: str) -> str:
    """Sanitize an incoming filename for use inside a job directory."""
    return SAFE_FILENAME.sub("_", Path(name).name[:180])


def split_origins(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    root: Path
    storage_root: Path
    max_upload_bytes: int
    retention_hours: int
    worker_count: int
    allowed_origins: tuple[str, ...]

    @classmethod
    def load(cls) -> "Settings":
        root = Path(__file__).resolve().parents[1]
        return cls(
            root=root,
            storage_root=Path(
                os.getenv("STEMFLOW_STORAGE_ROOT", root / "storage")
            ).resolve(),
            max_upload_bytes=int(
                os.getenv("STEMFLOW_MAX_UPLOAD_BYTES", 500 * 1024 * 1024)
            ),
            retention_hours=int(os.getenv("STEMFLOW_RETENTION_HOURS", 24)),
            worker_count=max(1, int(os.getenv("STEMFLOW_WORKER_COUNT", 1))),
            # Local dev origins only. Add a deployed origin through
            # STEMFLOW_ALLOWED_ORIGINS rather than hardcoding it here.
            allowed_origins=split_origins(
                os.getenv(
                    "STEMFLOW_ALLOWED_ORIGINS",
                    "http://localhost:3000,http://127.0.0.1:3000,"
                    "http://localhost:3001,http://127.0.0.1:3001",
                )
            ),
        )
