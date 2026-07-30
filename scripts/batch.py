"""Separate a batch of tracks without going through the HTTP API.

Jobs land in the normal storage root, so anything processed here also shows up
in the studio UI. Intended for experiment runs: point it at a folder, override
a knob, and compare the results afterwards.

    python scripts/batch.py tracks/
    python scripts/batch.py a.wav b.flac --overlap 16
    python scripts/batch.py tracks/ --no-autocast --tag no-autocast
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def collect(inputs: list[Path], suffixes: frozenset[str]) -> list[Path]:
    """Expand directories one level and keep only supported audio files."""
    found: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found.extend(
                child
                for child in sorted(item.iterdir())
                if child.is_file() and child.suffix.casefold() in suffixes
            )
        elif item.is_file():
            if item.suffix.casefold() not in suffixes:
                print(f"  skip  {item.name} (unsupported format)")
                continue
            found.append(item)
        else:
            print(f"  skip  {item} (not found)")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch stem separation for experiment runs."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio files or folders")
    parser.add_argument(
        "--overlap", type=int, help="Override STEMFLOW_MDXC_OVERLAP for this run"
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable mixed precision for reproducible output",
    )
    parser.add_argument(
        "--tag", help="Label stored on each job, for telling runs apart later"
    )
    parser.add_argument(
        "--storage", type=Path, help="Storage root (defaults to STEMFLOW_STORAGE_ROOT)"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failure instead of stopping",
    )
    args = parser.parse_args()

    # Set before importing the worker so inference_settings() sees the overrides.
    if args.overlap is not None:
        os.environ["STEMFLOW_MDXC_OVERLAP"] = str(args.overlap)
    if args.no_autocast:
        os.environ["STEMFLOW_USE_AUTOCAST"] = "false"

    from server.config import AUDIO_SUFFIXES, Settings, safe_name
    from server.store import JobStore
    from worker.separate_v4 import (
        SeparationCancelled,
        inference_settings,
        separate_job,
    )

    settings = Settings.load()
    storage_root = args.storage.resolve() if args.storage else settings.storage_root
    # Batch output is the point of the run, so never let retention reap it.
    store = JobStore(storage_root, retention_hours=0)

    tracks = collect(args.inputs, AUDIO_SUFFIXES)
    if not tracks:
        print("No supported audio files found.")
        return 1

    knobs = inference_settings()
    print(f"Storage   {storage_root}")
    print(f"Settings  overlap={knobs['mdxcOverlap']} autocast={knobs['autocast']}")
    if args.tag:
        print(f"Tag       {args.tag}")
    print(f"Tracks    {len(tracks)}\n")

    failures: list[tuple[Path, str]] = []
    completed_count = 0
    for index, track in enumerate(tracks, start=1):
        label = f"[{index}/{len(tracks)}] {track.name}"
        print(label, flush=True)
        suffix = track.suffix.casefold()
        job, metadata = store.create(safe_name(track.name) or f"audio{suffix}", suffix)
        shutil.copy2(track, job / "original" / f"source{suffix}")
        updates: dict[str, object] = {
            "sizeBytes": track.stat().st_size,
            "status": "processing",
            "sourcePath": str(track.resolve()),
        }
        if args.tag:
            updates["tag"] = args.tag
        metadata = store.update(metadata["jobId"], **updates)

        last_stage = ""

        def report(value: int, stage: str) -> None:
            nonlocal last_stage
            store.update(metadata["jobId"], progress=value, stage=stage)
            if stage != last_stage:
                last_stage = stage
                print(f"    {value:3d}%  {stage}", flush=True)

        try:
            completed = separate_job(job, metadata, report, None)
        except SeparationCancelled:
            store.update(metadata["jobId"], status="cancelled", stage="Cancelled")
            print("    cancelled")
            return 130
        except KeyboardInterrupt:
            store.update(metadata["jobId"], status="cancelled", stage="Interrupted")
            print("\n    interrupted")
            return 130
        except Exception as exc:  # noqa: BLE001 - report and move to the next track
            store.update(
                metadata["jobId"],
                status="failed",
                stage="Separation failed",
                error=str(exc)[-8000:],
            )
            failures.append((track, str(exc)))
            print(f"    FAILED  {exc}")
            if args.keep_going:
                continue
            print("\nStopping. Pass --keep-going to continue past failures.")
            break
        else:
            completed_count += 1
            error = completed["separation"]["reconstructionError"]
            print(
                f"    done   {completed['jobId']}  "
                f"reconstruction error {error:.2e}",
                flush=True,
            )

    print(f"\n{completed_count}/{len(tracks)} completed.")
    for track, message in failures:
        print(f"  {track.name}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
