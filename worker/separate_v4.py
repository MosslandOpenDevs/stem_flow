"""Production-oriented StemFlow v4 separation pipeline.

This module is dependency-light on purpose: the API process only needs NumPy
and SoundFile. Model inference is delegated to the audio-separator executable
from the dedicated CUDA environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
SIX_STEM_MODEL = "BS-Roformer-SW.ckpt"
STEM_NAMES = ("vocals", "drums", "bass", "guitar", "piano", "other")
INSTRUMENT_STEMS = ("drums", "bass", "guitar", "piano")
ProgressCallback = Callable[[int, str], None]


class SeparationCancelled(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def model_directory() -> Path:
    """Resolve the model directory as an absolute path.

    audio-separator defaults to the POSIX-looking '/tmp/audio-separator-models/',
    which on Windows resolves against the *current drive* — starting the API from
    another drive would silently look in the wrong place and try to download.
    Resolving here and passing --model_file_dir explicitly removes that
    dependency, and every candidate is derived from the project directory rather
    than a hardcoded absolute path.
    """
    configured = os.getenv("STEMFLOW_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (PROJECT_ROOT / "models", PROJECT_ROOT / "vendor" / "models")
    # Prefer a project-local directory that actually holds the assets, so a
    # half-built models/ folder never shadows a working install.
    for candidate in candidates:
        if candidate.is_dir() and not missing_model_files(candidate):
            return candidate
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Nothing set up yet: name the documented location so the "missing assets"
    # error points somewhere useful. No absolute path is baked into the project.
    return candidates[0]


def required_model_files() -> tuple[str, ...]:
    """Every local asset the pipeline needs in order to never reach the network.

    The .yaml configs matter as much as the checkpoints: without them
    audio-separator falls back to a hash lookup that downloads UVR model data.
    download_checks.json is fetched from GitHub when absent, even though the
    models themselves are already present.
    """
    return (
        VOCAL_MODEL,
        str(Path(VOCAL_MODEL).with_suffix(".yaml")),
        SIX_STEM_MODEL,
        str(Path(SIX_STEM_MODEL).with_suffix(".yaml")),
        "download_checks.json",
    )


def missing_model_files(directory: Path | None = None) -> tuple[str, ...]:
    directory = directory or model_directory()
    return tuple(
        name for name in required_model_files() if not (directory / name).is_file()
    )


def verify_offline_assets(directory: Path | None = None) -> None:
    """Fail fast and locally rather than stalling on a doomed download."""
    directory = directory or model_directory()
    missing = missing_model_files(directory)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} model asset(s) in {directory}: "
            + ", ".join(missing)
            + ". Separation needs these locally; set STEMFLOW_MODEL_DIR if they "
            "live elsewhere."
        )


def inference_settings() -> dict:
    """Separator knobs worth varying between experiment runs.

    Read once per job so that changing the environment mid-run cannot make the
    two model passes disagree, or diverge from what metadata records.
    """
    return {
        "mdxcOverlap": max(1, int(os.getenv("STEMFLOW_MDXC_OVERLAP", 8))),
        "autocast": env_flag("STEMFLOW_USE_AUTOCAST", True),
    }


def resolve_executable(
    env_name: str, bundled: tuple[Path, ...], command: str
) -> str:
    """Resolve a tool, preferring what the project owns.

    Order: explicit env var, then anything vendored inside the project, then
    PATH. Bundled beats PATH so that a self-contained checkout is deterministic
    instead of picking up whichever build happens to be installed system-wide.
    """
    configured = os.getenv(env_name)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"{env_name} points to a missing file")
    for candidate in bundled:
        if candidate.is_file():
            return str(candidate)
    discovered = shutil.which(command)
    if discovered:
        return discovered
    raise FileNotFoundError(
        f"{command} was not found. Set {env_name} to its absolute executable path."
    )


def bundled_candidates(*relative: str) -> tuple[Path, ...]:
    """Project-local locations, plus anything matching a glob under them."""
    found: list[Path] = []
    for item in relative:
        if any(char in item for char in "*?"):
            found.extend(sorted(PROJECT_ROOT.glob(item)))
        else:
            found.append(PROJECT_ROOT / item)
    return tuple(found)


def audio_separator_executable() -> str:
    return resolve_executable(
        "STEMFLOW_SEPARATOR_EXE",
        bundled_candidates(
            ".sepenv/Scripts/audio-separator.exe",
            "vendor/sepenv/Scripts/audio-separator.exe",
        ),
        "audio-separator",
    )


def ffmpeg_executable() -> str:
    return resolve_executable(
        "STEMFLOW_FFMPEG_EXE",
        bundled_candidates(
            "vendor/ffmpeg/ffmpeg.exe",
            "vendor/ffmpeg/bin/ffmpeg.exe",
            "vendor/ffmpeg/*/bin/ffmpeg.exe",
        ),
        "ffmpeg",
    )


def check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event and cancel_event.is_set():
        raise SeparationCancelled("Separation was cancelled")


def run_separator(
    source: Path,
    output: Path,
    model: str,
    sample_rate: int,
    cancel_event: threading.Event | None = None,
    settings: dict | None = None,
    log_path: Path | None = None,
) -> None:
    check_cancel(cancel_event)
    output.mkdir(parents=True, exist_ok=True)
    settings = settings or inference_settings()
    command = [
        audio_separator_executable(),
        str(source),
        "--model_filename",
        model,
        # Absolute, so resolution never depends on the current drive.
        "--model_file_dir",
        str(model_directory()),
        "--output_dir",
        str(output),
        "--output_format",
        "WAV",
        # Pinned: letting the separator rescale would invalidate the residual.
        "--normalization",
        "1.0",
        "--sample_rate",
        str(sample_rate),
        "--mdxc_overlap",
        str(settings["mdxcOverlap"]),
        "--use_soundfile",
    ]
    if settings["autocast"]:
        command.append("--use_autocast")
    process_env = os.environ.copy()
    ffmpeg_dir = str(Path(ffmpeg_executable()).parent)
    process_env["PATH"] = ffmpeg_dir + os.pathsep + process_env.get("PATH", "")
    if env_flag("STEMFLOW_OFFLINE", True):
        # Belt and braces. Assets are verified present before separation starts,
        # so nothing should be fetched; if a future version tries anyway, point
        # it at a dead loopback port so it fails in milliseconds instead of
        # stalling on the separator's 300s download timeout.
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            process_env[name] = "http://127.0.0.1:9"
        for name in ("NO_PROXY", "no_proxy"):
            process_env.pop(name, None)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=process_env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    # Line-buffered so a hung run still leaves a readable log behind.
    log = (
        log_path.open("w", encoding="utf-8", errors="replace", buffering=1)
        if log_path
        else None
    )
    try:
        if log:
            log.write(f"$ {' '.join(command)}\n\n")
        while True:
            if cancel_event and cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise SeparationCancelled("Separation was cancelled")
            line = process.stdout.readline()
            if line:
                if log:
                    log.write(line)
                output_lines.append(line.rstrip())
                output_lines = output_lines[-40:]
            if process.poll() is not None:
                for trailing in process.stdout.readlines():
                    if log:
                        log.write(trailing)
                    output_lines.append(trailing.rstrip())
                break
    finally:
        if log:
            log.close()
    if process.returncode:
        details = "\n".join(output_lines[-20:])
        raise RuntimeError(f"audio-separator failed for {model}\n{details}")


def find_output(directory: Path, needle: str) -> Path | None:
    """Locate a separator output, or None when the stem was silent.

    audio-separator skips writing a stem whose peak is below 1e-6 (see
    write_audio_soundfile) while still logging it as saved, so a missing file
    means "this stem is silent", not "something failed". A track with no piano
    part is ordinary input, not an error.
    """
    matches = sorted(
        p for p in directory.glob("*.wav") if needle.casefold() in p.name.casefold()
    )
    return matches[0] if matches else None


def aligned_audio(
    path: Path, channels: int, sample_rate: int, length: int
) -> np.ndarray:
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"Sample-rate mismatch for {path.name}")
    if data.shape[1] != channels:
        raise ValueError(f"Channel mismatch for {path.name}")
    if len(data) < length:
        raise ValueError(f"Unexpected short output for {path.name}")
    return data[:length]


def write_metadata(job: Path, metadata: dict) -> None:
    target = job / "metadata.json"
    temporary = job / "metadata.json.tmp"
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)


def encode_preview(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(destination),
        ],
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def separate_job(
    job: Path,
    metadata: dict,
    progress: ProgressCallback,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Run the v4 pipeline inside an already-created UUID job directory."""
    original_dir = job / "original"
    stage1, stage2 = job / "_stage1", job / "_stage2"
    stems_dir, preview_dir = job / "stems", job / "preview"
    exports_dir, logs_dir = job / "exports", job / "logs"
    for directory in (stage1, stage2, stems_dir, preview_dir, exports_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    sources = list(original_dir.glob("source.*"))
    if len(sources) != 1:
        raise FileNotFoundError("The job source file is missing")
    stored_source = sources[0]
    settings = inference_settings()

    try:
        progress(8, "Validating source audio")
        verify_offline_assets()
        source_info = sf.info(stored_source)
        if source_info.frames <= 0 or source_info.channels not in (1, 2):
            raise ValueError("Unsupported or empty audio file")

        progress(12, "Separating vocals")
        run_separator(
            stored_source,
            stage1,
            VOCAL_MODEL,
            source_info.samplerate,
            cancel_event,
            settings,
            logs_dir / "stage1-vocals.log",
        )
        vocals_path = find_output(stage1, "vocals")
        instrumental_path = find_output(stage1, "instrumental")

        paths: dict[str, Path | None] = {"vocals": vocals_path}
        if instrumental_path is None:
            # Nothing left after the vocal pass, so there is nothing to split.
            paths.update(dict.fromkeys(INSTRUMENT_STEMS))
        else:
            progress(46, "Separating instruments")
            run_separator(
                instrumental_path,
                stage2,
                SIX_STEM_MODEL,
                source_info.samplerate,
                cancel_event,
                settings,
                logs_dir / "stage2-instruments.log",
            )
            check_cancel(cancel_event)
            for name in INSTRUMENT_STEMS:
                paths[name] = find_output(stage2, name)

        progress(82, "Aligning and reconstructing stems")
        mix, rate = sf.read(stored_source, dtype="float64", always_2d=True)
        present = {name: path for name, path in paths.items() if path is not None}
        length = min(
            len(mix), *(sf.info(path).frames for path in present.values())
        ) if present else len(mix)
        if length <= 0:
            raise ValueError("The separator returned empty audio")
        mix = mix[:length]
        # A stem with no file was silent, so it contributes exactly zero.
        silent_stems = tuple(name for name, path in paths.items() if path is None)
        audio = {
            name: (
                aligned_audio(path, mix.shape[1], rate, length)
                if path is not None
                else np.zeros_like(mix)
            )
            for name, path in paths.items()
        }
        # v4 invariant: the six float stems reconstruct the aligned original.
        audio["other"] = mix - sum(
            audio[name]
            for name in ("vocals", "drums", "bass", "guitar", "piano")
        )

        progress(88, "Writing lossless stems")
        # Accumulate what actually lands on disk, so the invariant is measured
        # after the float32 cast rather than only in float64 memory.
        written_total = np.zeros_like(mix)
        for name in STEM_NAMES:
            check_cancel(cancel_event)
            wav_path = stems_dir / f"{name}.wav"
            stem_data = audio[name].astype(np.float32)
            sf.write(wav_path, stem_data, rate, subtype="FLOAT")
            written_total += stem_data
        reconstruction_error = float(np.max(np.abs(written_total - mix)))

        progress(92, "Encoding browser previews")
        for name in STEM_NAMES:
            check_cancel(cancel_event)
            encode_preview(stems_dir / f"{name}.wav", preview_dir / f"{name}.mp3")

        progress(97, "Packaging stem archive")
        shutil.make_archive(str(exports_dir / "stems"), "zip", stems_dir)
        completed = {
            **metadata,
            "status": "completed",
            "progress": 100,
            "stage": "Ready",
            "duration": length / rate,
            "sampleRate": rate,
            "channels": mix.shape[1],
            "completedAt": utc_now(),
            # Recorded so two runs of the same track stay comparable.
            "separation": {
                "vocalModel": VOCAL_MODEL,
                "instrumentModel": SIX_STEM_MODEL,
                **settings,
                # Peak |sum(stems) - source| as written. Expect ~1e-7 from the
                # float32 cast; anything larger means the invariant broke.
                "reconstructionError": reconstruction_error,
                # Stems the separator omitted as silent, filled with zeros here.
                "silentStems": list(silent_stems),
            },
            "files": {name: f"stems/{name}.wav" for name in STEM_NAMES},
            "previews": {name: f"preview/{name}.mp3" for name in STEM_NAMES},
            "archive": "exports/stems.zip",
        }
        write_metadata(job, completed)
        return completed
    finally:
        shutil.rmtree(stage1, ignore_errors=True)
        shutil.rmtree(stage2, ignore_errors=True)
