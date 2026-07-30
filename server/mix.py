from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

from worker.separate_v4 import STEM_NAMES


def render_mix(stems_dir: Path, levels: dict[str, float]) -> io.BytesIO:
    audio: list[tuple[np.ndarray, int]] = []
    for name in STEM_NAMES:
        path = stems_dir / f"{name}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} stem")
        data, rate = sf.read(path, dtype="float64", always_2d=True)
        value = float(levels.get(name, 1.0))
        if not 0 <= value <= 1.5:
            raise ValueError(f"Invalid level for {name}")
        audio.append((data * value, rate))
    sample_rates = {rate for _, rate in audio}
    if len(sample_rates) != 1:
        raise ValueError("Stem sample rates do not match")
    length = min(len(data) for data, _ in audio)
    mixed = sum(data[:length] for data, _ in audio)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1:
        mixed /= peak
    output = io.BytesIO()
    sf.write(output, mixed.astype(np.float32), sample_rates.pop(), format="WAV", subtype="PCM_24")
    output.seek(0)
    return output
