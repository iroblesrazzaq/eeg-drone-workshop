"""Download PhysioNet with MNE, then expose a small, explicit array interface."""
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from .config import FEET, HANDS, Settings


@dataclass(frozen=True)
class Cue:
    start: int  # inclusive sample index
    stop: int   # exclusive sample index
    label: int


@dataclass
class EEGRun:
    subject: int
    run_id: int
    samples: np.ndarray  # (time, channels), volts
    sfreq: float
    channels: tuple[str, ...]
    cues: tuple[Cue, ...]  # only hands/feet; gaps are rest or transitions

    def task_at(self, sample: int) -> int | None:
        for cue in self.cues:
            if cue.start <= sample < cue.stop:
                return cue.label
        return None

    def window_label(self, start: int, stop: int,
                     onset_delay_seconds: float = 0.5) -> int | None:
        """Return a label only if the WHOLE window is inside a stable task cue."""
        delay = round(onset_delay_seconds * self.sfreq)
        for cue in self.cues:
            if start >= cue.start + delay and stop <= cue.stop:
                return cue.label
        return None


def load_run(subject: int, run_id: int, data_dir: Path,
             settings: Settings) -> EEGRun:
    if not 1 <= subject <= 109:
        raise ValueError("PhysioNet subjects are numbered 1 through 109.")
    if run_id not in (6, 10, 14):
        raise ValueError("This workshop uses only hands/feet IMAGERY runs 6, 10, 14.")
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = mne.datasets.eegbci.load_data(
        subject, [run_id], path=str(data_dir), update_path=False, verbose="ERROR",
    )
    raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)  # e.g. "C3.." → "C3"
    raw.pick(list(settings.channels))
    raw.reorder_channels(list(settings.channels))
    sfreq = float(raw.info["sfreq"])
    samples = raw.get_data().T.copy()  # MNE returns volts, not microvolts
    cues = []
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description,
    ):
        label = {"T1": HANDS, "T2": FEET}.get(description)
        if label is not None:
            start = max(0, round(float(onset) * sfreq))
            stop = min(len(samples), round((float(onset) + float(duration)) * sfreq))
            if stop > start:
                cues.append(Cue(start, stop, label))
    if not cues or {c.label for c in cues} != {FEET, HANDS}:
        raise ValueError("Expected nonempty T1/T2 annotations with durations.")
    return EEGRun(subject, run_id, samples, sfreq, tuple(raw.ch_names), tuple(cues))
