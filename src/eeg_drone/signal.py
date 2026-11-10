"""The same causal filter is used during training and replay."""
import numpy as np
from scipy.signal import butter, sosfilt

from .config import Settings


class CausalBandpass:
    def __init__(self, sfreq: float, n_channels: int, settings: Settings):
        self.sos = butter(4, [settings.low_hz, settings.high_hz],
                          btype="bandpass", fs=sfreq, output="sos")
        # axis=0 is time, so filter state replaces that axis with length 2.
        self.state = np.zeros((len(self.sos), 2, n_channels))

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) == 0:
            return chunk.copy()
        filtered, self.state = sosfilt(self.sos, chunk, axis=0, zi=self.state)
        return filtered


def training_windows(run, settings: Settings, extract_features):
    """Filter continuously, including rest; then take windows within each cue."""
    filtered = CausalBandpass(run.sfreq, len(run.channels), settings).process(run.samples)
    length = round(settings.window_seconds * run.sfreq)
    step = round(settings.training_step_seconds * run.sfreq)
    delay = round(settings.onset_delay_seconds * run.sfreq)
    features, labels, trial_ids = [], [], []
    for trial_id, cue in enumerate(run.cues):
        for start in range(cue.start + delay, cue.stop - length + 1, step):
            vector = np.asarray(extract_features(
                filtered[start:start + length], run.sfreq, settings.bands,
            ), dtype=float)
            expected = len(run.channels) * len(settings.bands)
            if vector.shape != (expected,) or not np.isfinite(vector).all():
                raise ValueError(f"Features must be {expected} finite numbers; got {vector.shape}.")
            features.append(vector)
            labels.append(cue.label)
            trial_ids.append(trial_id)
    if not features:
        raise ValueError("No valid training windows; check cue durations and window size.")
    return np.stack(features), np.asarray(labels), np.asarray(trial_ids)
