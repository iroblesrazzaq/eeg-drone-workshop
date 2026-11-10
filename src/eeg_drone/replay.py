"""Chunked inference. No future EEG is read by the model.

The cue gate knows the experiment schedule; it is NOT a learned rest detector.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .config import HANDS, Settings
from .control import Controller
from .signal import CausalBandpass


@dataclass
class Prediction:
    time_seconds: float
    p_hands: float
    smoothed_p_hands: float
    truth: int | None
    enabled: bool
    velocity_mps: float

    def to_dict(self):
        return asdict(self)


class ReplayEngine:
    def __init__(self, run, model, settings: Settings, implementation):
        self.run, self.model, self.settings = run, model, settings
        self.extract_features = implementation.extract_features
        self.filter = CausalBandpass(run.sfreq, len(run.channels), settings)
        self.controller = Controller(settings, implementation.probability_to_velocity)
        self.length = round(settings.window_seconds * run.sfreq)
        self.step = round(settings.prediction_seconds * run.sfreq)
        self.buffer = np.empty((0, len(run.channels)))
        self.samples_seen = 0
        self.next_prediction = self.length
        self.hands_column = list(model.classes_).index(HANDS)
        self.latest: Prediction | None = None

    def push(self, chunk: np.ndarray) -> list[Prediction]:
        """Accept arbitrary chunk sizes; emit at exact sample-based deadlines."""
        filtered = self.filter.process(chunk)
        predictions = []
        offset = 0
        while offset < len(filtered):
            take = min(len(filtered) - offset, self.next_prediction - self.samples_seen)
            self.buffer = np.concatenate((self.buffer, filtered[offset:offset + take]))[-self.length:]
            self.samples_seen += take
            offset += take
            if self.samples_seen == self.next_prediction:
                start, stop = self.samples_seen - self.length, self.samples_seen
                features = np.asarray(self.extract_features(
                    self.buffer, self.run.sfreq, self.settings.bands,
                ), dtype=float)
                expected = len(self.run.channels) * len(self.settings.bands)
                if features.shape != (expected,) or not np.isfinite(features).all():
                    raise ValueError("Replay features have the wrong shape or nonfinite values.")
                p_hands = float(self.model.predict_proba(features[None, :])[0, self.hands_column])
                # Labels never enter features or predict_proba. Here the known cue
                # schedule only decides whether the controller is enabled.
                truth = self.run.window_label(start, stop, self.settings.onset_delay_seconds)
                # A window ending exactly at cue offset is valid for scoring,
                # but its new command must not start moving during rest.
                enabled = truth is not None and self.run.task_at(stop) == truth
                velocity = self.controller.update(p_hands, enabled)
                self.latest = Prediction(
                    stop / self.run.sfreq, p_hands, self.controller.smoothed,
                    truth, enabled, velocity,
                )
                predictions.append(self.latest)
                self.next_prediction += self.step
        return predictions


def replay_chunks(run, chunk_samples: int, seconds: float | None = None):
    """A local stream source. Replace this boundary with a headset inlet later."""
    stop = len(run.samples) if seconds is None else min(len(run.samples), round(seconds * run.sfreq))
    for start in range(0, stop, chunk_samples):
        yield run.samples[start:min(start + chunk_samples, stop)]
