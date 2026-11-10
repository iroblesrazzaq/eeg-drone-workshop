"""Your three tasks. Start with docs/WORKSHOP.md; keep these signatures unchanged.

Everything uses NumPy arrays, not MNE objects, after loading the recording.
"""
import numpy as np
from scipy.signal import welch  # Hint for Task 1
from sklearn.linear_model import LogisticRegression  # Hint for Task 2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(window: np.ndarray, sfreq: float, bands) -> np.ndarray:
    """TASK 1: return channel-major log band powers as a 1-D array.

    window: (n_samples, n_channels), already causally bandpass-filtered, in volts.
    bands: (("mu", 8.0, 12.0), ("beta", 13.0, 30.0)).
    Return: shape (n_channels * len(bands),), usually (6,).

    Steps:
      1. Welch PSD along axis=0, using a 1-second segment (nperseg=int(sfreq)).
      2. For each channel, then each band, select low <= frequency < high.
      3. Integrate PSD: sum(selected PSD) * frequency spacing.
      4. Take natural log after adding 1e-20 to avoid log(0).
      5. Return a flat float array in the documented order.
    """
    raise NotImplementedError("Task 1: implement extract_features in student.py")


def fit_classifier(X: np.ndarray, y: np.ndarray):
    """TASK 2: fit and return StandardScaler → LogisticRegression.

    X: (n_training_windows, n_features); y: (n_training_windows,).
    Labels: 0=feet, 1=hands. Only training data reaches this function.
    Hint: make_pipeline(...); use max_iter=1000 and random_state=42.
    The returned object must support predict, predict_proba, and classes_.
    """
    raise NotImplementedError("Task 2: implement fit_classifier in student.py")


def probability_to_velocity(p_hands: float, speed: float,
                            low: float, high: float) -> float:
    """TASK 3: map an already smoothed P(hands) to vertical velocity in m/s.

    p_hands >= high → +speed (up)
    p_hands <= low  → -speed (down)
    Otherwise      → 0.0 (hover)

    The provided controller handles rest gating and smoothing before calling you.
    """
    raise NotImplementedError("Task 3: implement probability_to_velocity in student.py")
