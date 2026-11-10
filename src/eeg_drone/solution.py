"""Instructor reference. Students work in student.py instead."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(window: np.ndarray, sfreq: float, bands) -> np.ndarray:
    # Welch averages power estimates from overlapping 1-second segments.
    frequencies, psd = welch(window, fs=sfreq, nperseg=int(sfreq), axis=0)
    df = frequencies[1] - frequencies[0]
    features = []
    for channel in range(window.shape[1]):
        for _, low, high in bands:
            mask = (frequencies >= low) & (frequencies < high)
            power = psd[mask, channel].sum() * df
            features.append(np.log(power + 1e-20))
    return np.asarray(features)


def fit_classifier(X: np.ndarray, y: np.ndarray):
    # Fitting the pipeline learns scaling ONLY from training windows.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X, y)
    return model


def probability_to_velocity(p_hands: float, speed: float,
                            low: float, high: float) -> float:
    if p_hands >= high:
        return speed
    if p_hands <= low:
        return -speed
    return 0.0
