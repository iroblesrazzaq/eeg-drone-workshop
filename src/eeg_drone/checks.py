"""Small student-facing checks using synthetic data; no download required."""
import numpy as np

from .config import Settings


def check(implementation, task="all"):
    settings = Settings()
    if task in ("all", "features"):
        sfreq = 160.0
        t = np.arange(320) / sfreq
        window = 1e-5 * np.column_stack((np.sin(2*np.pi*10*t),
                                        np.sin(2*np.pi*20*t), np.sin(2*np.pi*10*t)))
        features = np.asarray(implementation.extract_features(window, sfreq, settings.bands))
        assert features.shape == (6,), "Return six features, ordered by channel then band."
        assert np.isfinite(features).all(), "Features must be finite."
        f = features.reshape(3, 2)
        assert f[0, 0] > f[0, 1] and f[1, 1] > f[1, 0], "Check PSD axis, band masks, and feature order."
        doubled = implementation.extract_features(2 * window, sfreq, settings.bands)
        assert np.isclose(doubled[0] - features[0], np.log(4), atol=0.01), "Doubling amplitude multiplies power by four."
        zeros = implementation.extract_features(np.zeros_like(window), sfreq, settings.bands)
        assert np.isfinite(zeros).all(), "Protect log(0) with a small epsilon."
        print("PASS Task 1: features")
    if task in ("all", "model"):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 6))
        y = (X[:, 0] > 0).astype(int)
        model = implementation.fit_classifier(X, y)
        assert set(model.classes_) == {0, 1}
        assert (model.predict(X) == y).mean() > 0.9, "Fit the model on the supplied X and y."
        assert model.predict_proba(X[:2]).shape == (2, 2)
        print("PASS Task 2: classifier (synthetic sanity check, not an EEG score)")
    if task in ("all", "control"):
        mapping = implementation.probability_to_velocity
        for probability, expected in [(0.1, -0.65), (0.4, -0.65), (0.5, 0), (0.6, 0.65), (0.9, 0.65)]:
            assert mapping(probability, 0.65, 0.4, 0.6) == expected, "Check direction and inclusive thresholds."
        print("PASS Task 3: controller mapping")
