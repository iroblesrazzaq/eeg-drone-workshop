import numpy as np
import pytest

from eeg_drone.data import Cue, EEGRun


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", help="Download and test real EEG")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Add --integration to download real EEG"))


@pytest.fixture
def synthetic_run():
    sfreq = 160.0
    rng = np.random.default_rng(7)
    t = np.arange(160 * 22) / sfreq
    samples = rng.normal(0, 1e-6, (len(t), 3))
    cues = (Cue(160, 800, 1), Cue(1120, 1760, 0), Cue(2080, 2720, 1))
    for cue in cues:
        # Clearly separable synthetic data, not evidence about real EEG accuracy.
        frequency = 10 if cue.label == 1 else 20
        samples[cue.start:cue.stop] += 1e-5 * np.sin(
            2*np.pi*frequency*t[cue.start:cue.stop, None],
        )
    return EEGRun(1, 14, samples, sfreq, ("C3", "Cz", "C4"), cues)
