import json

import numpy as np
import pytest

from eeg_drone.config import Settings
from eeg_drone.data import load_run
from eeg_drone.simulator import fly
from eeg_drone.training import get_implementation, load_bundle, train


@pytest.mark.integration
def test_physionet_train_save_reload_and_fly(tmp_path):
    from pathlib import Path
    # Reuse local downloads but keep test artifacts out of the instructor demo.
    metrics = train(1, Path("data"), tmp_path, solution=True)
    assert metrics["train_runs"] == [6, 10] and metrics["test_run"] == 14
    assert metrics["n_train_windows"] > 0 and metrics["n_test_trials"] > 0
    assert 0 <= metrics["window_accuracy"] <= 1  # no arbitrary score requirement
    bundle, settings = load_bundle(tmp_path / "model.joblib")
    run = load_run(1, 14, Path("data"), settings)
    assert run.sfreq == 160 and run.channels == Settings().channels
    summary = fly(run, bundle, settings, get_implementation(True),
                  headless=True, speed=0, seconds=30, output_dir=tmp_path)
    assert summary["predictions"] > 0 and summary["enabled_predictions"] > 0
    assert summary["up_commands"] + summary["down_commands"] > 0
    assert np.isfinite(summary["final_altitude_m"])
    assert json.loads((tmp_path / "replay_summary.json").read_text()) == summary
