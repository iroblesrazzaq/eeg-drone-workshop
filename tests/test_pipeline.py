from dataclasses import replace

import numpy as np
import pytest

from eeg_drone import solution
from eeg_drone.checks import check
from eeg_drone.config import Settings
from eeg_drone.control import Controller, Drone
from eeg_drone.replay import ReplayEngine, replay_chunks
from eeg_drone.signal import CausalBandpass, training_windows


def test_reference_passes_student_contract():
    check(solution)


def test_stateful_filter_matches_whole_recording(synthetic_run):
    settings = Settings()
    whole = CausalBandpass(160, 3, settings).process(synthetic_run.samples)
    streaming_filter = CausalBandpass(160, 3, settings)
    # Awkward chunk boundaries expose lost state or incorrect axis assumptions.
    streamed = np.concatenate([streaming_filter.process(chunk)
                               for chunk in replay_chunks(synthetic_run, 37)])
    np.testing.assert_allclose(streamed, whole, rtol=1e-12, atol=1e-15)


def test_windows_never_cross_cue_boundaries(synthetic_run):
    run = synthetic_run
    assert run.window_label(240, 560) == 1
    assert run.window_label(160, 480) is None  # onset delay not satisfied
    assert run.window_label(700, 1020) is None  # task → rest
    assert run.window_label(800, 1120) is None  # rest
    X, y, trial_ids = training_windows(run, Settings(), solution.extract_features)
    assert X.shape[1] == 6 and set(y) == {0, 1}
    assert len(set(trial_ids)) == len(run.cues)


def test_stream_predictions_match_offline_features(synthetic_run):
    settings = Settings()
    X, y, _ = training_windows(synthetic_run, settings, solution.extract_features)
    model = solution.fit_classifier(X, y)
    engine = ReplayEngine(synthetic_run, model, settings, solution)
    streamed = [p for chunk in replay_chunks(synthetic_run, 173) for p in engine.push(chunk)]
    whole = CausalBandpass(160, 3, settings).process(synthetic_run.samples)
    for prediction in streamed:
        stop = round(prediction.time_seconds * 160)
        features = solution.extract_features(whole[stop-320:stop], 160, settings.bands)
        expected = model.predict_proba(features[None, :])[0, list(model.classes_).index(1)]
        assert prediction.p_hands == pytest.approx(expected, abs=1e-12)
        if not prediction.enabled:
            assert prediction.velocity_mps == 0
            assert prediction.smoothed_p_hands == 0.5
    assert streamed[0].time_seconds == 2
    np.testing.assert_allclose(np.diff([p.time_seconds for p in streamed]), 0.25)


def test_future_samples_cannot_change_past_predictions(synthetic_run):
    settings = Settings()
    X, y, _ = training_windows(synthetic_run, settings, solution.extract_features)
    model = solution.fit_classifier(X, y)
    changed = synthetic_run.samples.copy()
    changed[1600:] = 0.1  # change only the future after 10 seconds
    altered_run = replace(synthetic_run, samples=changed)
    def predict(run):
        engine = ReplayEngine(run, model, settings, solution)
        return [p for chunk in replay_chunks(run, 37, seconds=10) for p in engine.push(chunk)]
    assert predict(synthetic_run) == predict(altered_run)


def test_rest_resets_smoothing_and_overrides_confidence():
    controller = Controller(Settings(), solution.probability_to_velocity)
    assert controller.update(1, enabled=True) > 0
    assert controller.update(1, enabled=False) == 0
    assert controller.smoothed == 0.5


def test_drone_clamps_altitude():
    drone = Drone()
    drone.set_vertical_velocity(100)
    drone.update(1)
    assert drone.altitude == drone.maximum
    drone.set_vertical_velocity(-100)
    drone.update(1)
    assert drone.altitude == drone.minimum


def test_rendered_and_fast_replay_have_same_trajectory(synthetic_run, tmp_path, monkeypatch):
    from eeg_drone.simulator import fly
    settings = Settings()
    X, y, _ = training_windows(synthetic_run, settings, solution.extract_features)
    bundle = {"model": solution.fit_classifier(X, y)}
    fast = fly(synthetic_run, bundle, settings, solution, headless=True,
               speed=0, output_dir=tmp_path / "fast")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    rendered = fly(synthetic_run, bundle, settings, solution, speed=40,
                   output_dir=tmp_path / "rendered", screenshot=tmp_path / "frame.png")
    assert rendered["final_altitude_m"] == pytest.approx(fast["final_altitude_m"], abs=1e-10)
    for key in ("predictions", "enabled_predictions", "up_commands", "down_commands", "hover_commands"):
        assert rendered[key] == fast[key]
    assert (tmp_path / "frame.png").stat().st_size > 0
