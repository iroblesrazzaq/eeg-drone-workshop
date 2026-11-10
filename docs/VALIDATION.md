# Reference run

Verified locally on November 10, 2026, with Python 3.12.14 on macOS ARM64.
Exact installed package versions are in `requirements-tested.txt`.

## Real PhysioNet EEG

- Participant: S001 (the default; no participant search was performed).
- Training: runs 6 + 10, 120 windows.
- Test: run 14, 60 windows from 15 task trials.
- Input: C3, Cz, C4 at 160 Hz; continuous causal 7–30 Hz bandpass.
- Features: six natural-log mu/beta band powers.
- Model: StandardScaler → default regularized LogisticRegression.
- Held-out window accuracy: **80.0%** (48/60).
- Held-out balanced accuracy: **79.91%**.
- Majority-class baseline: **53.33%**.
- Trial accuracy after averaging within-trial probabilities: **86.67%** (13/15).
- Confusion matrix, rows actual / columns predicted, order feet then hands:

```text
             predicted feet  predicted hands
actual feet       26               6
actual hands       6              22
```

The trial score summarizes the same test data. Overlapping windows are not
independent. These numbers describe this subject and held-out recording; they
are not estimates of live-headset accuracy, new-person transfer, or free-running
rest detection. The test run is also used for the workshop demonstration.

## Replay and tests

The reference model was saved, reloaded, and used to decode the full 125-second
held-out recording. Both up and down commands occurred. Rest/transition periods
produce hover through the cue gate. Replay emits predictions from 2.0 seconds
onward at 250 ms intervals (493 predictions over the full recording).

`pytest -q --integration`: **9 passed**. Verification covers:

- Synthetic feature checks, model contract, and probability-to-command mapping.
- Whole-recording versus chunked causal filtering with awkward chunk lengths.
- Fully contained task windows and onset delay.
- Identical offline/streaming probabilities at matching window endpoints.
- Invariance of earlier predictions to changes in future samples.
- Rest gating and smoothing reset; altitude bounds.
- Matching trajectories in fast headless replay and the rendered event loop.
- Real EDF download/loading, train/test split, model save/reload, and replay.
- Simulator frame rendered and visually inspected; evaluation plot inspected.

Reproduce with:

```bash
eeg-drone check --solution
pytest -q --integration
eeg-drone prepare --subject 1
eeg-drone train --subject 1 --solution --output-dir artifacts/reference
eeg-drone fly --model artifacts/reference/model.joblib --headless --speed 0 \
  --output-dir artifacts/reference --screenshot artifacts/reference/drone.png
```

Generated metrics and replay logs are in `artifacts/reference/` on the prepared
machine; they are not tracked in Git or bundled with the source archive.
