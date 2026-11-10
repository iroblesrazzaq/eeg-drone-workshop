# Build the EEG decoder

Work in pairs. One person edits; the other explains shapes and checks assumptions.
Swap roles after the model is trained. Keep a terminal open in the repo root.

## 0. Understand your input

Run `eeg-drone prepare --subject 1` and inspect `artifacts/raw_eeg.png`.

We use 160 samples per second from three motor-area channels: C3, Cz, C4.
Arrays have shape **(time, channels)**. MNE loads signals in volts; the plot
multiplies by one million to show microvolts. Pink intervals are hands imagery;
blue intervals are feet imagery. The uncolored intervals are rest.

Checkpoint: A two-second window has shape `(320, 3)`. Explain why each number
is present and which axis a frequency analysis should use.

## 1. Turn a window into six features

Implement `extract_features` in `src/eeg_drone/student.py`.

1. Estimate a power spectral density with `scipy.signal.welch`. Use `fs=sfreq`,
   `nperseg=int(sfreq)`, and `axis=0`.
2. Select mu frequencies `8 <= f < 12` and beta frequencies `13 <= f < 30`.
3. Sum the selected PSD bins and multiply by the frequency spacing to get band
   power. Do this separately for each channel.
4. Take `np.log(power + 1e-20)`.
5. Return one flat array in this order:
   `C3_mu, C3_beta, Cz_mu, Cz_beta, C4_mu, C4_beta`.

Read `help(welch)` or its official documentation if you are unsure what it returns.
Don't feed the cue label, timestamp, run number, or trial number into the model.

```bash
eeg-drone check --task features
```

Checkpoint: Explain why a clean 10 Hz sine wave should have a larger mu feature
than beta feature, and what doubling its amplitude does to log power.

## 2. Fit a classifier and evaluate unseen EEG

Implement `fit_classifier` using `make_pipeline(StandardScaler(),
LogisticRegression(max_iter=1000, random_state=42))`. Fit on the supplied `X, y`
and return the fitted pipeline. A pipeline prevents you from accidentally
fitting the scaler to test data.

```bash
eeg-drone check --task model
eeg-drone train --subject 1
```

The provided training code splits by recording: runs 6 + 10 for training, 14 for
testing. Overlapping windows stay on their own side of this split. There is no
hyperparameter search on run 14. If you repeatedly select models using that
score, run 14 has become validation data, and you need another final test set.

Open `artifacts/evaluation.png` and `artifacts/metrics.json`. Compare accuracy
with the majority-class baseline. Inspect per-class errors and feature weights.
Trial-level accuracy averages probabilities over each trial's windows; it is
another view of the same data, not extra independent evidence.

Checkpoint: Which errors would push the drone in the wrong direction? Does your
test show that the model would work for a different person? Explain.

## 3. Turn probabilities into commands

Implement `probability_to_velocity`:

- `P(hands) >= 0.6`: return `+speed` (up).
- `P(hands) <= 0.4`: return `-speed` (down).
- Otherwise: return `0.0` (hover).

Use the supplied arguments; don't hard-code them. The provided controller first
checks the task gate, then smooths the probability with an exponential moving
average. When the gate is off, it hovers and resets smoothing to 0.5.

```bash
eeg-drone check --task control
eeg-drone fly
```

Checkpoint: Explain why the first command of a trial is delayed. A full window
must start at least 0.5 seconds after the cue begins, so the earliest eligible
prediction is roughly 2.5 seconds after onset, plus the prediction schedule.

## 4. Trace the real-time path

Read `ReplayEngine.push` in `replay.py`. Data arrives in small chunks; a stateful
filter processes each chunk, a buffer retains the latest two seconds, and sample
counters schedule a prediction every 40 samples. Rendering runs separately at
up to 60 frames per second.

Open `artifacts/replay.csv`. Find a row with movement, a hover row, and a row
where `enabled=False`. Explain each one using the probability and cue gate.

The fast version runs the same decoding without waiting for wall-clock time:

```bash
eeg-drone fly --headless --speed 0 --seconds 30
```

## Extensions if you finish

- Compare 1-, 2-, and 3-second windows. Discuss latency and signal estimation.
- Change smoothing or thresholds and measure command stability versus delay.
- Try more motor-area channels; retrain and keep the feature order consistent.
- Add a validation split within the training runs before tuning regularization.
- Try CSP + logistic regression. Fit CSP only inside the training split/fold.
- Add an explicit rest class and evaluate false movement during rest.
- Replace the replay source with an LSL inlet. Discuss channel names, units,
  timestamps, dropped samples, new-person calibration, and headset placement.

Don't describe a high held-out replay score as proof of live headset control.
