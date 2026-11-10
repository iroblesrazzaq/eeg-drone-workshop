# Instructor reference answers

These are the essential ideas; accept equivalent explanations. Ask students to
point to the relevant code and work a small example when answers sound memorized.

## High-level

**H1.** Both hands imagined versus both feet imagined. Runs 6, 10, 14. In
these runs T1=hands, T2=feet, T0=rest. Internal labels are hands=1, feet=0.
Other runs use different tasks; blindly reusing the label mapping is wrong.

**H2.** C3/C4 lie over left/right sensorimotor regions; Cz samples the medial
region relevant to leg representation. Mu/beta rhythms can change during motor
imagery. Power is a crude spatial/spectral summary; noise, artifacts, reference,
individual anatomy, and weak imagery effects can overwhelm it. A decrease or a
particular weight sign is not guaranteed for every person/channel.

**H3.** Band power compresses 960 correlated measurements into six features
related to a plausible motor-imagery signal, reducing dimensionality. It loses
precise timing, phase, waveform morphology, and rich spatial relationships.
Flattening is possible but needs different assumptions and more care about
dimensionality, alignment, and regularization; it is not the easiest baseline.

**H4.** The scaler learns each feature's training mean and standard deviation.
Scaling helps regularized logistic regression treat units/scales consistently.
The pipeline learns scaling during training and reuses it on test windows.
Using test statistics contaminates the held-out evaluation, even without labels.

**H5.** Two-second windows starting at 1.0 and 1.5 s share 1.5 s of signal.
Randomly assigning them to train/test puts nearly identical data on both sides.
Holding out a run avoids that overlap; holding out people or sessions is needed
to test those stronger forms of generalization. Trials/runs also share artifacts
and other correlations beyond exact sample overlap.

**H6.** Window accuracy measures this subject's held-out windows. Balanced
accuracy averages class recalls; the majority baseline shows a trivial rule's
accuracy. Trial accuracy averages probabilities first, producing one decision
per cue. Overlapping test windows are correlated, not independent replications;
15 task trials are much less evidence than hundreds of independent trials.

**H7.** Logistic regression forms a weighted linear score plus intercept and
passes it through a sigmoid for binary class probability. Thresholding and
smoothing produce commands. Estimated probabilities may be miscalibrated and
may become unreliable under a new person/device/session or artifacts.

**H8.** `window_label` checks whether an entire window is inside a task cue;
`enabled` gates the controller. It knows the experiment schedule, not an inferred
rest state. Binary classifiers can be confidently wrong on rest. Uncued control
needs an evaluated rest/intent detector or third class and measurements of
false activations; a manual arming switch is another explicit control design.

**H9.** The buffer needs 2 s of history. The cue gate additionally requires
the window to begin 0.5 s after onset, so the first enabled prediction is about
2.5 s after cue onset, rounded to the 250 ms prediction grid. Filter delay and
smoothing add temporal effects. Rendering at 60 Hz doesn't accelerate EEG
evidence collection. The refresh rate is not end-to-end reaction time.

**H10.** The feature/model/control boundary can remain, but replace file replay
with a source such as LSL. Verify timestamp continuity, missing samples, exact
channel order/names, volts versus µV, and sample rate; redesign/resample filtering
as needed. Match preprocessing and retrain/calibrate on the person and headset.
Electrode placement, reference, noise, and behavior change the distribution.
The current annotation-based gate also needs replacement.

## Code comprehension

**C1.** MNE: `(3, n_samples)`, volts, after channels are picked. Transposed:
`(n_samples, 3)`, volts. Window `(320, 3)`. Features `(6,)`, natural log of band
power expressed numerically in V² with epsilon. `X` is `(n_windows, 6)`.
`features[None, :]` is `(1, 6)`; probability output is `(1, 2)`.

**C2.** Time is axis 0. With 160 samples/segment at 160 Hz, bin spacing is
1 Hz. PSD has units V²/Hz; summing and multiplying by Hz approximates integrated
power. Doubling amplitude multiplies power by four, increasing natural log power
by `ln(4) ≈ 1.386` when power is much larger than epsilon.

**C3.** Delay states carry each IIR section's history across chunks. Two
states are required per second-order section per channel, and axis 0 is replaced
by length 2. Resetting introduces discontinuities/transients. Forward/backward
filtering uses future data and does not match a causal online pipeline.

**C4.** Window=320 samples, step=80, delay=80. Starts: 880, 960, 1040, 1120.
Windows: `[880,1200)`, `[960,1280)`, `[1040,1360)`, `[1120,1440)`.
The next would end beyond 1456. Cue duration, onset delay, window duration,
training step, and sample rate affect the count.

**C5.** First returns `None` because it violates onset delay. Second returns
that cue's label. Third returns `None` because it crosses the cue boundary.
Checking only the end could assign a task label to a window mostly containing
rest or another task.

**C6.** `model.fit(X_train, y_train)` inside the student function fits both
pipeline stages. `training.train` builds those arrays only from `TRAIN_RUNS`
before loading/evaluating the test run. `predict`/`predict_proba` reuse the
fitted scaler and coefficients, including during replay. No test-time fit.

**C7.** Probability columns correspond to `classes_`, not display names or an
arbitrary assumed order. Internal hands label is 1. Taking the feet probability
and naming it hands reverses the intended direction.

**C8.** 50 ms per chunk; 320 samples/window; 40 samples/prediction. `push`
filters the chunk causally, then appends only up to the next prediction deadline,
emits a prediction, and repeats for any remaining samples. The buffer is cropped
to the most recent 320. The stateful filter is causal even when called on a
larger block; later output samples are not used by earlier windows.

**C9.** `0.62 → up`, `0.704 → up`, `0.5228 → hover`, with speed 0.65 m/s.
If the third update is disabled, it immediately returns zero and resets the
smoothed value to 0.5, without applying the EMA or student mapping.

**C10.** `predict_proba` only receives extracted EEG features. `truth` is the
label of a fully contained cue window; the controller receives whether it
exists, not its hands/feet value. The model doesn't see the answer, but the
movement gate depends on annotation timing. This is explicitly cue-based control.

**C11.** A new chunk arrives after time elapsed under the previous command;
its prediction cannot act retroactively. Integration uses EEG time deltas, so
speed changes wall-clock pacing only. At a fixed processed sample index, the
same predictions/dynamics result. Height is clamped to minimum/maximum; this is
a kinematic visualization, not a model of drone thrust, inertia, or aerodynamics.

**C12.** The tests establish chunk/whole filter equivalence, matching streamed
and offline probabilities at identical window endpoints, and no dependence on
EEG samples later than the prediction time. They verify processing correctness,
not whether the features discriminate biological classes. An internally
consistent but uninformative model can pass all three.

## Demonstrations

**D1.** Examples: wrong channel order/units or feature order → compare a saved
window's offline and replay feature arrays. Wrong class column → inspect
`classes_` and p_hands on known held-out trials. Reversed threshold mapping →
run control checks and inject probabilities 0.1/0.9. Excess smoothing, thresholds,
gate duration, or weak class recall → inspect CSV raw/smoothed values and gate.
Poor biological generalization is also possible; do not assume every failure is
a code bug.

**D2.** Select features, CSP, and regularization using grouped trial/run folds
within the training data; scaler and CSP must fit inside each fold. Lock the
pipeline and evaluate run 14 once. This supports a held-out-run claim for that
person only. To claim new-person transfer, use subject-grouped selection and
entirely held-out subjects. More data may be needed for stable selection.

## Suggested assessment

For each exit question (H5, H8, C1, C8, C9), score 0 = incorrect, 1 = names the
idea, 2 = explains it with the code or a correct numerical example. An 8/10 with
no zero on leakage or rest gating is a useful discussion target, not a formal
validated exam. Ask each pair to trace one prediction independently.
