# Can you explain the system?

Answer using the code, diagrams, or a small numerical example. Aim to explain
the reason behind each choice, not just name the library. Reference answers are
in `ANSWER_KEY.md`; try these first without opening it.

## High-level questions

**H1. What exactly are the two classes?** Identify imagined versus actual
movement, both hands versus left/right hand, the relevant runs, and the meaning
of T0/T1/T2 in those runs.

**H2. Why C3, Cz, and C4?** What do mu/beta power features capture? Why might
three channels and two bands fail even if the participant follows the task?

**H3. Why features before logistic regression?** Compare six log-band-power
features with flattening 320 samples × 3 channels. What useful structure and
information does band power retain or discard?

**H4. Why standardize, and why inside a pipeline?** Explain what is learned by
the scaler and what goes wrong if you fit it before separating train and test.

**H5. Why split by recording rather than randomly by window?** Give an example
of two overlapping windows that would leak information. What stronger claim
would require holding out entire people or sessions?

**H6. What does the test accuracy establish?** Compare window accuracy,
balanced accuracy, majority-class baseline, and trial accuracy. Are overlapping
test windows independent observations?

**H7. What is logistic regression's output?** Explain a score, a probability,
and a final command. Does a reported probability of 0.9 guarantee 90% real-world
correctness?

**H8. How can this demo hover during rest?** Identify exactly where it uses
the known task schedule. Why would a confidence deadband alone not solve rest
detection? What would you need for uncued, continuous control?

**H9. What creates latency?** Discuss the 2-second window, 0.5-second onset
delay, 250 ms prediction interval, causal filter, smoothing, and rendering.
Does predicting four times per second mean the system reacts in 250 ms?

**H10. What changes with a real headset?** Explain what can stay and what must
be checked or retrained: stream source, channel order, units, sample rate,
timestamps, artifacts, person, and electrode placement.

## Code-comprehension questions

**C1. Trace the shapes.** Starting at `raw.get_data()` in `data.py`, give the
shape and units before and after `.T`, one window, the feature vector, `X`,
`features[None, :]`, and `predict_proba` for one window.

**C2. Trace the PSD.** In your `extract_features`, why is `axis=0` necessary?
At 160 Hz with `nperseg=160`, what is the frequency-bin spacing? Why multiply
the PSD sum by that spacing? What happens to a channel's log power when its
amplitude doubles?

**C3. Explain filter state.** In `CausalBandpass`, what does `self.state`
remember? Why is its shape `(n_sections, 2, n_channels)`? What breaks if it is
reset for every chunk or replaced by forward/backward filtering?

**C4. Count windows.** A cue starts at sample 800 and stops at 1456. With the
default settings and 160 Hz, list every training window's `[start, stop)`
indices. Which parameters change your count?

**C5. Explain the containment check.** For the cue in C4, what does
`window_label(800, 1120)` return? What about `(880, 1200)` and `(1200, 1520)`?
Why check the whole window rather than only its final sample?

**C6. Trace fitting versus prediction.** Find where the scaler is fitted,
where the model is fitted, and where each is reused without fitting. Show
that run 14 is not passed to `fit_classifier`.

**C7. Identify the correct probability column.** Why does `ReplayEngine` use
`list(model.classes_).index(HANDS)` instead of assuming column 0 means hands?
What direction error would the wrong column cause?

**C8. Trace chunk scheduling.** At 160 Hz, how long is an 8-sample chunk?
How many samples make a window and prediction interval? If `push` receives a
173-sample chunk crossing several deadlines, how does it avoid missing a
prediction or putting future samples into an earlier window?

**C9. Calculate smoothing and commands.** With initial smoothed probability
0.5, alpha 0.3, and successive raw probabilities 0.9, 0.9, 0.1, compute the
three smoothed probabilities and commands. What changes if the third update
has `enabled=False`?

**C10. Separate labels from features.** Trace `truth` and `enabled` in
`ReplayEngine.push`. Is the class label passed to `predict_proba`? Is the
controller completely independent of the annotation schedule? Explain both.

**C11. Explain the clocks and motion.** Why does `fly` update the drone with
the previous command before processing a newly arrived chunk? Why does doubling
playback speed not change the EEG sample rate or the trajectory at the same
EEG timestamp? How does `Drone.update` keep it in bounds?

**C12. Interpret the tests.** What do
`test_stateful_filter_matches_whole_recording`,
`test_stream_predictions_match_offline_features`, and
`test_future_samples_cannot_change_past_predictions` establish? Why can all
three pass even if EEG classification accuracy is poor?

## Demonstrate understanding

**D1. Diagnose a plausible bug.** Your offline score is decent, but replay
always moves down. Propose three distinct causes and a concrete inspection or
test for each. Include a feature mismatch, a class mapping issue, and a control
issue rather than three versions of the same explanation.

**D2. Design an honest next experiment.** You want to try CSP, five channel
sets, and ten regularization values. Design selection and final evaluation
without repeatedly optimizing on run 14. Explain which generalization claim
your final test supports.

For a short exit check, ask H5, H8, C1, C8, and C9. A member who can explain
these and trace one prediction from samples to velocity understands the core
workshop pipeline.
