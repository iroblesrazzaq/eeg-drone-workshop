# Run the meeting

## Before people arrive

1. Use Python 3.12 on the presentation machine. Install the repo and run the
   reference commands in the README. Open the Pygame window once.
2. Download subject 1 in advance. Keep this repo's `data/` folder available on
   participant machines if internet is unreliable. Preserve the folder layout
   and PhysioNet attribution. The repo does not bundle EEG files.
3. Have participants install Python/dependencies before the meeting if possible.
   A GUI needs a local desktop session. Colab can train and run headless replay,
   but this repo does not expose its Pygame window inside a notebook.
4. Keep `artifacts/reference/` separate from student outputs. Reference model
   artifacts are local conveniences; students should learn to retrain them.
5. Read `VALIDATION.md`. Subject choice is fixed for this workshop, not a
   best-subject search. Performance is not guaranteed for other people.

## 75-minute agenda

| Time | Activity | Observable checkpoint |
| --- | --- | --- |
| 0–10 min | Show finished demo, explain cues, inspect EEG | Explain `(320, 3)` and hands/feet labels |
| 10–25 min | Pairs implement features | Feature checks pass; explain mu/beta and log power |
| 25–40 min | Implement model and evaluate run 14 | Explain held-out run and confusion matrix |
| 40–55 min | Implement control and fly | Correct up/down/hover; explain rest gate |
| 55–65 min | Read replay code and inspect CSV | Trace a window and a command |
| 65–75 min | Exit questions, discuss next experiment | Explain leakage, latency, and live-data limits |

Allow 15 extra minutes for setup or extensions. For a 45-minute session, provide
Task 1 as a walkthrough, then have students implement the model and controller.

## Teaching prompts

- Start with: “We are learning a mapping from measured EEG rhythms to two
  experimental tasks, then deciding how a simulated drone responds.”
- Call it **recorded-EEG replay**. The annotation schedule supplies training
  labels and gates movement; it is not a live free-running brain controller.
- Ask a student to predict the feature dimension before running the code.
- Ask what happens to a binary model during rest before revealing the gate.
- Explain that the test run is now used for teaching/demo. Repeatedly changing
  parameters to improve its score turns it into a validation set.
- Don't substitute a neural network to rescue an unexplained failure. Check
  the labels, split, features, class mapping, and noise first.

## Common problems

| Symptom | Next action |
| --- | --- |
| `NotImplementedError` | Expected starter behavior; implement the named task, or use `train --solution` for the instructor demo |
| `eeg-drone` not found | Activate `.venv` and install `pip install -e '.[dev]'`; try `python -m eeg_drone --help` |
| Python 3.14 install fails | Use supported Python 3.11–3.13; 3.12 is verified |
| PhysioNet download fails | Check connectivity, rerun `prepare`, or copy the predownloaded `data/` tree |
| No desktop/display | Use `fly --headless --speed 0`; inspect CSV and a `--screenshot` |
| Drone waits at start | Expected: buffer fill plus fully contained post-onset task window; inspect enabled flag |
| Frequent hover | Inspect smoothed probabilities and thresholds; hover may reflect a weak decoder or rest gate |
| Poor score | Compare with baseline, per-class recall, and feature plots; individual variability is real |
| Features changed after training | Retrain; do not reuse the old model with new features |
| Another subject | Prepare/train with `--subject N`; saved model determines which subject is replayed |

The simulator uses `Drone.set_vertical_velocity(v)`, positive up. That small
interface is the later integration point for another simulator. The current
model has no aerodynamic physics and no real drone/network command support.

## What a live version would require

Keep the feature/model/controller interfaces. Replace the local file source
with an inlet (for example, MNE-LSL), respect acquisition timestamps, and handle
gaps. Verify preprocessing, units, channels, and calibration on the actual user.
Replace cue gating with explicit arming or a validated rest/intent detector.
Collect live validation data before claiming the recorded-data score transfers.
