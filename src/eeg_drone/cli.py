import argparse
import json
from pathlib import Path

from .config import TEST_RUN, TRAIN_RUNS, Settings


def parser():
    root = argparse.ArgumentParser(description="Hands/feet EEG → simulated drone workshop")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download three imagery runs and plot EEG")
    prepare.add_argument("--subject", type=int, default=1)
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train = commands.add_parser("train", help="Train on runs 6+10; evaluate run 14")
    train.add_argument("--subject", type=int, default=1)
    train.add_argument("--data-dir", type=Path, default=Path("data"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--solution", action="store_true", help="Use the complete instructor solution")
    fly = commands.add_parser("fly", help="Replay held-out EEG and control the simulated drone")
    fly.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    fly.add_argument("--data-dir", type=Path, default=Path("data"))
    fly.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    fly.add_argument("--seconds", type=float, help="Stop after this many EEG seconds")
    fly.add_argument("--speed", type=float, default=1.0, help="Playback speed; 0=fastest (headless only)")
    fly.add_argument("--headless", action="store_true")
    fly.add_argument("--screenshot", type=Path, help="Save the final simulator frame")
    check = commands.add_parser("check", help="Check student functions without downloading EEG")
    check.add_argument("--task", choices=["all", "features", "model", "control"], default="all")
    check.add_argument("--solution", action="store_true")
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            from .data import load_run
            settings = Settings()
            runs = [load_run(args.subject, run_id, args.data_dir, settings)
                    for run_id in (*TRAIN_RUNS, TEST_RUN)]
            for run in runs:
                print(f"S{run.subject:03d} R{run.run_id:02d}: {len(run.samples)/run.sfreq:.1f}s, "
                      f"{run.sfreq:g} Hz, {run.channels}, {len(run.cues)} task trials")
            _plot_raw(runs[0], args.output_dir)
            print(f"EEG plot: {args.output_dir / 'raw_eeg.png'}")
        elif args.command == "check":
            from .checks import check
            from .training import get_implementation
            check(get_implementation(args.solution), args.task)
        elif args.command == "train":
            from .training import train
            metrics = train(args.subject, args.data_dir, args.output_dir, args.solution)
            print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
            print(f"Saved model.joblib, metrics.json, evaluation.png in {args.output_dir}")
        elif args.command == "fly":
            from .data import load_run
            from .simulator import fly
            from .training import get_implementation, load_bundle
            bundle, settings = load_bundle(args.model)
            run = load_run(bundle["subject"], bundle["test_run"], args.data_dir, settings)
            if run.sfreq != bundle["sfreq"] or tuple(run.channels) != tuple(settings.channels):
                raise ValueError("Recording channels or sample rate do not match the trained model.")
            print(f"Using {bundle['implementation']} features and control; cue-gated RECORDED EEG replay.")
            summary = fly(run, bundle, settings, get_implementation(bundle["implementation"] == "solution"),
                          headless=args.headless, speed=args.speed, seconds=args.seconds,
                          output_dir=args.output_dir, screenshot=args.screenshot)
            print(json.dumps(summary, indent=2))
    except NotImplementedError as error:
        raise SystemExit(f"\n{error}\nSee docs/WORKSHOP.md. Instructor demo: eeg-drone train --solution") from None
    except (FileNotFoundError, ValueError, AssertionError) as error:
        raise SystemExit(f"\n{type(error).__name__}: {error}") from None


def _plot_raw(run, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .config import HANDS

    duration = min(40, len(run.samples) / run.sfreq)
    stop = round(duration * run.sfreq)
    time = np.arange(stop) / run.sfreq
    fig, axes = plt.subplots(len(run.channels), 1, figsize=(12, 6), sharex=True, layout="constrained")
    for channel, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(time, run.samples[:stop, channel] * 1e6, linewidth=0.5, color="#304f68")
        ax.set_ylabel(f"{run.channels[channel]} (µV)")
        for cue in run.cues:
            if cue.start / run.sfreq < duration:
                ax.axvspan(cue.start / run.sfreq, min(duration, cue.stop / run.sfreq),
                           color="#e8b8c7" if cue.label == HANDS else "#b9d9eb", alpha=0.55)
    axes[0].set_title("Raw EEG · pink: hands imagery · blue: feet imagery · white: rest")
    axes[-1].set_xlabel("Time (s)")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "raw_eeg.png", dpi=150)
    plt.close(fig)
