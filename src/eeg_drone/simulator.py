"""A small Pygame view. The decoder and drone also run without a display."""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from .config import LABELS
from .control import Drone
from .replay import ReplayEngine


class Renderer:
    def __init__(self, settings, hidden=False):
        self.settings = settings
        if hidden:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((1060, 660))
        pygame.display.set_caption("EEG → Drone | PhysioNet replay")
        self.font = pygame.font.SysFont("Arial", 21)
        self.small = pygame.font.SysFont("Arial", 17)
        self.title = pygame.font.SysFont("Arial", 32, bold=True)
        self.clock = pygame.time.Clock()

    def text(self, text, x, y, color=(221, 231, 243), font=None):
        self.screen.blit((font or self.font).render(text, True, color), (x, y))

    def draw(self, run, drone, latest, elapsed, paused=False):
        pg, screen = self.pg, self.screen
        screen.fill((14, 23, 37))
        self.text("BRAIN → FLIGHT", 38, 26, font=self.title)
        self.text(f"S{run.subject:03d} · run {run.run_id} · RECORDED EEG REPLAY", 38, 70,
                  (133, 159, 183), self.small)
        pg.draw.rect(screen, (22, 37, 55), (38, 118, 578, 448), border_radius=12)
        for height in range(0, 6):
            y = 540 - int(height / 5 * 390)
            pg.draw.line(screen, (43, 62, 81), (88, y), (590, y), 1)
            self.text(f"{height}m", 48, y - 9, (133, 159, 183), self.small)
        y = 540 - int(drone.altitude / 5 * 390)
        x = 335
        mint = (85, 225, 184)
        pg.draw.line(screen, mint, (x - 70, y), (x + 70, y), 5)
        pg.draw.rect(screen, mint, (x - 28, y - 13, 56, 28), border_radius=8)
        for dx in (-70, 70):
            pg.draw.ellipse(screen, mint, (x + dx - 30, y - 15, 60, 13), 3)
            pg.draw.line(screen, mint, (x + dx, y - 3), (x + dx, y + 10), 3)
        self.text(f"Altitude  {drone.altitude:.2f} m", 205, 578)

        self.text("DECODER", 656, 124, mint)
        self.text(f"EEG time: {elapsed:6.2f} s", 656, 165)
        gate = latest is not None and latest.enabled and run.task_at(round(elapsed * run.sfreq)) is not None
        cue = LABELS[latest.truth].upper() if gate else "REST / TRANSITION"
        self.text(f"Cue: {cue}", 656, 208, font=self.small)
        self.text("Cue schedule enables movement.", 656, 236, (133, 159, 183), self.small)
        p = latest.smoothed_p_hands if latest else 0.5
        self.text(f"Smoothed P(hands): {p:.2f}", 656, 289)
        pg.draw.rect(screen, (45, 65, 85), (656, 328, 340, 23), border_radius=4)
        pg.draw.rect(screen, mint, (656, 328, round(340 * p), 23), border_radius=4)
        for threshold in (self.settings.low_threshold, self.settings.high_threshold):
            tx = 656 + round(340 * threshold)
            pg.draw.line(screen, (245, 200, 113), (tx, 320), (tx, 359), 2)
        self.text("FEET / DOWN", 656, 370, font=self.small)
        self.text("HANDS / UP", 887, 370, font=self.small)
        action = "UP" if drone.velocity > 0 else "DOWN" if drone.velocity < 0 else "HOVER"
        self.text(f"Command: {action}  ({drone.velocity:+.2f} m/s)", 656, 432)
        self.text("PAUSED" if paused else "Space: pause    Esc: quit", 656, 487,
                  (245, 200, 113), self.small)
        self.text(f"{self.settings.window_seconds:g} s window · prediction every "
                  f"{self.settings.prediction_seconds * 1000:g} ms", 38, 625, (133, 159, 183), self.small)
        self.text("Binary decoder ≠ rest detector", 656, 578, (133, 159, 183), self.small)
        pg.display.flip()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.pg.image.save(self.screen, str(path))

    def close(self):
        self.pg.quit()


def fly(run, bundle, settings, implementation, *, headless=False, speed=1.0,
        seconds=None, output_dir=Path("artifacts"), screenshot=None):
    if speed < 0 or (not headless and speed == 0):
        raise ValueError("Use positive --speed, or --headless --speed 0 for fast replay.")
    if seconds is not None and seconds <= 0:
        raise ValueError("--seconds must be positive.")
    engine = ReplayEngine(run, bundle["model"], settings, implementation)
    drone = Drone()
    stop = len(run.samples) if seconds is None else min(len(run.samples), round(seconds * run.sfreq))
    records, cursor = [], 0
    renderer = Renderer(settings, hidden=headless) if (not headless or screenshot) else None

    def advance(target):
        nonlocal cursor
        while cursor < target:
            next_cue_end = min((cue.stop for cue in run.cues if cue.stop > cursor), default=stop)
            # Stop precisely at prediction deadlines and cue offsets. This keeps
            # motion invariant to display frame rate and closes the gate at rest.
            end = min(cursor + settings.chunk_samples, target, engine.next_prediction, next_cue_end)
            # Integrate the PREVIOUS command while this next chunk arrives.
            drone.update((end - cursor) / run.sfreq)
            predictions = engine.push(run.samples[cursor:end])
            for prediction in predictions:
                drone.set_vertical_velocity(prediction.velocity_mps)
                records.append({**prediction.to_dict(), "altitude_m": drone.altitude})
            cursor = end
            if run.task_at(cursor) is None:
                drone.set_vertical_velocity(engine.controller.update(0.5, enabled=False))

    try:
        if headless:
            started = time.monotonic()
            while cursor < stop:
                advance(min(cursor + settings.chunk_samples, stop))
                if speed > 0:
                    deadline = started + cursor / run.sfreq / speed
                    time.sleep(max(0.0, deadline - time.monotonic()))
        else:
            pg = renderer.pg
            paused, running, playhead = False, True, 0.0
            while running and cursor < stop:
                dt = renderer.clock.tick(60) / 1000.0
                for event in pg.event.get():
                    if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                        running = False
                    elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                        paused = not paused
                if not paused:
                    playhead = min(stop / run.sfreq, playhead + dt * speed)
                    advance(min(stop, int(playhead * run.sfreq)))
                renderer.draw(run, drone, engine.latest, cursor / run.sfreq, paused)
        if renderer:
            renderer.draw(run, drone, engine.latest, cursor / run.sfreq)
            if screenshot:
                renderer.save(screenshot)
    finally:
        if renderer:
            renderer.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    if records:
        with (output_dir / "replay.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    active = [row for row in records if row["enabled"]]
    summary = {
        "subject": run.subject, "run": run.run_id,
        "replayed_seconds": cursor / run.sfreq, "predictions": len(records),
        "enabled_predictions": len(active),
        "active_raw_accuracy": float(np.mean([
            int(row["p_hands"] >= 0.5) == row["truth"] for row in active
        ])) if active else None,
        "up_commands": sum(row["velocity_mps"] > 0 for row in records),
        "down_commands": sum(row["velocity_mps"] < 0 for row in records),
        "hover_commands": sum(row["velocity_mps"] == 0 for row in records),
        "final_altitude_m": drone.altitude,
        "cue_gated": True,
    }
    (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
