"""Keep confidence smoothing, experiment gating, and drone dynamics separate."""
from dataclasses import dataclass

from .config import Settings


class Controller:
    def __init__(self, settings: Settings, probability_to_velocity):
        self.settings = settings
        self.map_velocity = probability_to_velocity
        self.smoothed = 0.5

    def update(self, p_hands: float, enabled: bool) -> float:
        if not enabled:
            # Reset between trials: a previous hands cue must not bias the next.
            self.smoothed = 0.5
            return 0.0
        a = self.settings.smoothing_alpha
        self.smoothed = a * p_hands + (1 - a) * self.smoothed
        return float(self.map_velocity(
            self.smoothed, self.settings.speed_mps,
            self.settings.low_threshold, self.settings.high_threshold,
        ))


@dataclass
class Drone:
    """A 1-D kinematic simulation, with height bounds; positive velocity is UP."""
    altitude: float = 2.5
    velocity: float = 0.0
    minimum: float = 0.5
    maximum: float = 5.0

    def set_vertical_velocity(self, velocity: float):
        self.velocity = float(velocity)

    def update(self, dt: float):
        self.altitude = max(self.minimum, min(self.maximum,
                            self.altitude + self.velocity * dt))
