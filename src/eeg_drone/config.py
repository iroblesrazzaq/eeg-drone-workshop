from dataclasses import asdict, dataclass

HANDS = 1
FEET = 0
LABELS = {FEET: "feet", HANDS: "hands"}
TRAIN_RUNS = (6, 10)
TEST_RUN = 14


@dataclass(frozen=True)
class Settings:
    channels: tuple[str, ...] = ("C3", "Cz", "C4")
    bands: tuple[tuple[str, float, float], ...] = (
        ("mu", 8.0, 12.0), ("beta", 13.0, 30.0),
    )
    low_hz: float = 7.0
    high_hz: float = 30.0
    window_seconds: float = 2.0
    prediction_seconds: float = 0.25
    training_step_seconds: float = 0.5
    onset_delay_seconds: float = 0.5
    chunk_samples: int = 8  # 50 ms at this dataset's 160 Hz
    smoothing_alpha: float = 0.3
    low_threshold: float = 0.4
    high_threshold: float = 0.6
    speed_mps: float = 0.65

    def to_dict(self):
        return asdict(self)


def feature_names(settings: Settings) -> list[str]:
    """Channel first: C3_mu, C3_beta, Cz_mu, ... ."""
    return [f"{channel}_{band}" for channel in settings.channels
            for band, _, _ in settings.bands]
