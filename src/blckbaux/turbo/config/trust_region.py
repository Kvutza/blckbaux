from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Rescalarize(Enum):
    ON_RESTART = "on_restart"
    ON_PROPOSE = "on_propose"


@dataclass(frozen=True)
class NoTrustRegion:
    noise_aware: bool = False


@dataclass(frozen=True)
class TrustRegionBounds:
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6

    def __post_init__(self) -> None:
        if self.length_init <= 0:
            raise ValueError(f"length_init must be > 0, got {self.length_init}")
        if self.length_min <= 0:
            raise ValueError(f"length_min must be > 0, got {self.length_min}")
        if self.length_max <= 0:
            raise ValueError(f"length_max must be > 0, got {self.length_max}")
        if self.length_min >= self.length_max:
            raise ValueError(
                f"length_min must be < length_max, got {self.length_min} >= {self.length_max}"
            )
        if self.length_init > self.length_max:
            raise ValueError(
                f"length_init must be <= length_max, got {self.length_init} > {self.length_max}"
            )
        if self.length_min > self.length_init:
            raise ValueError(
                f"length_min must be <= length_init, got {self.length_min} > {self.length_init}"
            )


@dataclass(frozen=True)
class TurboTrustRegion:
    length: TrustRegionBounds = TrustRegionBounds()
    noise_aware: bool = False

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


@dataclass(frozen=True)
class MultiObjectiveConfig:
    num_metrics: int
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.num_metrics < 2:
            raise ValueError(
                f"num_metrics must be >= 2 for MORBO, got {self.num_metrics}"
            )
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")


@dataclass(frozen=True)
class RescalePolicy:
    rescalarize: Rescalarize = Rescalarize.ON_PROPOSE


@dataclass(frozen=True)
class MorboTrustRegion:
    multi_objective: MultiObjectiveConfig
    length: TrustRegionBounds = TrustRegionBounds()
    rescale_policy: RescalePolicy = RescalePolicy()
    noise_aware: bool = False

    @property
    def rescalarize(self) -> Rescalarize:
        return self.rescale_policy.rescalarize

    @property
    def num_metrics(self) -> int:
        return self.multi_objective.num_metrics

    @property
    def alpha(self) -> float:
        return self.multi_objective.alpha

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


@dataclass(frozen=True)
class MultiTrustRegionPolicy:
    num_regions: int = 4
    length: TrustRegionBounds = TrustRegionBounds()
    succ_tolerance: int = 3
    fail_tolerance: int = 5
    sharing_policy: Literal["shared", "nearest_center", "independent"] = "shared"

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


TrustRegionConfig = (
    NoTrustRegion | TurboTrustRegion | MorboTrustRegion | MultiTrustRegionPolicy
)
