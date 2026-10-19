from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AcquisitionKind(Enum):
    THOMPSON = "thompson"
    PARETO = "pareto"
    UCB = "ucb"


@dataclass(frozen=True)
class DrawAcquisitionConfig:
    pass


@dataclass(frozen=True)
class ParetoAcquisitionConfig:
    pass


@dataclass(frozen=True)
class RandomAcquisitionConfig:
    pass


@dataclass(frozen=True)
class UCBAcquisitionConfig:
    beta: float = 2.0

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")


@dataclass(frozen=True)
class RAASPOptimizerConfig:
    pass


@dataclass(frozen=True)
class NDSOptimizerConfig:
    pass


AcquisitionConfig = (
    UCBAcquisitionConfig
    | DrawAcquisitionConfig
    | ParetoAcquisitionConfig
    | RandomAcquisitionConfig
)
AcqOptimizerConfig = RAASPOptimizerConfig | NDSOptimizerConfig
