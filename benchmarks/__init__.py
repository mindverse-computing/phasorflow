"""Honest benchmark tasks, baselines, and studies for PhasorFlow.

Modules:
    tasks          -- synthetic task families + sklearn / self-attention baselines
    eeg_benchmark  -- real PhysioNet motor-imagery EEG benchmark (VPC vs BCI baselines)
    depth_study    -- VPC capacity ceiling + Phasor Transformer depth scaling
"""

from .tasks import (
    sum_cosine_task,
    phase_parity_task,
    multifreq_sequence_task,
    classification_baselines,
    regression_baselines,
    SelfAttnRegressor,
)

__all__ = [
    "sum_cosine_task",
    "phase_parity_task",
    "multifreq_sequence_task",
    "classification_baselines",
    "regression_baselines",
    "SelfAttnRegressor",
]
