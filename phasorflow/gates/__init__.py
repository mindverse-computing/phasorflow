# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/gates/__init__.py

"""
PhasorFlow Gate Library — 22 Gates.

Standard Unitary:   ShiftGate, MixGate, DFTGate, PermuteGate,
                    ReverseGate, AccumulateGate, GridPropagateGate
Non-Linear:         ThresholdGate, SaturateGate, NormalizeGate (PullBackGate),
                    LogCompressGate, CrossCorrelateGate, ConvolveGate
Neuromorphic:       SynapticGate, KuramotoGate, HebbianGate,
                    IsingGate, AsymmetricCoupleGate
Encoding:           EncodePhaseGate, EncodeAmplitudeGate
"""

from .base_gate import BaseGate

# Standard Unitary
from .standard import (
    ShiftGate, MixGate, DFTGate,
    PermuteGate, ReverseGate,
    AccumulateGate, GridPropagateGate
)

# Non-Linear (Pull-Back)
from .nonlinear import (
    ThresholdGate, SaturateGate,
    NormalizeGate, PullBackGate,
    LogCompressGate,
    CrossCorrelateGate, ConvolveGate
)

# Neuromorphic
from .neuromorphic import (
    SynapticGate,
    KuramotoGate, HebbianGate,
    IsingGate, AsymmetricCoupleGate
)

# Encoding
from .encoding import (
    EncodePhaseGate, EncodeAmplitudeGate
)

__all__ = [
    # Base
    "BaseGate",
    # Standard Unitary (7)
    "ShiftGate", "MixGate", "DFTGate",
    "PermuteGate", "ReverseGate",
    "AccumulateGate", "GridPropagateGate",
    # Non-Linear (7)
    "ThresholdGate", "SaturateGate",
    "NormalizeGate", "PullBackGate",
    "LogCompressGate",
    "CrossCorrelateGate", "ConvolveGate",
    # Neuromorphic (5)
    "SynapticGate",
    "KuramotoGate", "HebbianGate",
    "IsingGate", "AsymmetricCoupleGate",
    # Encoding (2)
    "EncodePhaseGate", "EncodeAmplitudeGate",
]
