# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/__init__.py

"""
PhasorFlow: A Unit Circle (U1) Phasor-based ML Framework.

PhasorFlow is a professional PyTorch-based machine learning library
built on phasor (unit-circle) computing. It provides:

  - 22 phasor gates (standard, nonlinear, neuromorphic, encoding)
  - Fluent circuit construction API
  - Variational Phasor Circuit (VPC) optimization
  - Phasor Transformer models
  - Neuromorphic layers (LIP, associative memory)
  - Text and matplotlib circuit visualization
"""

from .circuit import PhasorCircuit
from .engine import Simulator
from .visualization.text import TextDrawer
from .visualization.matplotlib_drawer import MatplotlibDrawer
from . import models
from .models import VPC, PhasorTransformer, PhasorGAN

# Re-export all gates for convenience
from .gates import (
    BaseGate,
    # Standard (7)
    ShiftGate, MixGate, DFTGate,
    PermuteGate, ReverseGate,
    AccumulateGate, GridPropagateGate,
    # Non-Linear (7)
    ThresholdGate, SaturateGate,
    NormalizeGate, PullBackGate,
    LogCompressGate,
    CrossCorrelateGate, ConvolveGate,
    # Neuromorphic (5)
    SynapticGate,
    KuramotoGate, HebbianGate,
    IsingGate, AsymmetricCoupleGate,
    # Encoding (2)
    EncodePhaseGate, EncodeAmplitudeGate,
)


def draw(circuit, mode='text'):
    if mode == 'text':
        print(TextDrawer.draw(circuit))
    elif mode == 'mpl':
        drawer = MatplotlibDrawer(circuit)
        return drawer.draw()


__version__ = "0.2.0"
__all__ = [
    "PhasorCircuit", "Simulator", "draw",
    # Models
    "models", "VPC", "PhasorTransformer", "PhasorGAN",
    # Gates
    "BaseGate",
    "ShiftGate", "MixGate", "DFTGate",
    "PermuteGate", "ReverseGate",
    "AccumulateGate", "GridPropagateGate",
    "ThresholdGate", "SaturateGate",
    "NormalizeGate", "PullBackGate",
    "LogCompressGate",
    "CrossCorrelateGate", "ConvolveGate",
    "SynapticGate",
    "KuramotoGate", "HebbianGate",
    "IsingGate", "AsymmetricCoupleGate",
    "EncodePhaseGate", "EncodeAmplitudeGate",
]
