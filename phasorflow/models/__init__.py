# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/models/__init__.py

"""
PhasorFlow Machine Learning Models.

Provides high-level, scikit-learn-style ML models built on top of
PhasorCircuit and the AnalyticEngine. These models abstract away
circuit construction while preserving full introspection.

Available Models:
    VPC               — Variational Phasor Circuit classifier
    PhasorTransformer — FNet-style Phasor Transformer for sequence prediction
    PhasorGAN         — Generative Adversarial Network for timeseries
"""

from .vpc import VPC
from .transformer import PhasorTransformer
from .gan import PhasorGAN

__all__ = ["VPC", "PhasorTransformer", "PhasorGAN"]

