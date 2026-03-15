# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/gates/encoding.py

"""
Encoding Gates for PhasorFlow.

Convert classical data into phasor representations on the
unit-circle manifold. These are the input layer of any
phasor-based computation pipeline.

Gate Inventory:
    EncodePhaseGate      Classical values → phase angles
    EncodeAmplitudeGate  Classical values → amplitude-modulated phasors
"""

import torch
import math
from .base_gate import BaseGate


class EncodePhaseGate(BaseGate):
    """
    Encodes classical values as phases on the unit circle.

    Maps each value x to angle θ = (x / max_val) · 2π,
    producing a unit-amplitude phasor e^{jθ}.
    """

    def __init__(self):
        super().__init__("EncodePhase", -1)

    def apply(self, values: torch.Tensor,
              max_val: float = 1.0) -> torch.Tensor:
        """
        Encode values as phases.

        Args:
            values: Real-valued input tensor.
            max_val: Maximum value for normalization.

        Returns:
            Complex-valued unit-amplitude phasor tensor.
        """
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        phases = (values.float() / max_val) * (2.0 * math.pi)
        return torch.exp(1j * phases.to(torch.complex64))

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("EncodePhaseGate is data-dependent.")


class EncodeAmplitudeGate(BaseGate):
    """
    Encodes classical values as amplitudes on the unit circle.

    Produces phasors at phase 0 with amplitude proportional to the
    input value: z = value · e^{j·0} = value + 0j.
    """

    def __init__(self):
        super().__init__("EncodeAmplitude", -1)

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode values as amplitudes (phase = 0).

        Args:
            values: Real-valued input tensor.

        Returns:
            Complex-valued phasor tensor with encoded amplitudes.
        """
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        return values.to(torch.complex64)

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("EncodeAmplitudeGate is data-dependent.")
