# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/gates/nonlinear.py

"""
Non-Linear (Pull-Back) Gates for PhasorFlow.

These gates enforce constraints by actively modifying the
amplitude/phase distribution. Unlike unitary gates, they are
NOT energy-conserving — they clip, threshold, normalize, or
compress the manifold state.

Gate Inventory:
    ThresholdGate       Inhibitory gating — zeroes low-amplitude threads
    SaturateGate        Phase quantization (error correction)
    NormalizeGate       Unit-circle projection z/|z| (PullBack)
    LogCompressGate     Logarithmic amplitude compression
    CrossCorrelateGate  Sliding-window phasor coherence
    ConvolveGate        Self-convolution for combinatorial counting
"""

import torch
from .base_gate import BaseGate


class ThresholdGate(BaseGate):
    """Inhibitory gating: zeroes threads with amplitude < threshold."""

    def __init__(self):
        super().__init__("Threshold", 1)

    def apply(self, phasor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Zeroes phasors whose amplitude falls below the threshold,
        then re-normalizes survivors to unit amplitude.

        Args:
            phasor: Complex-valued state tensor.
            threshold: Minimum amplitude to survive.

        Returns:
            Pruned and normalized state tensor.
        """
        magnitude = torch.abs(phasor)
        mask = magnitude >= threshold
        out = torch.zeros_like(phasor)
        out[mask] = phasor[mask] / magnitude[mask]
        return out

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("ThresholdGate is non-linear.")


class SaturateGate(BaseGate):
    """Phase quantization: snaps phases to N discrete levels (error correction)."""

    def __init__(self):
        super().__init__("Saturate", 1)

    def apply(self, phasor: torch.Tensor, levels: int = 2) -> torch.Tensor:
        """
        Snaps phases to the nearest multiple of 2π/levels.

        Args:
            phasor: Complex-valued state tensor.
            levels: Number of discrete phase levels.

        Returns:
            Phase-quantized unit-amplitude tensor.
        """
        angle = torch.angle(phasor)
        interval = (2.0 * torch.pi) / levels
        snapped_angle = torch.round(angle / interval) * interval
        return torch.exp(torch.complex(
            torch.zeros_like(snapped_angle), snapped_angle
        ))

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("SaturateGate is non-linear.")


class NormalizeGate(BaseGate):
    """
    Pull-Back Operator: z/|z|.

    Re-projects all threads onto the unit circle, preserving phase
    while forcing amplitude = 1. The canonical constraint used
    throughout phasor computing to enforce the S¹ manifold.
    """

    def __init__(self):
        super().__init__("Normalize", -1)

    def apply(self, phasor: torch.Tensor) -> torch.Tensor:
        """
        Normalize all phasors to unit amplitude.

        Args:
            phasor: Complex-valued state tensor.

        Returns:
            Unit-amplitude state tensor (phases preserved).
        """
        magnitudes = torch.abs(phasor).clamp(min=1e-8)
        return phasor / magnitudes

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("NormalizeGate is non-linear.")


# Canonical alias
PullBackGate = NormalizeGate


class LogCompressGate(BaseGate):
    """
    Logarithmic amplitude compression.

    Maps amplitude through log(1 + β|z|) while preserving phase.
    Prevents combinatorial explosion in path-counting problems.
    """

    def __init__(self):
        super().__init__("LogCompress", -1)

    def apply(self, phasor: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Compress amplitudes logarithmically.

        Args:
            phasor: Complex-valued state tensor.
            beta: Compression scaling factor.

        Returns:
            Amplitude-compressed state tensor.
        """
        mag = torch.abs(phasor)
        angle = torch.angle(phasor)
        compressed_mag = torch.log1p(beta * mag)
        return compressed_mag.to(torch.complex64) * torch.exp(
            1j * angle.to(torch.complex64)
        )

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("LogCompressGate is non-linear.")


class CrossCorrelateGate(BaseGate):
    """
    Sliding-window phasor cross-correlation.

    Slides a pattern phasor across a signal and measures coherence
    |Σ z_signal · z_pattern*| / M at each offset. Peak coherence
    indicates a match.
    """

    def __init__(self):
        super().__init__("CrossCorrelate", -1)

    def apply(self, signal: torch.Tensor,
              pattern: torch.Tensor) -> torch.Tensor:
        """
        Compute sliding coherence between signal and pattern.

        Args:
            signal: Complex-valued signal vector [N].
            pattern: Complex-valued pattern vector [M].

        Returns:
            Coherence values at each valid offset [N-M+1].
        """
        n, m = signal.shape[0], pattern.shape[0]
        if m > n:
            return torch.tensor([], dtype=torch.float32)

        conj_pattern = torch.conj(pattern)
        coherences = []
        for i in range(n - m + 1):
            window = signal[i:i + m]
            coh = torch.abs(torch.sum(window * conj_pattern)) / m
            coherences.append(coh)
        return torch.stack(coherences)

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("CrossCorrelateGate is non-linear.")


class ConvolveGate(BaseGate):
    """
    Self-convolution for combinatorial counting.

    Computes C[n] = Σ state[j] · state[n-1-j] — the next wavefront
    amplitude from all previous split-point products. Used for
    Catalan numbers and matrix chain counting.
    """

    def __init__(self):
        super().__init__("Convolve", -1)

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """
        Self-convolve the state amplitude vector.

        Args:
            state: Complex-valued state vector [N].

        Returns:
            Convolved state vector [N].
        """
        n = state.shape[0]
        result = state.clone()
        for i in range(2, n):
            total = torch.tensor(0.0, dtype=torch.complex64)
            for j in range(i):
                total = total + state[j] * state[i - 1 - j]
            result[i] = total
        return result

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("ConvolveGate is non-linear.")
