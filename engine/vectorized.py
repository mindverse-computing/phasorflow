# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/engine/vectorized.py

"""
VectorizedEngine: batched, fully-differentiable PhasorFlow forward passes.

The per-sample :class:`AnalyticEngine` rebuilds a :class:`PhasorCircuit` and
loops in Python for every input, which (a) is slow and (b) historically broke
the autograd graph at block boundaries because inter-block phases were read out
with ``.item()``.  This module provides batched tensor primitives that operate
on a complex state of shape ``(batch, N)`` with full autograd support, plus
composed forward passes for the VPC and Phasor-Transformer model families.

Design principles
-----------------
* The complex state ``z ∈ C^{B×N}`` is threaded through every operation as a
  differentiable tensor.  Trainable phase parameters enter only through
  ``z * exp(i·θ)``; nothing is ever detached.
* Unitary primitives (shift / mix / DFT) match the matrices in
  :mod:`phasorflow.gates.standard` exactly, so single-block results are
  numerically identical to the reference engine.
* Between stacked blocks the *complex amplitude* is carried forward.  This is
  what makes ``inter_stack='none'`` (amplitude allowed to drift into C^N) and
  ``inter_stack='pullback'`` (renormalised back onto the torus T^N) genuinely
  different operations — the distinction the deep-stack experiments rely on.
"""

import math
import torch


# ----------------------------------------------------------------------
# Batched unitary primitives.  State shape is (batch, N), dtype complex64.
# ----------------------------------------------------------------------

def encode_phase(phases: torch.Tensor) -> torch.Tensor:
    """Lift real phase angles (B, N) onto the torus: z = e^{iφ}."""
    phases = phases.to(torch.float32)
    return torch.exp(1j * phases.to(torch.complex64))


def shift_all(state: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Per-thread phase rotation z_k → e^{iθ_k}·z_k (θ broadcast over batch)."""
    return state * torch.exp(1j * thetas.to(torch.complex64))


def mix_adjacent(state: torch.Tensor) -> torch.Tensor:
    """Apply the 50/50 beam-splitter to every even/odd adjacent pair.

    Matches :class:`phasorflow.gates.standard.MixGate`:
        [a, b] → (1/√2) [a + i·b,  i·a + b].
    Threads are processed in non-overlapping pairs (0,1), (2,3), … so the
    operation is a single block-diagonal unitary (order-independent, unlike the
    sequential per-pair loop in the reference engine — see note in
    ``mixgate_equivalence`` test).  A trailing unpaired thread is passed through.
    """
    B, N = state.shape
    inv = 1.0 / math.sqrt(2.0)
    out = state.clone()
    n_pairs = N // 2
    if n_pairs:
        a = state[:, 0:2 * n_pairs:2]
        b = state[:, 1:2 * n_pairs:2]
        new_a = inv * (a + 1j * b)
        new_b = inv * (1j * a + b)
        out[:, 0:2 * n_pairs:2] = new_a
        out[:, 1:2 * n_pairs:2] = new_b
    return out


def _dft_matrix(n: int, device, dtype=torch.complex64) -> torch.Tensor:
    """Normalised DFT matrix, identical to gates.standard.DFTGate."""
    idx = torch.arange(n, dtype=torch.float32, device=device)
    i, j = torch.meshgrid(idx, idx, indexing='ij')
    omega = torch.exp(torch.tensor(-2j * math.pi / n, dtype=dtype, device=device))
    return (1.0 / math.sqrt(n)) * (omega ** (i * j).to(dtype))


def dft_all(state: torch.Tensor, dft_mat: torch.Tensor = None) -> torch.Tensor:
    """Global DFT token mixing. State (B,N) → (B,N).  DFT is symmetric so
    ``state @ DFT`` equals ``(DFT @ state_sampleᵀ)ᵀ``."""
    if dft_mat is None:
        dft_mat = _dft_matrix(state.shape[1], state.device)
    return state @ dft_mat


def pullback(state: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project every thread back onto the unit circle: z → z/|z|."""
    return state / (state.abs().to(state.dtype) + eps)


def threshold_gate(state: torch.Tensor, tau: float) -> torch.Tensor:
    """Zero threads whose amplitude falls below ``tau`` (straight-through)."""
    mask = (state.abs() >= tau).to(state.dtype)
    return state * mask
