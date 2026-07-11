# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.

"""
Correctness tests for PhasorFlow after the v0.3.0 fixes.

Run with:  pytest phasorflow/tests/test_correctness.py -q
Covers:
  * Gate unitarity (Shift / Mix / DFT preserve the L2 norm).
  * Autograd finiteness of the transformer readout at the former singularities.
  * Numerical equivalence of the vectorized engine and the reference engine.
  * Inter-stack semantics: 'pullback' differs from 'none'; gradient reaches
    every stack / block (the .item() detachment regression).
"""

import math
import torch

from phasorflow.circuit import PhasorCircuit
from phasorflow.engine.analytic import AnalyticEngine
from phasorflow.engine import vectorized as V
from phasorflow.models.vpc import VPC
from phasorflow.models.transformer import PhasorTransformer


# ----------------------------------------------------------------------
# Gate unitarity
# ----------------------------------------------------------------------

def test_shift_preserves_norm():
    z = torch.randn(1, 8, dtype=torch.complex64)
    theta = torch.empty(8).uniform_(-math.pi, math.pi)
    out = V.shift_all(z, theta.unsqueeze(0))
    assert torch.allclose(out.abs(), z.abs(), atol=1e-6)


def test_mix_preserves_norm():
    z = torch.randn(4, 8, dtype=torch.complex64)
    out = V.mix_adjacent(z)
    n0 = z.abs().pow(2).sum(1)
    n1 = out.abs().pow(2).sum(1)
    assert torch.allclose(n0, n1, atol=1e-5)


def test_dft_preserves_norm():
    z = torch.randn(4, 16, dtype=torch.complex64)
    out = V.dft_all(z)
    assert torch.allclose(z.abs().pow(2).sum(1), out.abs().pow(2).sum(1), atol=1e-4)


# ----------------------------------------------------------------------
# Readout gradient finiteness (the NaN regression)
# ----------------------------------------------------------------------

def test_readout_gradient_finite_at_boundary():
    m = PhasorTransformer(seq_length=4, num_blocks=1)
    # phases exactly on the former singularities pi/2 + k*pi
    phase = torch.tensor([math.pi / 2, -math.pi / 2, 3 * math.pi / 2],
                         requires_grad=True)
    out = m._readout(phase)
    g = torch.autograd.grad(out.sum(), phase)[0]
    assert torch.isfinite(g).all()
    # and it still matches arcsin(sin) in value
    assert torch.allclose(out, torch.asin(torch.sin(phase)).detach(), atol=1e-4)


def test_transformer_training_no_nan():
    torch.manual_seed(0)
    X = torch.empty(64, 8).uniform_(-math.pi / 2, math.pi / 2)
    y = X.mean(1)
    m = PhasorTransformer(seq_length=8, num_blocks=2, readout_layer=True)
    h = m.fit(X, y, epochs=30, lr=0.05, verbose=False)
    assert not any(math.isnan(v) for v in h['loss_history'])


# ----------------------------------------------------------------------
# Vectorized == reference engine
# ----------------------------------------------------------------------

def test_vectorized_matches_reference():
    torch.manual_seed(0)
    N = 6
    x = torch.empty(N).uniform_(0, 2 * math.pi)
    w = torch.empty(N).uniform_(-math.pi, math.pi)
    pc = PhasorCircuit(N)
    for i in range(N):
        pc.shift(i, x[i].item())
    for i in range(N):
        pc.shift(i, w[i].item())
    for i in range(0, N - 1, 2):
        pc.mix(i, i + 1)
    pc.dft()
    ref = AnalyticEngine().run(pc)['state_vector']

    z = V.encode_phase(x.unsqueeze(0))
    z = V.shift_all(z, w.unsqueeze(0))
    z = V.mix_adjacent(z)
    z = V.dft_all(z)
    assert torch.allclose(z.squeeze(0), ref, atol=1e-5)


# ----------------------------------------------------------------------
# Inter-stack semantics and gradient reach
# ----------------------------------------------------------------------

def test_pullback_differs_from_none():
    torch.manual_seed(1)
    w = torch.empty(3 * 6).uniform_(-math.pi, math.pi)
    X = torch.empty(4, 6).uniform_(0, 2 * math.pi)
    mn = VPC(6, num_layers=1, num_stacks=3, inter_stack='none')
    mp = VPC(6, num_layers=1, num_stacks=3, inter_stack='pullback')
    mn.weights.data = w.clone()
    mp.weights.data = w.clone()
    assert (mn.forward_batch(X) - mp.forward_batch(X)).abs().max() > 1e-3


def test_gradient_reaches_all_vpc_stacks():
    torch.manual_seed(0)
    m = VPC(num_features=6, num_layers=1, num_stacks=3, inter_stack='pullback')
    X = torch.empty(4, 6).uniform_(0, 2 * math.pi)
    y = torch.tensor([0., 1., 0., 1.])
    loss = ((m.forward_batch(X) - y) ** 2).mean()
    loss.backward()
    pps = m.params_per_stack
    for s in range(m.num_stacks):
        assert m.weights.grad[s * pps:(s + 1) * pps].norm() > 0


def test_gradient_reaches_all_transformer_blocks():
    torch.manual_seed(0)
    m = PhasorTransformer(seq_length=10, num_blocks=3, stacking='separate')
    X = torch.empty(8, 10).uniform_(-math.pi / 2, math.pi / 2)
    y = X.mean(1)
    loss = ((m.forward_batch(X) - y) ** 2).mean()
    loss.backward()
    ppb = m._params_per_block
    for b in range(m.num_blocks):
        assert m.weights.grad[b * ppb:(b + 1) * ppb].norm() > 0
