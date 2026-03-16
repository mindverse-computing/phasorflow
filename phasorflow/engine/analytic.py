# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/engine/analytic.py

"""
AnalyticEngine: Pure-PyTorch simulator for PhasorFlow circuits.

Dispatches all 22 gate types from PhasorCircuit, executing them
sequentially on a complex-valued state vector. Supports autograd
for variational parameter optimization.
"""

import torch
import math
from ..gates.standard import (
    ShiftGate, MixGate, DFTGate,
    PermuteGate, ReverseGate,
    AccumulateGate, GridPropagateGate
)
from ..gates.nonlinear import (
    ThresholdGate, SaturateGate,
    NormalizeGate, LogCompressGate,
    CrossCorrelateGate, ConvolveGate
)
from ..gates.neuromorphic import (
    KuramotoGate, HebbianGate,
    IsingGate, AsymmetricCoupleGate
)
from ..gates.encoding import (
    EncodePhaseGate, EncodeAmplitudeGate
)


class AnalyticEngine:
    """Pure-mathematical PyTorch simulator for PhasorFlow circuits."""

    def __init__(self):
        # Pre-instantiate reusable gate objects
        self._shift = ShiftGate()
        self._mix = MixGate()
        self._threshold = ThresholdGate()
        self._saturate = SaturateGate()
        self._normalize = NormalizeGate()
        self._log_compress = LogCompressGate()
        self._cross_correlate = CrossCorrelateGate()
        self._convolve = ConvolveGate()
        self._accumulate = AccumulateGate()
        self._kuramoto = KuramotoGate()
        self._hebbian = HebbianGate()
        self._ising = IsingGate()
        self._asymmetric = AsymmetricCoupleGate()
        self._encode_phase = EncodePhaseGate()
        self._encode_amp = EncodeAmplitudeGate()

    def run(self, circuit, initial_state=None):
        """
        Execute a PhasorCircuit.

        Args:
            circuit: PhasorCircuit instance.
            initial_state: Optional initial state vector. If None,
                          all threads start at phase 0 (z = 1+0j).

        Returns:
            dict with 'state_vector', 'phases', 'amplitudes',
            and 'measurements' (snapshots at measure gates).
        """
        n = circuit.num_threads

        if initial_state is not None:
            state = initial_state.clone()
        else:
            state = torch.ones(n, dtype=torch.complex64)

        measurements = {}

        for gate_name, targets, params in circuit.data:

            # ── Standard Unitary ──────────────────────────────
            if gate_name == 'shift':
                idx = targets[0]
                phi = params['phi']
                mat = self._shift.get_matrix(phi=phi)
                new_val = mat[0, 0] * state[idx]
                state = state.clone()
                state[idx] = new_val

            elif gate_name == 'mix':
                idx_a, idx_b = targets
                mat = self._mix.get_matrix()
                sub = torch.stack([state[idx_a], state[idx_b]])
                new_sub = torch.matmul(mat, sub)
                state = state.clone()
                state[idx_a] = new_sub[0]
                state[idx_b] = new_sub[1]

            elif gate_name == 'dft':
                dft_gate = DFTGate(len(targets))
                mat = dft_gate.get_matrix()
                if len(targets) == n:
                    state = torch.matmul(mat, state)
                else:
                    sub = state[targets]
                    transformed = torch.matmul(mat, sub)
                    state = state.clone()
                    for i, t in enumerate(targets):
                        state[t] = transformed[i]

            elif gate_name == 'permute':
                perm = params['permutation']
                perm_gate = PermuteGate(n)
                state = perm_gate.apply(state, perm)

            elif gate_name == 'reverse':
                state = torch.flip(state, dims=[0])

            # ── Non-Linear (Pull-Back) ────────────────────────
            elif gate_name == 'threshold':
                state = self._threshold.apply(state, threshold=params['threshold'])

            elif gate_name == 'saturate':
                state = self._saturate.apply(state, levels=params['levels'])

            elif gate_name == 'normalize':
                state = self._normalize.apply(state)

            elif gate_name == 'log_compress':
                state = self._log_compress.apply(state, beta=params.get('beta', 1.0))

            elif gate_name == 'cross_correlate':
                pattern = params['pattern']
                if not isinstance(pattern, torch.Tensor):
                    pattern = torch.tensor(pattern, dtype=torch.complex64)
                coherences = self._cross_correlate.apply(state, pattern)
                measurements[f'cross_correlate'] = coherences

            elif gate_name == 'convolve':
                state = self._convolve.apply(state)

            # ── Neuromorphic ──────────────────────────────────
            elif gate_name == 'kuramoto':
                weights = params['weights']
                if not isinstance(weights, torch.Tensor):
                    weights = torch.tensor(weights, dtype=torch.float32)
                state = self._kuramoto.apply(
                    state, weights,
                    dt=params.get('dt', 0.01),
                    coupling_k=params.get('coupling_k', 1.0)
                )

            elif gate_name == 'hebbian':
                weights = params['weights']
                if not isinstance(weights, torch.Tensor):
                    weights = torch.tensor(weights, dtype=torch.complex64)
                state = self._hebbian.apply(
                    state, weights,
                    alpha=params.get('alpha', 0.1)
                )

            elif gate_name == 'ising':
                adj = params['adjacency']
                if not isinstance(adj, torch.Tensor):
                    adj = torch.tensor(adj, dtype=torch.float32)
                state = self._ising.apply(
                    state, adj,
                    dt=params.get('dt', 0.05),
                    coupling_k=params.get('coupling_k', 1.0)
                )

            elif gate_name == 'asymmetric_couple':
                dag = params['dag_matrix']
                if not isinstance(dag, torch.Tensor):
                    dag = torch.tensor(dag, dtype=torch.float32)
                state = self._asymmetric.apply(
                    state, dag,
                    dt=params.get('dt', 0.01)
                )

            # ── Symphony-Specific ─────────────────────────────
            elif gate_name == 'accumulate':
                state = self._accumulate.apply(state, targets=targets)

            elif gate_name == 'grid_propagate':
                rows = params['rows']
                cols = params['cols']
                grid_gate = GridPropagateGate()
                state_2d = state[:rows * cols].reshape(rows, cols)
                result_2d = grid_gate.apply(state_2d)
                state = state.clone()
                state[:rows * cols] = result_2d.flatten()

            # ── Encoding ──────────────────────────────────────
            elif gate_name == 'encode_phases':
                values = params['values']
                max_val = params.get('max_val', 1.0)
                encoded = self._encode_phase.apply(
                    torch.tensor(values, dtype=torch.float32)
                    if not isinstance(values, torch.Tensor) else values,
                    max_val=max_val
                )
                state = state.clone()
                for i in range(min(len(encoded), n)):
                    state[i] = encoded[i]

            elif gate_name == 'encode_amplitudes':
                values = params['values']
                encoded = self._encode_amp.apply(
                    torch.tensor(values, dtype=torch.float32)
                    if not isinstance(values, torch.Tensor) else values
                )
                state = state.clone()
                for i in range(min(len(encoded), n)):
                    state[i] = encoded[i]

            # ── Control ───────────────────────────────────────
            elif gate_name == 'barrier':
                pass  # Synchronization only, no-op in simulation

            elif gate_name == 'measure':
                label = params.get('label', '')
                measurements[label if label else f'measure_{len(measurements)}'] = {
                    'state_vector': state.clone(),
                    'phases': torch.angle(state),
                    'amplitudes': torch.abs(state),
                }

            else:
                raise ValueError(f"Unknown gate: '{gate_name}'")

        return {
            'state_vector': state,
            'phases': torch.angle(state),
            'amplitudes': torch.abs(state),
            'measurements': measurements
        }
