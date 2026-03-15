# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/gates/neuromorphic.py

"""
Neuromorphic Gates for PhasorFlow.

Dynamic coupling operations that implement biologically-inspired
interaction layers. These gates handle non-linear coupling required
for optimization, community detection, and associative memory.

Gate Inventory:
    SynapticGate           Weighted neural connection (rotation coupling)
    KuramotoGate           Phase-lock synchronization dynamics
    HebbianGate            Associative relaxation toward attractors
    IsingGate              Binary 0/π phase coupling
    AsymmetricCoupleGate   Directional DAG coupling for ordering
"""

import torch
from .base_gate import BaseGate


class SynapticGate(BaseGate):
    """
    A weighted neural connection between two threads.

    Unlike MixGate (50/50), SynapticGate has a learnable strength
    parameter that controls the coupling rotation.
    """

    def __init__(self, strength: float = 1.0):
        super().__init__("Synapse", 2)
        self.strength = strength

    def get_matrix(self, dt: float = 0.01) -> torch.Tensor:
        """Returns a 2×2 infinitesimal rotation matrix."""
        s = torch.tensor(self.strength * dt, dtype=torch.float32)
        return torch.tensor([
            [torch.cos(s), -torch.sin(s)],
            [torch.sin(s),  torch.cos(s)]
        ], dtype=torch.complex64)

    def apply(self, phasor_a: torch.Tensor, phasor_b: torch.Tensor,
              dt: float = 0.01):
        """Apply synaptic coupling and return (new_a, new_b)."""
        mat = self.get_matrix(dt=dt)
        pair = torch.stack([phasor_a, phasor_b])
        result = torch.matmul(mat, pair)
        return result[0], result[1]


class KuramotoGate(BaseGate):
    """
    Kuramoto phase-lock synchronization gate.

    Each oscillator adjusts its phase toward its weighted neighbors:
        dθ_i/dt = K · Σ_j w_ij · sin(θ_j - θ_i)

    This is the core dynamics of phase-frustration and spectral
    entrainment problems.
    """

    def __init__(self):
        super().__init__("Kuramoto", -1)

    def apply(self, state: torch.Tensor, weights: torch.Tensor,
              dt: float = 0.01, coupling_k: float = 1.0) -> torch.Tensor:
        """
        Single Kuramoto update step.

        Args:
            state: Complex-valued state vector [N].
            weights: Coupling weight matrix [N × N].
            dt: Integration time step.
            coupling_k: Global coupling strength.

        Returns:
            Updated state vector on S¹.
        """
        phases = torch.angle(state)
        phase_diff = phases.unsqueeze(1) - phases.unsqueeze(0)
        w_real = weights.real.float() if weights.is_complex() else weights.float()
        interaction = coupling_k * torch.matmul(
            w_real, torch.sin(phase_diff)
        )
        d_theta = torch.diagonal(interaction) * dt
        new_phases = phases + d_theta
        return torch.exp(1j * new_phases.to(torch.complex64))

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("KuramotoGate is state-dependent.")


class HebbianGate(BaseGate):
    """
    Hebbian associative relaxation gate.

    Pulls the current state toward the weighted average of neighbors,
    then re-normalizes to the unit circle. Implements attractor dynamics
    for associative memory recall.
    """

    def __init__(self):
        super().__init__("Hebbian", -1)

    def apply(self, state: torch.Tensor, weight_matrix: torch.Tensor,
              alpha: float = 0.1) -> torch.Tensor:
        """
        Single Hebbian relaxation step.

        Args:
            state: Complex-valued state vector [N].
            weight_matrix: Hebbian weight matrix [N × N].
            alpha: Relaxation rate.

        Returns:
            Updated state on the unit circle.
        """
        field = torch.matmul(weight_matrix, state)
        new_state = state + (alpha * field)
        return new_state / torch.abs(new_state).clamp(min=1e-8)

    @staticmethod
    def store_patterns(num_threads: int, phase_patterns: list) -> torch.Tensor:
        """
        Constructs a Hebbian weight matrix from stored phase patterns.
        Uses outer-product rule: W = (1/P) · Σ z · z^H

        Args:
            num_threads: Number of oscillator threads.
            phase_patterns: List of phase-angle arrays.

        Returns:
            Complex weight matrix [N × N].
        """
        weights = torch.zeros(
            (num_threads, num_threads), dtype=torch.complex64
        )
        for pattern in phase_patterns:
            phasors = torch.exp(
                1j * torch.tensor(pattern, dtype=torch.float32)
            )
            weights += torch.outer(phasors, torch.conj(phasors))
        weights /= len(phase_patterns)
        weights.fill_diagonal_(0.0 + 0j)
        return weights

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("HebbianGate is state-dependent.")


class IsingGate(BaseGate):
    """
    Binary Ising coupling gate.

    Drives phases toward 0 or π using anti-ferromagnetic coupling:
        sin(2(θ_j - θ_i)) creates two basins of attraction.

    Used for binary partition problems (Max-Cut, graph coloring).
    """

    def __init__(self):
        super().__init__("Ising", -1)

    def apply(self, state: torch.Tensor, adjacency: torch.Tensor,
              dt: float = 0.05, coupling_k: float = 1.0) -> torch.Tensor:
        """
        Single Ising coupling step.

        Args:
            state: Complex-valued state vector [N].
            adjacency: Adjacency/coupling matrix [N × N].
            dt: Integration time step.
            coupling_k: Coupling strength.

        Returns:
            Updated state with phases pulled toward 0/π.
        """
        phases = torch.angle(state)
        phase_diff = phases.unsqueeze(1) - phases.unsqueeze(0)
        a_real = adjacency.real.float() if adjacency.is_complex() else adjacency.float()
        interaction = coupling_k * torch.matmul(
            a_real, torch.sin(2 * phase_diff)
        )
        d_theta = torch.diagonal(interaction) * dt
        new_phases = phases + d_theta
        return torch.exp(1j * new_phases.to(torch.complex64))

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("IsingGate is state-dependent.")


class AsymmetricCoupleGate(BaseGate):
    """
    Directional (asymmetric) coupling gate for DAG-based problems.

    Unlike symmetric Kuramoto, only forward edges propagate phase
    pressure: child phases accumulate from parent phases. Used for
    topological sort, critical path, and dependency ordering.
    """

    def __init__(self):
        super().__init__("AsymmetricCouple", -1)

    def apply(self, state: torch.Tensor, dag_matrix: torch.Tensor,
              dt: float = 0.01) -> torch.Tensor:
        """
        Single asymmetric coupling step.

        Args:
            state: Complex-valued state vector [N].
            dag_matrix: Directed adjacency [N × N] where [i,j]=1 means i→j.
            dt: Integration time step.

        Returns:
            Updated state with directional phase pressure.
        """
        phases = torch.angle(state)
        amps = torch.abs(state)
        n = state.shape[0]

        new_phases = phases.clone()
        for j in range(n):
            parents = dag_matrix[:, j].real.float() if dag_matrix.is_complex() else dag_matrix[:, j].float()
            if parents.sum() > 0:
                parent_phases = phases * parents
                pressure = parent_phases.sum() / parents.sum().clamp(min=1e-8)
                new_phases[j] = max(new_phases[j].item(),
                                    pressure.item()) + dt

        return amps.to(torch.complex64) * torch.exp(
            1j * new_phases.to(torch.complex64)
        )

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("AsymmetricCoupleGate is state-dependent.")
