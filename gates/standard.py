# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/gates/standard.py

"""
Standard Unitary Gates for PhasorFlow.

Energy-conserving operators on the unit-circle manifold.
All gates are PyTorch-native and support autograd differentiation
for use in variational phasor circuits (VPC training).

Gate Inventory:
    ShiftGate          Phase rotation z → e^{jφ}·z
    MixGate            50/50 beam-splitter (interference junction)
    DFTGate            Discrete Fourier Transform
    PermuteGate        Thread reordering via permutation matrix
    ReverseGate        Time-reversal (anti-diagonal permutation)
    AccumulateGate     Coherent amplitude summation on a bus
    GridPropagateGate  2D wavefront DP-grid propagation
"""

import torch
import math
from .base_gate import BaseGate


class ShiftGate(BaseGate):
    """Phase rotation gate: z_i → e^{jφ} · z_i."""

    def __init__(self):
        super().__init__("Shift", 1)

    def get_matrix(self, phi: float = 0.0) -> torch.Tensor:
        """Returns a 1×1 unitary rotation matrix."""
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.float32)
        val = torch.exp(1j * phi.to(torch.complex64))
        return val.view(1, 1)

    def apply(self, phasor: torch.Tensor, phi: float = 0.0) -> torch.Tensor:
        """Apply phase shift to a phasor tensor."""
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.float32)
        return phasor * torch.exp(1j * phi.to(torch.complex64))


class MixGate(BaseGate):
    """50/50 beam-splitter (interference junction) between two threads."""

    def __init__(self):
        super().__init__("Mix", 2)

    def get_matrix(self) -> torch.Tensor:
        """Returns the 2×2 unitary beam-splitter matrix."""
        val = 1 / torch.sqrt(torch.tensor(2.0))
        return val * torch.tensor([
            [1.0 + 0j, 0.0 + 1j],
            [0.0 + 1j, 1.0 + 0j]
        ], dtype=torch.complex64)

    def apply(self, phasor_a: torch.Tensor, phasor_b: torch.Tensor):
        """Apply beam-splitter and return (new_a, new_b)."""
        mat = self.get_matrix()
        pair = torch.stack([phasor_a, phasor_b])
        result = torch.matmul(mat, pair)
        return result[0], result[1]


class DFTGate(BaseGate):
    """Discrete Fourier Transform across N threads."""

    def __init__(self, size: int):
        super().__init__("DFT", size)

    def get_matrix(self) -> torch.Tensor:
        """Returns the N×N DFT matrix (normalized)."""
        n = self.num_threads
        i_vec = torch.arange(n, dtype=torch.float32).to(torch.complex64)
        j_vec = torch.arange(n, dtype=torch.float32).to(torch.complex64)
        i, j = torch.meshgrid(i_vec, j_vec, indexing='ij')
        omega = torch.exp(torch.tensor(-2j * math.pi / n, dtype=torch.complex64))
        return (1.0 / math.sqrt(n)) * (omega ** (i * j))

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Apply DFT to a state vector."""
        return torch.matmul(self.get_matrix(), state)


class PermuteGate(BaseGate):
    """Thread reordering via a permutation map."""

    def __init__(self, size: int):
        super().__init__("Permute", size)

    def get_matrix(self, permutation: list) -> torch.Tensor:
        """
        Returns an N×N permutation matrix.

        Args:
            permutation: List of target indices. Thread i goes to permutation[i].
        """
        n = self.num_threads
        matrix = torch.zeros(n, n, dtype=torch.complex64)
        for i, j in enumerate(permutation):
            matrix[j, i] = 1.0 + 0j
        return matrix

    def apply(self, state: torch.Tensor, permutation: list) -> torch.Tensor:
        """Apply permutation to state vector."""
        return torch.matmul(self.get_matrix(permutation), state)


class ReverseGate(BaseGate):
    """Time-reversal gate: flips thread order (anti-diagonal permutation)."""

    def __init__(self, size: int):
        super().__init__("Reverse", size)

    def get_matrix(self) -> torch.Tensor:
        """Returns the anti-diagonal permutation matrix."""
        n = self.num_threads
        perm = list(range(n - 1, -1, -1))
        matrix = torch.zeros(n, n, dtype=torch.complex64)
        for i, j in enumerate(perm):
            matrix[j, i] = 1.0 + 0j
        return matrix

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Apply time-reversal to state vector."""
        return torch.flip(state, dims=[0])


class AccumulateGate(BaseGate):
    """
    Coherent amplitude summation on a bus.

    Sums all target thread amplitudes into the first target thread.
    Used for counting, aggregation, and wavefront summation problems.
    """

    def __init__(self):
        super().__init__("Accumulate", -1)  # Variable threads

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("AccumulateGate is a non-linear operation.")

    def apply(self, state: torch.Tensor, targets: list = None) -> torch.Tensor:
        """
        Sum amplitudes of target threads into the first target thread.

        Args:
            state: Complex state vector [N].
            targets: List of thread indices to accumulate (default: all).

        Returns:
            Updated state vector.
        """
        new_state = state.clone()
        if targets is None:
            targets = list(range(state.shape[0]))
        total = torch.sum(state[targets])
        new_state[targets[0]] = total
        return new_state


class GridPropagateGate(BaseGate):
    """
    2D wavefront propagation for DP-grid problems.

    Each cell sums its incoming wavefronts from above and from the left
    (Huygens summation). Used for lattice path counting, LCS, edit distance.
    """

    def __init__(self):
        super().__init__("GridPropagate", -1)

    def get_matrix(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("GridPropagateGate is a non-linear operation.")

    def apply(self, state_2d: torch.Tensor,
              direction: str = 'right_down') -> torch.Tensor:
        """
        Propagate wavefront through a 2D grid.

        Args:
            state_2d: 2D complex tensor [M × N].
            direction: 'right_down' (default) or 'all'.

        Returns:
            Evolved 2D state tensor.
        """
        m, n = state_2d.shape
        result = state_2d.clone()
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                val = torch.tensor(0.0, dtype=torch.complex64)
                if i > 0:
                    val = val + result[i - 1, j]
                if j > 0:
                    val = val + result[i, j - 1]
                result[i, j] = val
        return result
