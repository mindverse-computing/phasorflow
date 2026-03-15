# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/circuit.py

"""
PhasorCircuit: The High-Level Circuit Construction API.

Provides a fluent interface for building phasor circuits on the
unit-circle manifold. Instructions are stored as a lightweight list
of (gate_name, targets, params) tuples, then dispatched to the
engine at execution time.

Gate Inventory (22 gates):
    Standard Unitary:  shift, mix, dft, invert, permute, reverse
    Pull-Back:         threshold, saturate, normalize, pullback, log_compress
    Neuromorphic:      kuramoto, hebbian, ising, asymmetric_couple
    Symphony-Specific: accumulate, cross_correlate, convolve, grid_propagate
    Encoding:          encode_phases, encode_amplitudes
    Control:           barrier, measure

Usage:
    from phasorflow import PhasorCircuit
    circ = PhasorCircuit(4, name="MyCircuit")
    circ.shift(0, phi=1.57).mix(0, 1).dft().measure("result")
"""

from typing import List, Dict, Any, Tuple, Optional
import math


class PhasorCircuit:
    """
    A declarative phasor circuit on the unit-circle manifold.

    Each thread represents one oscillator on S¹. Gates are appended
    to an instruction list and executed lazily by the Simulator engine.
    """

    def __init__(self, num_threads: int, name: str = "PhasorCircuit"):
        """
        Initialize a phasor circuit.

        Args:
            num_threads: Number of oscillator threads (wires).
            name: Human-readable circuit name.
        """
        self.num_threads = num_threads
        self.name = name
        self.data: List[Tuple[str, List[int], Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Standard Unitary Gates
    # ------------------------------------------------------------------

    def shift(self, thread_idx: int, phi: float) -> 'PhasorCircuit':
        """Phase rotation gate: z_i → e^{jφ} · z_i."""
        self._validate_thread(thread_idx)
        self.data.append(('shift', [thread_idx], {'phi': phi}))
        return self

    def mix(self, thread_a: int, thread_b: int) -> 'PhasorCircuit':
        """50/50 beam-splitter (interference junction) between two threads."""
        self._validate_thread(thread_a)
        self._validate_thread(thread_b)
        self.data.append(('mix', [thread_a, thread_b], {}))
        return self

    def dft(self, threads: Optional[List[int]] = None) -> 'PhasorCircuit':
        """Global or partial Discrete Fourier Transform."""
        targets = threads if threads else list(range(self.num_threads))
        for t in targets:
            self._validate_thread(t)
        self.data.append(('dft', targets, {}))
        return self

    def invert(self, thread_idx: int) -> 'PhasorCircuit':
        """Phase inversion (π-shift): z_i → -z_i."""
        return self.shift(thread_idx, math.pi)

    def permute(self, permutation: List[int]) -> 'PhasorCircuit':
        """Reorder threads according to a permutation map."""
        if len(permutation) != self.num_threads:
            raise ValueError(
                f"Permutation length {len(permutation)} != num_threads {self.num_threads}"
            )
        self.data.append(('permute', list(range(self.num_threads)),
                         {'permutation': permutation}))
        return self

    def reverse(self) -> 'PhasorCircuit':
        """Time-reversal gate: flip thread order."""
        self.data.append(('reverse', list(range(self.num_threads)), {}))
        return self

    # ------------------------------------------------------------------
    # Non-Linear (Pull-Back) Gates
    # ------------------------------------------------------------------

    def threshold(self, threshold: float = 0.5) -> 'PhasorCircuit':
        """Inhibitory gating: zeroes threads with amplitude < threshold."""
        self.data.append(('threshold', list(range(self.num_threads)),
                         {'threshold': threshold}))
        return self

    def saturate(self, levels: int = 2) -> 'PhasorCircuit':
        """Phase quantization: snap phases to N discrete levels."""
        self.data.append(('saturate', list(range(self.num_threads)),
                         {'levels': levels}))
        return self

    def normalize(self) -> 'PhasorCircuit':
        """Re-project all threads onto the unit circle (|z_i| = 1)."""
        self.data.append(('normalize', list(range(self.num_threads)), {}))
        return self

    def pullback(self) -> 'PhasorCircuit':
        """Canonical Pull-Back Operator: z/|z|. Alias for normalize."""
        return self.normalize()

    def log_compress(self, beta: float = 1.0) -> 'PhasorCircuit':
        """Logarithmic amplitude compression to prevent overflow."""
        self.data.append(('log_compress', list(range(self.num_threads)),
                         {'beta': beta}))
        return self

    # ------------------------------------------------------------------
    # Neuromorphic Gates
    # ------------------------------------------------------------------

    def kuramoto(self, weights, dt: float = 0.01,
                 coupling_k: float = 1.0) -> 'PhasorCircuit':
        """Kuramoto phase-lock synchronization step."""
        self.data.append(('kuramoto', list(range(self.num_threads)),
                         {'weights': weights, 'dt': dt,
                          'coupling_k': coupling_k}))
        return self

    def hebbian(self, weights, alpha: float = 0.1) -> 'PhasorCircuit':
        """Hebbian associative relaxation step."""
        self.data.append(('hebbian', list(range(self.num_threads)),
                         {'weights': weights, 'alpha': alpha}))
        return self

    def ising(self, adjacency, dt: float = 0.05,
              coupling_k: float = 1.0) -> 'PhasorCircuit':
        """Binary Ising coupling: drives phases toward 0 or π."""
        self.data.append(('ising', list(range(self.num_threads)),
                         {'adjacency': adjacency, 'dt': dt,
                          'coupling_k': coupling_k}))
        return self

    def asymmetric_couple(self, dag_matrix,
                          dt: float = 0.01) -> 'PhasorCircuit':
        """Directional DAG coupling for ordering problems."""
        self.data.append(('asymmetric_couple', list(range(self.num_threads)),
                         {'dag_matrix': dag_matrix, 'dt': dt}))
        return self

    # ------------------------------------------------------------------
    # Symphony-Specific Operations
    # ------------------------------------------------------------------

    def accumulate(self, targets: Optional[List[int]] = None) -> 'PhasorCircuit':
        """Coherent amplitude summation on a bus."""
        t = targets if targets else list(range(self.num_threads))
        self.data.append(('accumulate', t, {}))
        return self

    def cross_correlate(self, pattern_data) -> 'PhasorCircuit':
        """Sliding-window phasor cross-correlation."""
        self.data.append(('cross_correlate', list(range(self.num_threads)),
                         {'pattern': pattern_data}))
        return self

    def convolve(self) -> 'PhasorCircuit':
        """Self-convolution for combinatorial counting."""
        self.data.append(('convolve', list(range(self.num_threads)), {}))
        return self

    def grid_propagate(self, rows: int, cols: int) -> 'PhasorCircuit':
        """2D wavefront propagation for DP-grid problems."""
        self.data.append(('grid_propagate', list(range(self.num_threads)),
                         {'rows': rows, 'cols': cols}))
        return self

    # ------------------------------------------------------------------
    # Encoding Gates
    # ------------------------------------------------------------------

    def encode_phases(self, values, max_val: float = 1.0) -> 'PhasorCircuit':
        """Encode classical values as phases: θ = (x/max) · 2π."""
        self.data.append(('encode_phases', list(range(self.num_threads)),
                         {'values': values, 'max_val': max_val}))
        return self

    def encode_amplitudes(self, values) -> 'PhasorCircuit':
        """Encode classical values as amplitudes on threads."""
        self.data.append(('encode_amplitudes', list(range(self.num_threads)),
                         {'values': values}))
        return self

    # ------------------------------------------------------------------
    # Control Instructions
    # ------------------------------------------------------------------

    def barrier(self) -> 'PhasorCircuit':
        """Synchronization barrier (marks a layer boundary)."""
        self.data.append(('barrier', [], {}))
        return self

    def measure(self, label: str = '') -> 'PhasorCircuit':
        """Insert a measurement snapshot point."""
        self.data.append(('measure', list(range(self.num_threads)),
                         {'label': label}))
        return self

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, other: 'PhasorCircuit') -> 'PhasorCircuit':
        """Append another circuit's instructions to this one."""
        if other.num_threads != self.num_threads:
            raise ValueError("Cannot compose circuits with different thread counts.")
        self.data.extend(other.data)
        return self

    def repeat(self, times: int) -> 'PhasorCircuit':
        """Repeat the current circuit instructions N times."""
        original = list(self.data)
        for _ in range(times - 1):
            self.data.extend(original)
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Number of gate layers (barriers count as layer separators)."""
        layers = 1
        for gate_name, _, _ in self.data:
            if gate_name == 'barrier':
                layers += 1
        return layers

    @property
    def gate_count(self) -> int:
        """Total number of gates (excluding barriers and measures)."""
        return sum(1 for g, _, _ in self.data
                   if g not in ('barrier', 'measure'))

    def _validate_thread(self, idx: int):
        if not (0 <= idx < self.num_threads):
            raise IndexError(
                f"Thread index {idx} out of range [0, {self.num_threads})"
            )

    def __repr__(self):
        return (f"PhasorCircuit(name='{self.name}', "
                f"threads={self.num_threads}, "
                f"gates={self.gate_count})")

    def __len__(self):
        return len(self.data)
