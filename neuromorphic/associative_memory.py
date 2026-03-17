# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/neuromorphic/associative_memory.py

import torch

class PhasorFlowMemory:
    def __init__(self, num_threads: int):
        self.n = num_threads
        # Complex weight matrix to store phase relations
        self.weights = torch.zeros((num_threads, num_threads), dtype=torch.complex64)

    def store(self, phase_patterns: list):
        """
        Stores patterns into the oscillatory memory using Hebbian learning.
        Each pattern should be an array of phases in radians.
        """
        for pattern in phase_patterns:
            phasors = torch.exp(1j * torch.tensor(pattern, dtype=torch.float32))
            # Hebbian rule: W = outer(z, conj(z))
            self.weights += torch.outer(phasors, torch.conj(phasors))
        
        # Normalize and remove self-connections
        self.weights /= len(phase_patterns)
        self.weights.fill_diagonal_(0.0 + 0.0j)

    def converge(self, state_vector: torch.Tensor, iterations: int = 50, dt: float = 0.1) -> torch.Tensor:
        """
        Evolves the state vector until it settles into the nearest stored attractor.
        """
        current_state = state_vector.clone()
        for _ in range(iterations):
            # Compute the 'Net Field' acting on each phasor
            # Each phasor is pulled by the weighted average of its neighbors
            field = torch.matmul(self.weights, current_state)
            
            # Update: Move toward the field (Phase-Locking)
            # This is essentially the Kuramoto update rule
            new_state = current_state + (field * dt)
            
            # Re-normalize to the unit circle
            current_state = new_state / torch.abs(new_state)
        
        return current_state
