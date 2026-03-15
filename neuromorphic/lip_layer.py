# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/neuromorphic/lip_layer.py

import torch

class LIPLayer:
    def __init__(self, num_threads: int, leak_rate: float = 0.1, rest_phase: float = 0.0):
        """
        Leaky-Integrate-and-Phase Layer.
        
        Args:
            num_threads: Number of oscillators (neurons).
            leak_rate: How fast the phase returns to rest_phase (gamma).
            rest_phase: The baseline phase of the system.
        """
        self.num_threads = num_threads
        self.gamma = leak_rate
        self.theta_rest = rest_phase
        self.weights = torch.rand((num_threads, num_threads), dtype=torch.float32) * 0.1

    def update(self, state_vector: torch.Tensor, external_input: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Performs one time-step of the LIP integration on a given state vector.
        """
        current_phases = torch.angle(state_vector)
        
        # 1. Calculate Leakage: Move phase toward rest
        leak = -self.gamma * (current_phases - self.theta_rest)
        
        # 2. Integrate: Coupling between oscillators (Synaptic weights)
        # In LIP, weights represent how much one phase pulls another (Kuramoto-like)
        phase_diff = current_phases.unsqueeze(1) - current_phases.unsqueeze(0)
        coupling = torch.matmul(self.weights, torch.sin(phase_diff))
        
        # 3. Total Phase Change
        d_theta = (leak + torch.diagonal(coupling) + external_input) * dt
        new_phases = current_phases + d_theta
        
        # 4. Return new State Vector
        return torch.exp(1j * new_phases)
