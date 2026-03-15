# tests/test_neuromorphic.py

import torch
import math
from phasorflow.neuromorphic.associative_memory import PhasorFlowMemory
from phasorflow.neuromorphic.lip_layer import LIPLayer

def test_associative_memory():
    num_threads = 4
    memory = PhasorFlowMemory(num_threads)
    
    # Store a target limit-cycle pattern
    pattern1 = [0.0, math.pi/2, math.pi, 3*math.pi/2]
    memory.store([pattern1])
    
    # Generate an imperfectly aligned initial tensor state
    initial_phases = torch.tensor([0.1, math.pi/2 + 0.1, math.pi - 0.1, 3*math.pi/2 + 0.1], dtype=torch.float32)
    initial_state = torch.exp(1j * initial_phases)
    
    # Evaluate explicit convergence logic
    final_state = memory.converge(initial_state, iterations=10)
    
    # Check that magnitude is strictly locked back to Torus radius=1
    magnitudes = torch.abs(final_state)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5), "Associative limits fail to resolve magnitude constraint!"
    
    assert isinstance(final_state, torch.Tensor)
    assert final_state.dtype == torch.complex64

def test_lip_layer():
    num_threads = 3
    layer = LIPLayer(num_threads, leak_rate=0.2, rest_phase=0.0)
    
    # Set continuous physics parameters
    initial_phases = torch.tensor([math.pi/4, math.pi/2, math.pi], dtype=torch.float32)
    initial_state = torch.exp(1j * initial_phases)
    external_input = torch.tensor([0.1, 0.0, -0.1], dtype=torch.float32)
    
    new_state = layer.update(initial_state, external_input, dt=0.01)
    
    # Enforce LIP parameter limits
    assert isinstance(new_state, torch.Tensor)
    assert new_state.dtype == torch.complex64
    assert new_state.shape == (num_threads,)
    
    # Magnitude should dynamically reform to exactly 1.0 due to strict phase casting
    magnitudes = torch.abs(new_state)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5), "LIP Output Magnitude shifted off N-Torus continuously."
