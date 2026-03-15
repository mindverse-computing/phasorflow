# tests/test_engine.py

import torch
import math
from phasorflow import PhasorCircuit
from phasorflow.engine.analytic import AnalyticEngine

def test_analytic_engine_state_evolution():
    # Setup generator topology
    pc = PhasorCircuit(3)
    engine = AnalyticEngine()
    
    # Pipeline operations
    pc.shift(0, math.pi / 2)
    pc.shift(1, math.pi)
    pc.mix(0, 1)
    pc.dft()
    
    # Execute auto-diff engine
    result = engine.run(pc)
    
    # Tensor Integrity Assertions
    state = result['state_vector']
    phases = result['phases']
    
    assert isinstance(state, torch.Tensor), "State vector should be a PyTorch Tensor."
    assert state.dtype == torch.complex64, "State vector must support complex math natively."
    assert state.shape == (3,), "State output dimensionality mismatched."
    
    assert isinstance(phases, torch.Tensor), "Phase angle extraction should remain a Tensor."
    assert phases.dtype == torch.float32, "Phase angles strictly require float formats."
