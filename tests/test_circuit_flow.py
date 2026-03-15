# tests/test_circuit_flow.py

import torch
import math
from phasorflow import PhasorCircuit
from phasorflow.engine.analytic import AnalyticEngine

def test_full_adder_logic():
    # 1. Setup
    pc = PhasorCircuit(2)
    engine = AnalyticEngine()
    
    # 2. Rotate first circle by PI, second by 0
    pc.shift(0, math.pi)
    
    # 3. Mix them (Interference)
    pc.mix(0, 1)
    
    # 4. Run
    result = engine.run(pc)
    
    # 5. Assert: The mix of PI and 0 should result in specific amplitudes
    # 1/sqrt(2) * [1, j] @ [exp(j*pi), 1] -> 1/sqrt(2) * [-1+j, -j+1]
    expected_val_0 = (1 / math.sqrt(2)) * torch.tensor(-1.0 + 1.0j, dtype=torch.complex64)
    expected_val_1 = (1 / math.sqrt(2)) * torch.tensor(1.0 - 1.0j, dtype=torch.complex64)
    
    state = result['state_vector']
    assert torch.allclose(state[0], expected_val_0, atol=1e-5)
    assert torch.allclose(state[1], expected_val_1, atol=1e-5)
