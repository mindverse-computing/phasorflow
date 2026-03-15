# tests/test_gates.py

import pytest
import torch
import math
from phasorflow.gates.standard import ShiftGate, MixGate, DFTGate

def test_shift_gate():
    gate = ShiftGate()
    phi = math.pi / 2
    matrix = gate.get_matrix(phi=phi)
    # 90 degree shift should be purely imaginary
    expected = torch.tensor(1.0j, dtype=torch.complex64)
    assert torch.allclose(matrix[0, 0], expected, atol=1e-5)

def test_mix_gate_unitary():
    gate = MixGate()
    matrix = gate.get_matrix()
    # Check if matrix is unitary: M * M_dagger = I
    identity_check = torch.matmul(matrix, matrix.conj().T)
    expected_eye = torch.eye(2, dtype=torch.complex64)
    assert torch.allclose(identity_check, expected_eye, atol=1e-5)

def test_dft_gate_resonance():
    n = 4
    gate = DFTGate(n)
    matrix = gate.get_matrix()
    # A DFT matrix should also be unitary
    identity_check = torch.matmul(matrix, matrix.conj().T)
    expected_eye = torch.eye(n, dtype=torch.complex64)
    assert torch.allclose(identity_check, expected_eye, atol=1e-5)
