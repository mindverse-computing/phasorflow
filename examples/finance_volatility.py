#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import phasorflow as pf
from phasorflow.engine.analytic import AnalyticEngine


def main() -> None:
    returns = torch.tensor([0.01, -0.03, 0.02, -0.01, 0.04, -0.02])
    phases = returns * (2 * math.pi)

    circuit = pf.PhasorCircuit(len(phases), name="VolatilitySpectrum")
    circuit.encode_phases(phases, max_val=2 * math.pi)
    circuit.dft()

    result = AnalyticEngine().run(circuit)
    print("input returns:", returns)
    print("spectral amplitudes:", result["amplitudes"])


if __name__ == "__main__":
    main()