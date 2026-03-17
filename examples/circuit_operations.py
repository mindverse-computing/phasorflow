#!/usr/bin/env python3

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import phasorflow as pf
from phasorflow.engine.analytic import AnalyticEngine


def main() -> None:
    circuit = pf.PhasorCircuit(4, name="ThresholdFilter")
    circuit.shift(0, math.pi)
    circuit.shift(1, math.pi / 4)
    circuit.shift(2, math.pi / 2)
    circuit.shift(3, 0.1)
    circuit.mix(0, 1)
    circuit.mix(2, 3)
    circuit.threshold(threshold=0.9)
    circuit.normalize()

    result = AnalyticEngine().run(circuit)
    survivors = (result["amplitudes"] > 0.5).sum().item()

    print("surviving signals:", int(survivors))
    print("amplitudes:", result["amplitudes"])


if __name__ == "__main__":
    main()