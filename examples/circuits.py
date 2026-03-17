#!/usr/bin/env python3

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import phasorflow as pf
from phasorflow.engine.analytic import AnalyticEngine


def main() -> None:
    circuit = pf.PhasorCircuit(2, name="BasicCircuit")
    circuit.shift(0, math.pi / 2)
    circuit.mix(0, 1)

    engine = AnalyticEngine()
    result = engine.run(circuit)

    pf.draw(circuit, mode="text")
    print("phases:", result["phases"])
    print("amplitudes:", result["amplitudes"])


if __name__ == "__main__":
    main()