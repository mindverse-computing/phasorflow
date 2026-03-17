#!/usr/bin/env python3

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import phasorflow as pf
from phasorflow.engine.analytic import AnalyticEngine


def estimate_period(base: int, modulus: int, max_steps: int = 20) -> int:
    value = 1
    for r in range(1, max_steps + 1):
        value = (value * base) % modulus
        if value == 1:
            return r
    return -1


def main() -> None:
    base = 2
    modulus = 15
    period = estimate_period(base, modulus)

    circuit = pf.PhasorCircuit(2, name="PeriodEncoding")
    circuit.shift(0, (2 * math.pi / modulus) * base)
    circuit.shift(1, (2 * math.pi / modulus) * period)
    circuit.mix(0, 1)

    result = AnalyticEngine().run(circuit)
    print(f"base={base}, modulus={modulus}, period={period}")
    print("encoded phases:", result["phases"])


if __name__ == "__main__":
    main()