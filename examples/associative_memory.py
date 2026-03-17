#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phasorflow.neuromorphic.associative_memory import PhasorFlowMemory


def main() -> None:
    memory = PhasorFlowMemory(num_threads=4)
    patterns = [
        [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
        [0.1, 1.2, 2.4, 3.5],
    ]
    memory.store(patterns)

    noisy = torch.exp(1j * torch.tensor([0.0, 1.45, 3.0, 4.8]))
    recovered = memory.converge(noisy, iterations=30, dt=0.1)

    print("recovered phases:", torch.angle(recovered))


if __name__ == "__main__":
    main()