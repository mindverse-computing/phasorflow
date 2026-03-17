#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phasorflow.neuromorphic.lip_layer import LIPLayer


def main() -> None:
    layer = LIPLayer(num_threads=3, leak_rate=0.15, rest_phase=0.0)
    state = torch.exp(1j * torch.tensor([0.0, math.pi / 3, math.pi / 2]))
    external = torch.tensor([0.02, 0.0, -0.01])

    for _ in range(25):
        state = layer.update(state, external_input=external, dt=0.02)

    print("final phases:", torch.angle(state))


if __name__ == "__main__":
    main()