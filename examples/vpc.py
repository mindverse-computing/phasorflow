#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phasorflow.models.vpc import VPC


def main() -> None:
    X = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.2],
        [math.pi / 2, math.pi / 2],
        [1.5, 1.4],
    ], dtype=torch.float32)
    y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    model = VPC(num_features=2, num_layers=1)
    model.fit(X, y, epochs=30, lr=0.05, verbose=False)

    preds = model.predict(X)
    acc = model.score(X, y)
    print("predictions:", preds)
    print(f"accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()