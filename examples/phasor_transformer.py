#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phasorflow.models.transformer import PhasorTransformer


def build_dataset(seq_len: int = 6, points: int = 40):
    t = torch.linspace(0, 4 * math.pi, points)
    signal = torch.sin(t)

    X = []
    y = []
    for i in range(points - seq_len - 1):
        X.append(signal[i : i + seq_len])
        y.append(signal[i + seq_len])
    return torch.stack(X), torch.tensor(y)


def main() -> None:
    X, y = build_dataset()
    model = PhasorTransformer(seq_length=X.shape[1], num_blocks=1)
    model.fit(X, y, epochs=30, lr=0.03, verbose=False)

    mse = model.score(X, y)
    pred = model.predict(X[:1])[0].item()
    print(f"sample prediction: {pred:.4f}")
    print(f"training mse: {mse:.4f}")


if __name__ == "__main__":
    main()