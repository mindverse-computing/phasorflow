"""
Depth-scaling and capacity study for the VPC and Phasor Transformer.

Reproduces the two central architectural findings:

  * VPC capacity ceiling: on the phase-XOR (parity) task the VPC stays at
    chance regardless of stack count, while an RBF-SVM and MLP solve it. The
    VPC is a linear classifier in a fixed cos/sin feature lifting.

  * Phasor Transformer depth benefit: on a variable-period continuation task
    the transformer's test MSE decreases monotonically with block count up to
    ~3 blocks, then saturates. Requires the corrected (v0.3.0) stacking path.

Run:
    python -m phasorflow.benchmarks.depth_study
"""

import json
import math

import numpy as np
import torch

from phasorflow.models.vpc import VPC
from phasorflow.models.transformer import PhasorTransformer
from phasorflow.benchmarks.tasks import phase_parity_task


def _periodic(n, T=16, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n):
        per = rng.uniform(3, 8)
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.6, 1.0)
        t = np.arange(T + 1)
        s = amp * np.sin(2 * np.pi * t / per + ph) + rng.normal(0, 0.05, T + 1)
        s = np.clip(s, -1, 1) * (math.pi / 2)
        Xs.append(s[:T]); ys.append(s[T])
    return (torch.tensor(np.array(Xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32))


def vpc_capacity(depths=(1, 2, 3, 4), k=2, seeds=3, epochs=300):
    """VPC accuracy vs stack count on phase-XOR (should stay ~chance)."""
    Xtr, ytr = phase_parity_task(800, 8, k, seed=1)
    Xte, yte = phase_parity_task(400, 8, k, seed=2)
    out = []
    for s in depths:
        accs = []
        for sd in range(seeds):
            torch.manual_seed(sd)
            m = VPC(8, num_layers=2, num_stacks=s, inter_stack="pullback")
            m.fit(Xtr, ytr, epochs=epochs, lr=0.05, verbose=False)
            accs.append(m.score(Xte, yte))
        out.append({"stacks": s, "params": m.total_params,
                    "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))})
    return out


def transformer_depth(blocks=(1, 2, 3, 4), seeds=3, epochs=200):
    """Phasor Transformer MSE vs block count on variable-period forecast."""
    Xtr, ytr = _periodic(1500, 16, 1)
    Xte, yte = _periodic(400, 16, 2)
    out = []
    for nb in blocks:
        mses = []
        for sd in range(seeds):
            torch.manual_seed(sd)
            m = PhasorTransformer(seq_length=16, num_blocks=nb, readout_layer=True)
            m.fit(Xtr, ytr, epochs=epochs, lr=0.03, verbose=False)
            mses.append(m.score(Xte, yte))
        out.append({"blocks": nb, "params": m.total_params,
                    "mse_mean": float(np.mean(mses)), "mse_std": float(np.std(mses))})
    return out


def verify_gradient_flow(num_stacks=4):
    """Assert every stack receives gradient (regression test for the fix)."""
    torch.manual_seed(0)
    X = torch.empty(64, 8).uniform_(0, 2 * math.pi)
    y = (torch.arange(64) % 2).float()
    m = VPC(8, num_layers=2, num_stacks=num_stacks, inter_stack="pullback")
    m.zero_grad()
    ((m.forward_batch(X) - y) ** 2).mean().backward()
    pps = m.params_per_stack
    norms = [m.weights.grad[i * pps:(i + 1) * pps].norm().item()
             for i in range(num_stacks)]
    assert all(g > 0 for g in norms), f"frozen stack detected: {norms}"
    return norms


def main(out_path="handoff/depth_scaling.json"):
    import os
    grads = verify_gradient_flow()
    print("gradient reaches all stacks:", [round(g, 3) for g in grads])
    cap = vpc_capacity()
    dep = transformer_depth()
    print("\nVPC capacity (phase-XOR):")
    for r in cap:
        print(f"  stacks={r['stacks']} p={r['params']}: {r['acc_mean']:.3f} +/- {r['acc_std']:.3f}")
    print("\nPhasor Transformer depth (periodic forecast):")
    for r in dep:
        print(f"  blocks={r['blocks']} p={r['params']}: {r['mse_mean']:.3f} +/- {r['mse_std']:.3f}")
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump({"vpc_capacity": cap, "transformer_depth": dep,
                   "grad_norms": grads}, open(out_path, "w"), indent=2)
    return cap, dep


if __name__ == "__main__":
    main()
