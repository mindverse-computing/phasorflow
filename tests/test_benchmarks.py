# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.

"""
Offline smoke tests for the benchmarks and data modules.

These do NOT hit the network. They verify that the synthetic task family,
baselines, and depth-study utilities import and run, and that the phase-parity
task is genuinely non-linear (baselines behave as documented). The real-EEG
benchmark itself requires a PhysioNet download and is exercised by
examples/ex_09_real_eeg_vpc.py, not by CI.

Run with:  PYTHONPATH=. pytest phasorflow/tests/test_benchmarks.py -q
"""

import numpy as np
import torch

from phasorflow.benchmarks import (
    sum_cosine_task, phase_parity_task, classification_baselines,
)
from phasorflow.benchmarks.depth_study import verify_gradient_flow, vpc_capacity


def test_task_shapes():
    X, y = phase_parity_task(200, 8, k=2, seed=0)
    assert X.shape == (200, 8)
    assert set(np.unique(y.numpy())) <= {0.0, 1.0}
    # balanced-ish
    assert 0.3 < y.float().mean().item() < 0.7


def test_parity_is_nonlinear():
    # Logistic regression should be near chance on phase-XOR; an RBF-SVM should
    # be clearly above chance. This is the property the capacity analysis rests
    # on, so we guard it.
    Xtr, ytr = phase_parity_task(600, 8, k=2, seed=1)
    Xte, yte = phase_parity_task(300, 8, k=2, seed=2)
    res = classification_baselines(Xtr, ytr, Xte, yte, feature="cos")
    # values are (test_acc, n_params) tuples
    assert res["LogReg"][0] < 0.65       # linear model ~ chance
    assert res["SVM-rbf"][0] > 0.75       # kernel model solves it


def test_gradient_reaches_all_stacks():
    norms = verify_gradient_flow(num_stacks=4)
    assert len(norms) == 4
    assert all(g > 0 for g in norms), norms


def test_vpc_capacity_ceiling():
    # VPC stays near chance on phase-XOR regardless of depth (quick config).
    res = vpc_capacity(depths=(1, 2), k=2, seeds=1, epochs=80)
    for r in res:
        assert r["acc_mean"] < 0.65, r


def test_data_module_importable():
    # The real-data pipeline must import without a network call.
    from phasorflow.data import (
        prepare_dataset, download_subject, load_subject_epochs,
        DEFAULT_SUBJECTS, DEFAULT_RUNS,
    )
    assert DEFAULT_RUNS == (4, 8, 12)
    assert len(DEFAULT_SUBJECTS) == 10
