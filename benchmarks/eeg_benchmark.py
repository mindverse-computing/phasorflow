"""
Real motor-imagery EEG benchmark for the VPC (PhysioNet eegmmidb).

Pipeline: band-pass 8-30 Hz -> Common Spatial Patterns (fit inside each CV
fold) -> standardize -> classify. VPC is compared against LDA, logistic
regression, RBF-SVM, and an MLP under subject-wise stratified 5-fold CV.

Run:
    NUMBA_DISABLE_JIT=1 python -m phasorflow.benchmarks.eeg_benchmark

Produces handoff/eeg_results.json-style output and prints grand means.
"""

import os
import json

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/nb")

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from phasorflow.models.vpc import VPC
from phasorflow.data import prepare_dataset, DEFAULT_SUBJECTS


def run_vpc(Xtr, ytr, Xte, yte, num_stacks=1, seed=0, epochs=300, lr=0.05):
    """Two-layer VPC on standardized features, phase-encoded via pi*tanh."""
    torch.manual_seed(seed)
    N = Xtr.shape[1]
    Xtr_p = torch.tensor(np.pi * np.tanh(Xtr), dtype=torch.float32)
    Xte_p = torch.tensor(np.pi * np.tanh(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    m = VPC(num_features=N, num_layers=2, num_stacks=num_stacks,
            inter_stack="pullback")
    m.fit(Xtr_p, ytr_t, epochs=epochs, lr=lr, verbose=False)
    acc = m.score(Xte_p, torch.tensor(yte, dtype=torch.float32))
    return acc, m.total_params


def evaluate(subjects=DEFAULT_SUBJECTS, n_csp=6, n_splits=5,
             out_path="handoff/eeg_results.json", verbose=True):
    from mne.decoding import CSP
    data = prepare_dataset(subjects=subjects, verbose=verbose)
    rows = []
    for rec in data:
        sid, Xraw, y = rec["subject"], rec["X"], rec["y"]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        accs = {k: [] for k in ["LDA", "LogReg", "SVM", "MLP", "VPC", "VPC-S2"]}
        n_feat = None
        for tr, te in skf.split(Xraw, y):
            csp = CSP(n_components=n_csp, reg="ledoit_wolf", log=True,
                      norm_trace=False)
            Ftr = csp.fit_transform(Xraw[tr], y[tr])
            Fte = csp.transform(Xraw[te])
            n_feat = Ftr.shape[1]
            sc = StandardScaler().fit(Ftr)
            Xtr, Xte = sc.transform(Ftr), sc.transform(Fte)
            ytr, yte = y[tr].astype(float), y[te].astype(float)
            accs["LDA"].append(LinearDiscriminantAnalysis().fit(Xtr, ytr).score(Xte, yte))
            accs["LogReg"].append(LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte))
            accs["SVM"].append(SVC(kernel="rbf").fit(Xtr, ytr).score(Xte, yte))
            accs["MLP"].append(MLPClassifier((32,), max_iter=1500, random_state=0).fit(Xtr, ytr).score(Xte, yte))
            accs["VPC"].append(run_vpc(Xtr, ytr, Xte, yte, num_stacks=1)[0])
            accs["VPC-S2"].append(run_vpc(Xtr, ytr, Xte, yte, num_stacks=2)[0])
        row = {"subject": sid, "n_trials": int(len(y)), "n_features": int(n_feat)}
        for k, v in accs.items():
            row[k] = float(np.mean(v))
            row[k + "_std"] = float(np.std(v))
        rows.append(row)
        if verbose:
            print(f"S{sid}: n={len(y)} " +
                  " ".join(f"{k}={np.mean(v):.3f}" for k, v in accs.items()))
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump(rows, open(out_path, "w"), indent=2)
    if verbose:
        print("\n=== GRAND MEAN across subjects (CSP features) ===")
        for k in ["LDA", "LogReg", "SVM", "MLP", "VPC", "VPC-S2"]:
            vals = [r[k] for r in rows]
            print(f"  {k:8} {np.mean(vals):.3f} +/- {np.std(vals):.3f}")
    return rows


if __name__ == "__main__":
    evaluate()
