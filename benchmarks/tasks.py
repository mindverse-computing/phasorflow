"""
Honest benchmark tasks + baseline harness for PhasorFlow.

Two synthetic classification families, designed so that separability is
*known* rather than assumed:

  * ``sum_cosine_task``  — label = sign(sum_i cos x_i).  Linearly separable in
    cosine-feature space (LogReg ~= 96%).  This is the family the original demo
    used; kept only to document that it is trivial.

  * ``phase_parity_task`` — label = XOR of k phase-quadrant indicators.  A
    parity function: provably NOT linearly separable in phase or cos/sin
    features.  Difficulty scales with k.  A model that beats logistic
    regression here is doing genuine non-linear work.

Baselines (all on identical splits, honest parameter counts):
  persistence (regression only), logistic regression, RBF-SVM, small MLP,
  and — for sequences — a PyTorch self-attention encoder.
"""

import math
import numpy as np
import torch


# ----------------------------------------------------------------------
# Classification task families
# ----------------------------------------------------------------------

def sum_cosine_task(n, N=16, seed=0):
    """Trivial demo task: label = 1[sum_i cos(x_i) > 0]. Linearly separable
    in cos-features (documented, not recommended as evidence)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.empty(n, N).uniform_(0, 2 * math.pi, generator=g)
    y = (torch.cos(X).sum(1) > 0).float()
    return X, y


def phase_parity_task(n, N=8, k=2, seed=0):
    """Genuinely non-linear task: label = (# of first-k threads with cos>0) mod 2.
    A k-way XOR/parity. Not linearly separable; difficulty rises with k."""
    g = torch.Generator().manual_seed(seed)
    X = torch.empty(n, N).uniform_(0, 2 * math.pi, generator=g)
    y = ((torch.cos(X[:, :k]) > 0).long().sum(1) % 2).float()
    return X, y


# ----------------------------------------------------------------------
# Sequence task (for the Phasor Transformer)
# ----------------------------------------------------------------------

def multifreq_sequence_task(n, T=10, seed=0):
    """Autoregressive next-step prediction of a sum of 3 random sinusoids +
    Gaussian noise, mapped to phase domain [-pi/2, pi/2]. Returns (X, y)."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n):
        f = rng.uniform(0.1, 0.5, 3)
        p = rng.uniform(0, 2 * np.pi, 3)
        t = np.arange(T + 1)
        s = sum(np.sin(fi * t + pi) for fi, pi in zip(f, p)) / 3
        s = s + rng.normal(0, 0.1, T + 1)
        s = np.clip(s, -1, 1) * (math.pi / 2)
        Xs.append(s[:T]); ys.append(s[T])
    return (torch.tensor(np.array(Xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32))


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------

def classification_baselines(Xtr, ytr, Xte, yte, feature='cos'):
    """Return {name: (test_acc, n_params)} for LogReg / RBF-SVM / MLP.
    Features: 'cos' applies cos() (matches the phasor encoding nonlinearity),
    'raw' uses raw phases."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    f = (lambda a: np.cos(a)) if feature == 'cos' else (lambda a: a)
    Xtr_f, Xte_f = f(Xtr.numpy()), f(Xte.numpy())
    ytr_n, yte_n = ytr.numpy(), yte.numpy()
    out = {}

    lr = LogisticRegression(max_iter=3000).fit(Xtr_f, ytr_n)
    out['LogReg'] = (lr.score(Xte_f, yte_n), lr.coef_.size + lr.intercept_.size)

    svm = SVC(kernel='rbf').fit(Xtr_f, ytr_n)
    out['SVM-rbf'] = (svm.score(Xte_f, yte_n), int(svm.support_vectors_.size))

    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                        random_state=0).fit(Xtr_f, ytr_n)
    n_p = sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    out['MLP-32'] = (mlp.score(Xte_f, yte_n), int(n_p))
    return out


def regression_baselines(Xtr, ytr, Xte, yte):
    """Return {name: (test_mse, n_params)} for persistence / linear / MLP."""
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor

    out = {}
    persist = torch.mean((Xte[:, -1] - yte) ** 2).item()
    out['persistence'] = (persist, 0)

    lin = LinearRegression().fit(Xtr.numpy(), ytr.numpy())
    pred = lin.predict(Xte.numpy())
    out['linear'] = (float(np.mean((pred - yte.numpy()) ** 2)),
                     lin.coef_.size + 1)

    mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=2000,
                       random_state=0).fit(Xtr.numpy(), ytr.numpy())
    pred = mlp.predict(Xte.numpy())
    n_p = sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    out['MLP-32'] = (float(np.mean((pred - yte.numpy()) ** 2)), int(n_p))
    return out


class SelfAttnRegressor(torch.nn.Module):
    """Standard PyTorch self-attention encoder baseline for sequence
    regression (honest parameter count via .numel())."""

    def __init__(self, seq_len, d_model=16, nhead=4, dim_ff=64):
        super().__init__()
        self.proj = torch.nn.Linear(1, d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.head = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x.unsqueeze(-1))
        h = self.enc(h)
        return self.head(h.mean(1)).squeeze(-1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
