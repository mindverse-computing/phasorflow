# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

#!/usr/bin/env python
# coding: utf-8

# # PhasorFlow — Associative Memory & Pattern Recognition
# 
# ## Phasor-Based Content-Addressable Memory Systems
# 
# This notebook explores **associative memory** — systems that store patterns and recall the closest match from a noisy or partial input. PhasorFlow implements this using phase-based Hebbian learning on the unit circle.
# 
# | # | Example | Gates/Modules Used | Concept |
# |---|---------|-------------------|---------|
# | 1 | Basic Pattern Recovery | PhasorFlowMemory | Hopfield-phase model attractor |
# | 2 | Multi-Pattern Storage | HebbianGate, Normalize | Storing/recalling multiple patterns |
# | 3 | Image Denoising | EncodePhase, Hebbian, Saturate | Recovering corrupted binary images |
# | 4 | Capacity Limits | HebbianGate | Memory capacity vs storage count |
# | 5 | Hetero-Associative Memory | Hebbian, DFT | Input-output pattern mapping |

# In[1]:


import sys, os, shutil
parent = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..', '..'))
if parent not in sys.path:
    sys.path.insert(0, parent)
for root, dirs, _ in os.walk(os.path.join(parent, 'PhasorFlow')):
    for d in dirs:
        if d == '__pycache__':
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
for key in list(sys.modules.keys()):
    if 'phasorflow' in key.lower():
        del sys.modules[key]

import phasorflow as pf
from phasorflow import PhasorCircuit
from phasorflow.engine.analytic import AnalyticEngine
from phasorflow.gates import *
from phasorflow.neuromorphic.lip_layer import LIPLayer
from phasorflow.neuromorphic.associative_memory import PhasorFlowMemory
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(precision=4, sci_mode=False)
engine = AnalyticEngine()

def print_circuit_stats(circ):
    print(f"  Circuit:  {circ.name}")
    print(f"  Threads:  {circ.num_threads}, Gates: {circ.gate_count}, Depth: {circ.depth}")

def print_results(result):
    print(f"  Phases:     {result['phases']}")
    print(f"  Amplitudes: {result['amplitudes']}")

print(f"phasorflow v{pf.__version__} loaded")


# ---
# ## 1. Basic Pattern Recovery — Hopfield-Phase Attractor
# 
# Store a binary phase pattern (0 or π) and recover it from a corrupted input. The phase-Hopfield network converges to the nearest stored attractor.
# 
# **Module:** `PhasorFlowMemory`

# In[2]:


print("=" * 60)
print("  1. Basic Pattern Recovery (Hopfield-Phase Model)")
print("=" * 60)

n = 10
mem = PhasorFlowMemory(n)

# Store a binary phase pattern
stored_pattern = [0.0, math.pi, 0.0, math.pi, 0.0, math.pi, 0.0, math.pi, 0.0, math.pi]
mem.store([stored_pattern])
stored_tensor = torch.tensor(stored_pattern, dtype=torch.float32)

# Corrupt 2 positions
corrupted = stored_tensor.clone()
corrupted[0] = math.pi  # was 0
corrupted[1] = 0.0      # was π
state = torch.exp(1j * corrupted)

def phase_diff(a, b):
    diff = torch.abs(a - b)
    return torch.minimum(diff, 2*math.pi - diff)

initial_err = torch.mean(phase_diff(torch.angle(state), stored_tensor))
print(f"  Initial Error: {initial_err.item():.4f} rad")

# Converge
state = mem.converge(state, iterations=100)
final_phases = torch.angle(state)

err1 = torch.mean(phase_diff(final_phases, stored_tensor))
err2 = torch.mean(phase_diff(final_phases, (stored_tensor + math.pi) % (2*math.pi)))
final_err = min(err1.item(), err2.item())
print(f"  Final Error:   {final_err:.4f} rad")
print(f"  Status:        {'✓ RECOVERED' if final_err < 0.2 else '✗ FAILED'}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, data, title in zip(axes,
    [stored_pattern, corrupted.numpy(), final_phases.detach().numpy()],
    ['Stored Pattern', 'Corrupted Input', 'Recovered']):
    colors = ['#3498db' if abs(p) < 1.0 else '#e74c3c' for p in data]
    ax.bar(range(n), data, color=colors, edgecolor='black', alpha=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Thread'); ax.set_ylabel('Phase')
    ax.set_ylim(-0.5, 4.0); ax.grid(True, alpha=0.3)
plt.suptitle('Hopfield-Phase Pattern Recovery', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()


# ---
# ## 2. Multi-Pattern Storage — Hebbian Memory with Multiple Attractors
# 
# Store 4 distinct patterns and show that the network reliably recalls the nearest one. Uses PhasorFlow's `HebbianGate.store_patterns()` to build the weight matrix.
# 
# **Gates:** `shift` → `hebbian` (×iterations) → `normalize`

# In[3]:


print("=" * 60)
print("  2. Multi-Pattern Storage and Selective Recall")
print("=" * 60)

N = 8
patterns = [
    [0, 0, 0, 0, math.pi, math.pi, math.pi, math.pi],       # A: 0000ππππ
    [math.pi, math.pi, math.pi, math.pi, 0, 0, 0, 0],       # B: ππππ0000
    [0, math.pi, 0, math.pi, 0, math.pi, 0, math.pi],       # C: 0π0π0π0π
    [math.pi, 0, math.pi, 0, math.pi, 0, math.pi, 0],       # D: π0π0π0π0
]
labels = ['A: 0000ππππ', 'B: ππππ0000', 'C: 0π0π0π0π', 'D: π0π0π0π0']

W = HebbianGate.store_patterns(N, patterns)

# Test: give noisy version of Pattern C
torch.manual_seed(123)
target_idx = 2  # Pattern C
noisy = torch.tensor(patterns[target_idx], dtype=torch.float32)
noisy = noisy + torch.randn(N) * 0.5  # Add noise

# Build circuit
circ = PhasorCircuit(N, name="MultiPatternRecall")
for i in range(N):
    circ.shift(i, noisy[i].item())
circ.barrier()
for _ in range(50):
    circ.hebbian(W, alpha=0.1)
circ.barrier()
circ.normalize()

result = engine.run(circ)
recalled = result['phases']

# Find best match
def pattern_error(recalled_phases, pattern):
    p = torch.tensor(pattern, dtype=torch.float32)
    d1 = torch.mean(torch.minimum(torch.abs(recalled_phases - p),
                                   2*math.pi - torch.abs(recalled_phases - p)))
    d2 = torch.mean(torch.minimum(torch.abs(recalled_phases - (p + math.pi) % (2*math.pi)),
                                   2*math.pi - torch.abs(recalled_phases - (p + math.pi) % (2*math.pi))))
    return min(d1.item(), d2.item())

print(f"  Target: Pattern {chr(65+target_idx)} ({labels[target_idx]})")
print(f"  Input:  {[round(n.item(), 2) for n in noisy]}")
print(f"  Output: {[round(p.item(), 2) for p in recalled]}")
print(f"\n  Pattern errors:")
for i, (pat, label) in enumerate(zip(patterns, labels)):
    err = pattern_error(recalled, pat)
    marker = ' ← MATCH' if err < 0.5 else ''
    print(f"    {label}: {err:.4f}{marker}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, data, title in zip(axes,
    [noisy.numpy(), [p.item() for p in recalled], patterns[target_idx]],
    ['Noisy Input', 'Recalled Output', f'Target (Pattern {chr(65+target_idx)})']):
    colors = ['#3498db' if abs(p) < 1.5 else '#e74c3c' for p in data]
    ax.bar(range(N), data, color=colors, edgecolor='black', alpha=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Thread'); ax.set_ylabel('Phase')
    ax.grid(True, alpha=0.3)
plt.suptitle('Multi-Pattern Hebbian Memory', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()


# ---
# ## 3. Binary Image Denoising — Pattern Recovery from Noise
# 
# Store a simple binary "image" (encoded as a 2D grid of phases 0/π), corrupt it with random noise, and use Hebbian relaxation to recover the original pattern.
# 
# **Gates:** `shift` → `hebbian` (×iterations) → `saturate` (binary snap)
# 
# **Application:** Error correction, image restoration.

# In[4]:


print("=" * 60)
print("  3. Binary Image Denoising via Hebbian Memory")
print("=" * 60)

# Define a 4x4 binary image (cross pattern)
image = [
    [0, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 0],
]
N = 16  # Flattened

# Convert to phases (0→0, 1→π)
pattern = [row_val * math.pi for row in image for row_val in row]

# Store pattern
W = HebbianGate.store_patterns(N, [pattern])

# Corrupt: flip 25% of pixels
torch.manual_seed(7)
corrupted = torch.tensor(pattern, dtype=torch.float32)
flip_indices = torch.randperm(N)[:4]
for idx in flip_indices:
    corrupted[idx] = math.pi - corrupted[idx]  # Flip 0↔π

# Recover via Hebbian circuit
circ = PhasorCircuit(N, name="ImageDenoise")
for i in range(N):
    circ.shift(i, corrupted[i].item())
circ.barrier()
for _ in range(60):
    circ.hebbian(W, alpha=0.1)
circ.barrier()
circ.saturate(levels=2)  # Snap to binary

result = engine.run(circ)
recovered = result['phases']

# Reshape for visualization
def to_grid(data, size=4):
    arr = np.array([p if isinstance(p, float) else p.item() for p in data])
    grid = (arr > 1.0).astype(int)  # > π/2 → 1, else → 0
    return grid.reshape(size, size)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
grids = [to_grid(pattern), to_grid(corrupted.numpy()), to_grid([p.item() for p in recovered])]
titles = ['Original', f'Corrupted (4 flips)', 'Recovered']

for ax, grid, title in zip(axes, grids, titles):
    ax.imshow(grid, cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(grid[i,j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if grid[i,j] == 1 else 'black')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

match = np.sum(grids[2] == grids[0]) / 16 * 100
plt.suptitle(f'Binary Image Denoising ({match:.0f}% accuracy)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()
print(f"\n  Accuracy: {match:.0f}%")


# ---
# ## 4. Memory Capacity — How Many Patterns Can Be Stored?
# 
# The theoretical capacity of a Hopfield network is ~0.14N patterns. We test this empirically by storing increasing numbers of random patterns and measuring recall accuracy.
# 
# **Gates:** `shift` → `hebbian` (×iterations) → `saturate`

# In[5]:


print("=" * 60)
print("  4. Memory Capacity Test")
print("=" * 60)

N = 20
pattern_counts = [1, 2, 3, 4, 5, 8, 10, 15]
accuracies = []

for n_patterns in pattern_counts:
    # Generate random binary patterns
    torch.manual_seed(42 + n_patterns)
    random_patterns = []
    for _ in range(n_patterns):
        pat = (torch.randint(0, 2, (N,)).float() * math.pi).tolist()
        random_patterns.append(pat)

    W = HebbianGate.store_patterns(N, random_patterns)

    # Test recall of first pattern
    target = random_patterns[0]
    # Add noise
    noisy = torch.tensor(target, dtype=torch.float32)
    noise_idx = torch.randperm(N)[:3]
    for idx in noise_idx:
        noisy[idx] = math.pi - noisy[idx]

    circ = PhasorCircuit(N, name=f"Capacity_{n_patterns}")
    for i in range(N):
        circ.shift(i, noisy[i].item())
    circ.barrier()
    for _ in range(40):
        circ.hebbian(W, alpha=0.1)
    circ.barrier()
    circ.saturate(levels=2)

    result = engine.run(circ)
    recalled = result['phases']

    # Check accuracy
    target_t = torch.tensor(target, dtype=torch.float32)
    correct = 0
    for i in range(N):
        if abs(recalled[i].item() - target_t[i].item()) < 1.0 or            abs(recalled[i].item() - (target_t[i].item() + math.pi) % (2*math.pi)) < 1.0:
            correct += 1
    acc = correct / N
    inv_correct = sum(1 for i in range(N)
                      if abs(recalled[i].item() - (target_t[i].item() + math.pi) % (2*math.pi)) < 1.0)
    acc = max(correct, inv_correct) / N
    accuracies.append(acc)
    print(f"  {n_patterns:2d} patterns: accuracy = {acc*100:.1f}%")

# Capacity limit
theoretical_limit = 0.14 * N
print(f"\n  Theoretical capacity: ~{theoretical_limit:.1f} patterns")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pattern_counts, [a*100 for a in accuracies], 'o-', color='#3498db', lw=2.5, ms=10)
ax.axhline(y=80, color='#e74c3c', linestyle='--', lw=1.5, label='80% threshold')
ax.axvline(x=theoretical_limit, color='#2ecc71', linestyle='--', lw=1.5,
           label=f'Theoretical limit ({theoretical_limit:.1f})')
ax.set_xlabel('Number of Stored Patterns', fontweight='bold')
ax.set_ylabel('Recall Accuracy (%)', fontweight='bold')
ax.set_title(f'Hebbian Memory Capacity (N={N} threads)', fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ---
# ## 5. Hetero-Associative Memory — Input-to-Output Mapping
# 
# Unlike auto-associative memory (input = output pattern), **hetero-associative** memory maps input patterns to different output patterns. We combine Hebbian storage with DFT transformation.
# 
# **Gates:** `encode_phases` → `dft` → `hebbian` → `normalize`
# 
# **Application:** Lookup tables, function approximation, translation memory.

# In[6]:


print("=" * 60)
print("  5. Hetero-Associative Memory: Input → Output Mapping")
print("=" * 60)

N = 6

# Define input-output pairs
# Input:  encoded as phases
# Output: DFT of input (spectral signature)
inputs = [
    [0, 0, 0, math.pi, math.pi, math.pi],     # Pattern I1
    [math.pi, math.pi, 0, 0, math.pi, math.pi], # Pattern I2
]

# Build association using Hebbian learning
# Store input patterns
W = HebbianGate.store_patterns(N, inputs)

# Test with noisy version of I1
noisy_input = [0.2, -0.1, 0.15, math.pi-0.2, math.pi+0.1, math.pi-0.15]

# Circuit: Encode → Hebbian recall → DFT transform
circ = PhasorCircuit(N, name="HeteroAssociative")
for i, p in enumerate(noisy_input):
    circ.shift(i, p)
circ.barrier()
circ.measure("input")

# Hebbian recall to clean up
for _ in range(30):
    circ.hebbian(W, alpha=0.1)
circ.barrier()
circ.normalize()
circ.measure("recalled")

# DFT to produce output
circ.dft()
circ.barrier()
circ.measure("output")

print_circuit_stats(circ)
result = engine.run(circ)

# Visualize
stages = ['input', 'recalled', 'output']
titles = ['Noisy Input', 'Hebbian Recall', 'DFT Output (Association)']
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, stage, title in zip(axes, stages, titles):
    snap = result['measurements'][stage]
    if isinstance(snap, dict):
        phases = snap['phases'].detach().numpy()
        amps = snap['amplitudes'].detach().numpy()

        ax2 = ax.twinx()
        ax.bar(range(N), phases, color='#3498db', alpha=0.6, label='Phase', edgecolor='black')
        ax2.plot(range(N), amps, 'o-', color='#e74c3c', lw=2, label='Amplitude')
        ax.set_xlabel('Thread'); ax.set_ylabel('Phase', color='#3498db')
        ax2.set_ylabel('Amplitude', color='#e74c3c')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)

plt.suptitle('Hetero-Associative Memory Pipeline', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

print("\n  Pipeline: Noisy Input → Hebbian cleanup → DFT transformation → Output")
print("  The DFT produces a unique spectral signature for each stored pattern")


# ---
# ## Summary
# 
# | Example | Mechanism | Key Result |
# |---------|-----------|-----------|
# | Basic Recovery | PhasorFlowMemory | Recover binary pattern from corruption |
# | Multi-Pattern | HebbianGate × 4 patterns | Selective recall by proximity |
# | Image Denoising | Hebbian + Saturate | Recover binary images from noise |
# | Capacity Limits | Varying pattern count | ~0.14N empirical capacity |
# | Hetero-Associative | Hebbian + DFT | Input→output pattern mapping |
# 
# > **Key insight:** Phasor associative memory stores patterns as phase relationships in a Hebbian weight matrix. The system naturally converges to the nearest attractor, enabling robust recall from noisy inputs — a bio-inspired alternative to traditional lookup tables.

# ## 6. Manuscript-Grade Figure Exports
# This section generates publication-quality associative memory figures and writes both PNG and PDF files for manuscript use.

# In[2]:


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import phasorflow as pf
from phasorflow.gates.neuromorphic import HebbianGate

np.random.seed(42)
torch.manual_seed(42)

repo_root = Path.cwd().resolve().parents[1]
figure_dir = repo_root / "manuscript" / "PhasorFlow" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

backend = pf.Simulator.get_backend("analytic_simulator")

def phase_error(phases_a, phases_b):
    d = np.abs(phases_a - phases_b)
    return np.minimum(d, 2 * np.pi - d)

# --- Binary image denoising experiment (4x4 pattern) ---
image = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
])
pattern = (image.flatten() * np.pi).astype(np.float32)
N = len(pattern)

corrupted = pattern.copy()
flip_idx = np.random.choice(N, size=max(1, N // 4), replace=False)
corrupted[flip_idx] = (corrupted[flip_idx] + np.pi) % (2 * np.pi)

W = HebbianGate.store_patterns(N, [pattern.tolist()])
circ = pf.PhasorCircuit(N, name="AssocMemoryDenoise")
for i in range(N):
    circ.shift(i, float(corrupted[i]))
for _ in range(12):
    circ.hebbian(W, alpha=0.1)
circ.saturate(levels=2)
res = backend.run(circ)
recovered = res["phases"].detach().cpu().numpy()

err_corrupted = phase_error(corrupted, pattern)
err_recovered = phase_error(recovered, pattern)
mean_err_before = float(np.mean(err_corrupted))
mean_err_after = float(np.mean(err_recovered))

def to_binary_grid(phases):
    vals = np.cos(phases) < 0
    return vals.astype(float).reshape(4, 4)

# --- Capacity curve experiment ---
pattern_counts = [1, 2, 3, 4, 5, 8, 10, 12]
accuracies = []

for n_patterns in pattern_counts:
    torch.manual_seed(100 + n_patterns)
    random_patterns = []
    for _ in range(n_patterns):
        bits = torch.randint(0, 2, (N,), dtype=torch.float32)
        random_patterns.append((bits * math.pi).tolist())

    Wn = HebbianGate.store_patterns(N, random_patterns)
    target = np.array(random_patterns[0], dtype=np.float32)
    noisy = target.copy()
    noisy_flip = np.random.choice(N, size=max(1, N // 5), replace=False)
    noisy[noisy_flip] = (noisy[noisy_flip] + np.pi) % (2 * np.pi)

    circ = pf.PhasorCircuit(N, name=f"Cap_{n_patterns}")
    for i in range(N):
        circ.shift(i, float(noisy[i]))
    for _ in range(10):
        circ.hebbian(Wn, alpha=0.1)
    circ.saturate(levels=2)
    out = backend.run(circ)
    rec = out["phases"].detach().cpu().numpy()

    same = np.abs(rec - target) < 1.0
    inverse = np.abs(rec - ((target + np.pi) % (2 * np.pi))) < 1.0
    acc = float(np.mean(np.logical_or(same, inverse)))
    accuracies.append(acc)

# --- Multi-panel manuscript figure ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

im0 = axes[0, 0].imshow(to_binary_grid(pattern), cmap="gray_r", vmin=0, vmax=1)
axes[0, 0].set_title("A. Stored Pattern")
axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

axes[0, 1].imshow(to_binary_grid(corrupted), cmap="gray_r", vmin=0, vmax=1)
axes[0, 1].set_title("B. Corrupted Input")
axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

axes[1, 0].imshow(to_binary_grid(recovered), cmap="gray_r", vmin=0, vmax=1)
axes[1, 0].set_title("C. Recovered Output")
axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

axes[1, 1].plot(pattern_counts, [a * 100 for a in accuracies], marker="o", color="#1f77b4", linewidth=2.0)
axes[1, 1].axvline(0.14 * N, color="#e74c3c", linestyle=":", linewidth=1.3, label=f"0.14N={0.14 * N:.1f}")
axes[1, 1].set_title("D. Empirical Memory Capacity")
axes[1, 1].set_xlabel("Stored Pattern Count")
axes[1, 1].set_ylabel("Recall Accuracy (%)")
axes[1, 1].set_ylim(0, 105)
axes[1, 1].legend(loc="upper right")

fig.suptitle(
    f"Associative Memory Results | Mean phase error: {mean_err_before:.3f} -> {mean_err_after:.3f} rad",
    fontsize=11,
    y=1.02
)

assoc_png = figure_dir / "associative_memory_multi_panel.png"
assoc_pdf = figure_dir / "associative_memory_multi_panel.pdf"
fig.savefig(assoc_png, bbox_inches="tight")
fig.savefig(assoc_pdf, bbox_inches="tight")
plt.show()

# Supplementary error distribution plot.
fig2, ax2 = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
ax2.hist(err_corrupted, bins=10, alpha=0.7, color="#d62728", label="Before Recall")
ax2.hist(err_recovered, bins=10, alpha=0.7, color="#2ca02c", label="After Recall")
ax2.set_title("Per-Thread Phase Error Distribution")
ax2.set_xlabel("Absolute Circular Error (rad)")
ax2.set_ylabel("Count")
ax2.legend(loc="upper right")
err_png = figure_dir / "associative_memory_error_distribution.png"
err_pdf = figure_dir / "associative_memory_error_distribution.pdf"
fig2.savefig(err_png, bbox_inches="tight")
fig2.savefig(err_pdf, bbox_inches="tight")
plt.show()

print(f"Saved associative-memory multi-panel: {assoc_png}")
print(f"Saved associative-memory vector figure: {assoc_pdf}")
print(f"Saved error-distribution figure: {err_png}")
print(f"Saved error-distribution vector figure: {err_pdf}")

