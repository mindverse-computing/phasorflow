
# PhasorFlow 🌀

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19044565.svg)](https://doi.org/10.5281/zenodo.19044565)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

**PhasorFlow** is a high-performance Python library for **Unit Circle (Phasor) based Computing**. Built on PyTorch, it provides a complete framework for building, training, and deploying machine learning models that operate entirely on the unit circle through continuous phase interference.

> **Academic & Research Notice:** PhasorFlow is released under the **CC BY-NC 4.0** license. Commercial use is strictly prohibited. See the [LICENSE](LICENSE) file for details regarding patent and trademark reservations.

---

## 📚 Reference Manuscripts

This repository is the single reference implementation for three companion
manuscripts, each covering a different layer of unit-circle computing. All
results, models, and examples in those papers are reproducible from this
codebase.

| Manuscript | Focus | Implementation |
| --- | --- | --- |
| **PhasorFlow: A Python Library for Unit Circle Based Computing** | The core framework — phasor circuits, the 22-gate library, and the analytic PyTorch backend | the library as a whole (`circuit.py`, `gates/`, `engine/`) |
| **Variational Phasor Circuits for Phase-Native Brain–Computer Interface Classification** | The VPC classifier — trainable phase shifts + unitary mixing as a parameter-efficient alternative to dense nets, benchmarked on real motor-imagery EEG | `phasorflow.VPC` (`models/vpc.py`), `examples/ex_07_vpc_single.py`, `ex_09_real_eeg_vpc.py` |
| **The Phasor Transformer: Resolving Attention Bottlenecks on the Unit Circle** | The Phasor Transformer block and the Large Phasor Model (LPM) — DFT token mixing at $\mathcal{O}(N\log N)$ in place of dot-product attention | `phasorflow.PhasorTransformer` (`models/transformer.py`), `examples/ex_08_phasor_transformer.py`, `ex_10_depth_scaling.py` |

See [How to Cite](#-how-to-cite) for the BibTeX of each.

---

## 🚀 Key Features and Gate Capabilities

A `PhasorCircuit` evaluates logic mathematically using discrete chronological combinations of unitary bounded complex operators. PhasorFlow directly exposes 22 highly-optimized native Gates covering everything from deep learning parameters to neuromorphic simulation out of the box.

### General & Topologies
- **Shift (`S`)**: Applies a localized scalar rotation $\phi$ to a single feature-thread. Serves correspondingly as parameterized feed-forward layers in Variational circuits.
- **Mix**: Implements a symmetrical topological entanglement boundary coupling adjacent dimension threads into unified representations.
- **DFT**: Applies the unparameterized global Discrete Fourier Transform natively to the full tensor space in $O(T\log T)$, mixing multi-dimensional sequences frictionlessly.
- **CrossCorrelate / Convolve**: Executes sliding spatial convolutions directly in the complex phase domain for pattern matching.

### Discontinuous Boundaries
Unlike fully continuous quantum operators, classical simulation natively supports evaluating discontinuous sub-states necessary for holographic memory binding.
- **Threshold**: Mathematically zeroes specific state propagation if numerical coherence falls below user-designated boundaries.
- **Saturate**: Immediately forces thread phases geometrically into fixed angular sub-bins (e.g. $[0, \pi]$).
- **Ising**: Drives bi-modal $(0, \pi)$ symmetry for graph partitioning and optimization.

### Neuromorphic Sub-Rhythms
PhasorFlow rigorously reproduces continuous differential equation physics directly via phase representations.
- **Kuramoto**: Global uniform phase-coupling imitating biological macroscopic coherence synchronization.
- **LIP-Layer**: *Leaky-Integrate-and-Phase* dynamics mirroring binary neural spiking arrays into continuous rhythmic flows.
- **Oscillatory Associative Memory (Hebbian)**: Structural Hopfield algorithms utilizing uncoupled Hebbian sum rules to generate robust fault-tolerant geometric attractors natively.
- **Synaptic Coupling**: Directed continuous phase momentum transfer between distinct computational reservoirs.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/mindverse-computing/phasorflow.git
cd phasorflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

---

## ⚡ Quick Start

```python
import math
import phasorflow as pf

# Create a circuit with 2 oscillator threads
circuit = pf.PhasorCircuit(2)
circuit.shift(0, math.pi)   # Parameter Rotation on thread 0 by π
circuit.mix(0, 1)            # Topological Interference gate 

# Run via the PyTorch analytic backend
engine = pf.Simulator.get_backend('analytic_simulator')
result = engine.run(circuit)

print(f"State Vector: {result['state_vector']}")
print(f"Output Angles (rad): {result['phases']}")
```

---

## 🧠 Model Zoo Capabilities

### VPC — Variational Phasor Circuit Classifier
*Reference: the VPC manuscript.* The VPC architecture statically maps feature vectors into physical initial conditions and dynamically optimizes sequential `Shift` operators globally via gradient methods against categorical targets—evaluating complex separating structures with just dozens of weights instead of thousands. On real PhysioNet motor-imagery EEG it reaches a mean decoding accuracy of $\approx 0.60$—the best of the standard BCI baselines (LDA, logistic regression, RBF-SVM, MLP)—with an order of magnitude fewer parameters and the lowest cross-subject variance. Its capacity is characterized honestly: phase-only shifts with unitary mixing realize a linear decision function in a fixed cosine/sine lifting, well matched to separable band-power structure but unable to represent parity-type functions—a ceiling depth does not raise. Available as `phasorflow.VPC`.

### PhasorTransformer / LPM — Continuous Sequence Architectures
*Reference: the Phasor Transformer (LPM) manuscript.* Each Phasor Transformer block pairs lightweight trainable phase-shifts with a parameter-free Discrete Fourier Transform (`.dft()`) for token coupling, giving global $\mathcal{O}(N\log N)$ mixing without dot-product attention maps ($Q K^T V$)—extending Google's FNet in the unit-circle setting. Stacking these blocks defines the **Large Phasor Model (LPM)** for autoregressive sequence prediction: it beats a zero-parameter persistence baseline and, with the corrected gradient path, improves monotonically with depth before saturating, while remaining competitive-but-not-superior to self-attention at a fraction of the parameter count. Available as `phasorflow.PhasorTransformer`.

---

## 📓 Examples & Validation

PhasorFlow ships with ten runnable example scripts in `examples/` that exercise
every capability — from base circuit construction to the DFT-based Phasor
Transformer — plus a `pytest` suite in `tests/` covering gate unitarity, readout
gradient finiteness, and reference/vectorized-engine equivalence.

| Example | Focus Area | Extra deps |
| --- | --- | --- |
| `ex_01_circuits.py` | Circuit construction, gates, and visualization | — |
| `ex_02_circuit_operations.py` | Circuit operations and algorithmic (DSA) applications | — |
| `ex_03_shors_algorithm.py` | Classical phasor extraction of Shor's period-finding physics | `qiskit` (`.[qiskit]`) |
| `ex_04_neural_binding.py` | LIP-layer and Kuramoto phase-synchronization binding | — |
| `ex_05_associative_memory.py` | Oscillatory (Hopfield-phase) associative memory | — |
| `ex_06_finance_volatility.py` | OHLCV phase-coherence volatility detection | — |
| `ex_07_vpc_single.py` | Single-block Variational Phasor Circuit classifier | — |
| `ex_08_phasor_transformer.py` | FNet-style Phasor Transformer sequence model | — |
| `ex_09_real_eeg_vpc.py` | Real PhysioNet EEG motor-imagery VPC benchmark | `mne`, `scikit-learn` (`.[eeg]`) |
| `ex_10_depth_scaling.py` | VPC capacity ceiling and transformer depth scaling | `scikit-learn` (`.[benchmarks]`) |

```bash
# Run any example from the repository root
python examples/ex_01_circuits.py

# Run the test suite
pip install -e .[test]
pytest -q
```

---

## 📑 How to Cite

If you use **PhasorFlow** in your research, please cite the software. If you use
a specific model, please also cite the corresponding manuscript.

### Software

```bibtex
@software{sigdel_2026_phasorflow,
  author       = {Sigdel, Dibakar and Panday, Namuna},
  title        = {PhasorFlow: A Python Library for Unit Circle Based Computing},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.3.0},
  doi          = {10.5281/zenodo.19044565},
  url          = {https://doi.org/10.5281/zenodo.19044565}
}
```

APA — Sigdel, D., & Panday, N. (2026). *PhasorFlow: A Python Library for Unit Circle Based Computing* (Version v0.3.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19044565

### Manuscripts

The three companion manuscripts backed by this codebase (see
[Reference Manuscripts](#-reference-manuscripts)):

```bibtex
@article{sigdel_phasorflow_framework,
  author  = {Sigdel, Dibakar and Panday, Namuna},
  title   = {PhasorFlow: A Python Library for Unit Circle Based Computing},
  year    = {2026}
}

@article{sigdel_vpc,
  author  = {Sigdel, Dibakar},
  title   = {Variational Phasor Circuits for Phase-Native
             Brain--Computer Interface Classification},
  year    = {2026}
}

@article{sigdel_lpm,
  author  = {Sigdel, Dibakar},
  title   = {The Phasor Transformer: Resolving Attention Bottlenecks
             on the Unit Circle},
  year    = {2026}
}
```

---

## ⚖️ License

**Copyright (c) 2024-2026 Mindverse Computing LLC**

PhasorFlow is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

* **Attribution:** You must give appropriate credit and provide a link to the license.
* **Non-Commercial:** You may not use the material for commercial purposes.
* **No Patent Rights:** This license pertains strictly to copyright. No patent rights are granted, implied, or transferred.

---

**Contact:** [Mindverse Computing](https://github.com/mindverse-computing)
