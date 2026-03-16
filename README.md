
# PhasorFlow 🌀

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19044565.svg)](https://doi.org/10.5281/zenodo.19044565)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

**PhasorFlow** is a high-performance Python library for **Unit Circle (Phasor) based Computing**. Built on PyTorch, it provides a complete framework for building, training, and deploying machine learning models that operate entirely on the unit circle through continuous phase interference.

> **Academic & Research Notice:** PhasorFlow is released under the **CC BY-NC 4.0** license. Commercial use is strictly prohibited. See the [LICENSE](LICENSE) file for details regarding patent and trademark reservations.

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
The VPC architecture statically maps feature vectors into physical initial conditions and dynamically optimizes sequential `Shift` operators globally via gradient methods against categorical targets—evaluating complex separating structures with just dozens of weights instead of thousands.

### PhasorTransformer — Continuous Sequence Architectures
Extends Google's FNet theory by abandoning multi-head attention $Q K^T V$ weight projection entirely in favor of unparameterized sequence token mixing (`.dft()`) on the unit circle. Emulates classical autoregressive predictive transformers physically.

---

## 📓 Research Notebooks & Validation

PhasorFlow ships with rigorous mathematically validated `Jupyter` notebooks proving every theoretical capability spanning algorithm equivalents identically matching Qiskit to complete Hopfield Neural Denoising tasks.

| Section | Notebook | Focus Area |
| --- | --- | --- |
| 1 | `1-Circuits.ipynb` | Base architecture & visualization validation |
| 2.2 | `2.2-Shor's-Algorithm.ipynb` | Deterministic classical extraction of Shor's quantum period physics |
| 2.3 | `2.3-Neural-Binding.ipynb` | Validation of LIP Layer and Kuramoto binding physics |
| 2.4 | `2.4-Associative-Memory.ipynb` | Convergence properties of Holographic Multi-Pattern Phase Storage |
| 2.5 | `2.5-Finance-Volatility-Phasor.ipynb` | Unsupervised OHLCV Phase Coherence charting anomaly detection |
| 3.1 | `3.1-VPC-Single.ipynb` | Gradient evaluation limits of minimal continuous classification models |
| 4.1 | `4.1-Phasor-Transformer.ipynb` | Regressive mapping of $T$-temporal continuous sinusoidal windows |

*Pre-generated python execution configurations exist for all capabilities in `/phasorflow/examples/...`*

---

## 📑 How to Cite

If you use **PhasorFlow** in your research, please cite the software and the corresponding manuscript:

### BibTeX

```bibtex
@software{sigdel_2026_phasorflow,
  author       = {Sigdel, Dibakar and Panday, Namuna},
  title        = {PhasorFlow: A Python Library for Unit Circle Based Computing},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.19044565},
  url          = {[https://doi.org/10.5281/zenodo.19044565](https://doi.org/10.5281/zenodo.19044565)}
}

```

### APA

Sigdel, D., & Panday, N. (2026). *PhasorFlow: A Python Library for Unit Circle Based Computing* (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19044565

---

## ⚖️ License

**Copyright (c) 2024-2026 Mindverse Computing LLC**

PhasorFlow is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

* **Attribution:** You must give appropriate credit and provide a link to the license.
* **Non-Commercial:** You may not use the material for commercial purposes.
* **No Patent Rights:** This license pertains strictly to copyright. No patent rights are granted, implied, or transferred.

---

**Contact:** [Mindverse Computing](https://www.google.com/search?q=https://github.com/mindverse-computing)
