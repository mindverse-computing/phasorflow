
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

If you use **PhasorFlow** in your research, please cite the software and the corresponding manuscript:

### BibTeX

```bibtex
@software{sigdel_2026_phasorflow,
  author       = {Sigdel, Dibakar and Panday, Namuna},
  title        = {PhasorFlow: A Python Library for Unit Circle Based Computing},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.3.0},
  doi          = {10.5281/zenodo.19044565},
  url          = {[https://doi.org/10.5281/zenodo.19044565](https://doi.org/10.5281/zenodo.19044565)}
}

```

### APA

Sigdel, D., & Panday, N. (2026). *PhasorFlow: A Python Library for Unit Circle Based Computing* (Version v0.3.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19044565

---

## ⚖️ License

**Copyright (c) 2024-2026 Mindverse Computing LLC**

PhasorFlow is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

* **Attribution:** You must give appropriate credit and provide a link to the license.
* **Non-Commercial:** You may not use the material for commercial purposes.
* **No Patent Rights:** This license pertains strictly to copyright. No patent rights are granted, implied, or transferred.

---

**Contact:** [Mindverse Computing](https://github.com/mindverse-computing)
