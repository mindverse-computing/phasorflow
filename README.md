
# PhasorFlow 🌀

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19044565.svg)](https://doi.org/10.5281/zenodo.19044565)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

**PhasorFlow** is a high-performance Python library for **Unit Circle (Phasor) based Computing**. Built on PyTorch, it provides a complete framework for building, training, and deploying machine learning models that operate entirely on the unit circle through continuous phase interference.

> **Academic & Research Notice:** PhasorFlow is released under the **CC BY-NC 4.0** license. Commercial use is strictly prohibited. See the [LICENSE](LICENSE) file for details regarding patent and trademark reservations.

---

## 🚀 Key Features

| Category | Capabilities |
| :--- | :--- |
| **22 Phasor Gates** | **Standard:** Shift, Mix, DFT, Permute, Reverse, Accumulate, GridPropagate <br> **Non-linear:** Threshold, Saturate, Normalize, PullBack, LogCompress, CrossCorrelate, Convolve <br> **Neuromorphic:** Synaptic, Kuramoto, Hebbian, Ising, AsymmetricCouple <br> **Encoding:** EncodePhase, EncodeAmplitude |
| **PhasorCircuit API** | Fluent, Qiskit-style circuit construction with full introspection and state visualization. |
| **AnalyticEngine** | High-performance PyTorch simulator with full autograd support for variational optimization. |
| **Experimental ML** | **VPC** (Classifier), **PhasorTransformer** (Sequence Prediction), and **PhasorGAN** (Generative Modeling). |
| **Neuromorphic** | LIP integration layers and associative oscillator memory modules. |
| **Visualization** | Native support for text-based and Matplotlib-based circuit diagrams. |

---

## 🛠 Installation

```bash
# Clone the repository
git clone [https://github.com/mindverse-computing/phasorflow.git](https://github.com/mindverse-computing/phasorflow.git)
cd phasorflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

---

## ⚡ Quick Start

```python
import math
from PhasorFlow import PhasorCircuit
from PhasorFlow.engine.analytic import AnalyticEngine

# Create a circuit with 2 oscillator threads
pc = PhasorCircuit(2)
pc.shift(0, math.pi)   # Shift thread 0 by π
pc.mix(0, 1)            # Interference gate (analog of Hadamard)

# Run via the PyTorch backend
engine = AnalyticEngine()
result = engine.run(pc)

print(f"State: {result['state_vector']}")
print(f"Phases: {result['phases']}")

```

---

## 🧠 Model Zoo

### VPC — Variational Phasor Circuit Classifier

The VPC architecture maps input features to phases and optimizes circuit parameters for classification tasks.

```python
from PhasorFlow.models import VPC

model = VPC(num_features=12, num_layers=2)
model.fit(X_train, y_train, epochs=50, lr=0.1)
accuracy = model.score(X_test, y_test)

```

### PhasorTransformer — FNet-Style Sequence Prediction

Utilizes Discrete Fourier Transform (DFT) gates to perform mixing, offering a phasor-based alternative to traditional attention.

```python
from PhasorFlow.models import PhasorTransformer

model = PhasorTransformer(seq_length=10, num_blocks=2)
model.fit(X_train, y_train, epochs=30, lr=0.05)
future = model.predict_autoregressive(context, horizon=20)

```

---

## 📓 Research Notebooks

| Index | Notebook | Focus Area |
| --- | --- | --- |
| 1 | `1-Circuits.ipynb` | Core operations & gate catalog |
| 2.2 | `2.2-Shor's-Algorithm.ipynb` | Period finding via phasor circuits |
| 2.4 | `2.4-Associative-Memory.ipynb` | Oscillator-based associative memory |
| 3.1 | `3.1-VPC-Single.ipynb` | VPC first principles |
| 4.4 | `4.4-LPM.ipynb` | Large Phasor Model (Autoregressive) |
| 5.1 | `5.1-Phasor-GAN.ipynb` | Generative Adversarial Networks |

*For a full list of all 20+ documentation notebooks, see the [notebooks/](https://www.google.com/search?q=notebooks/) directory.*

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
