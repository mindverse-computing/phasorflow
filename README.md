# PhasorFlow 🌀

**PhasorFlow** is a high-performance Python library for **Unit Circle (U1) Phasor-based Computing**. Built on PyTorch, it provides a complete framework for building, training, and deploying machine learning models that operate entirely on the unit circle through continuous phase interference.

> PhasorFlow is released for academic and research purposes under the **CC BY-NC 4.0** license.
> Commercial use is strictly prohibited. See the [LICENSE](LICENSE) file for details regarding patent and trademark reservations.

---

## Key Features

| Category | Capabilities |
|---|---|
| **22 Phasor Gates** | Standard (Shift, Mix, DFT, Permute, Reverse, Accumulate, GridPropagate), Non-linear (Threshold, Saturate, Normalize, PullBack, LogCompress, CrossCorrelate, Convolve), Neuromorphic (Synaptic, Kuramoto, Hebbian, Ising, AsymmetricCouple), Encoding (EncodePhase, EncodeAmplitude) |
| **PhasorCircuit API** | Fluent, Qiskit-style circuit construction with full introspection |
| **AnalyticEngine** | Pure-PyTorch simulator with autograd support for variational optimization |
| **ML Models** | VPC (classifier), PhasorTransformer (sequence prediction), PhasorGAN (timeseries generation) |
| **Neuromorphic** | LIP integration layers, associative oscillator memory |
| **Visualization** | Text-based and Matplotlib circuit drawing |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mindverse-computing/phasorflow.git
cd phasorflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

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

## Models

### VPC — Variational Phasor Circuit Classifier

```python
from PhasorFlow.models import VPC

model = VPC(num_features=12, num_layers=2)
model.fit(X_train, y_train, epochs=50, lr=0.1)
accuracy = model.score(X_test, y_test)
```

Supports binary & multi-class classification, single & multi-stack architectures with pullback/threshold inter-stack operations.

### PhasorTransformer — FNet-Style Sequence Prediction

```python
from PhasorFlow.models import PhasorTransformer

model = PhasorTransformer(seq_length=10, num_blocks=2)
model.fit(X_train, y_train, epochs=30, lr=0.05)
future = model.predict_autoregressive(context, horizon=20)
```

Supports single-circuit and stacked modes with optional readout smoothing layer.

### PhasorGAN — Generative Adversarial Network

```python
from PhasorFlow.models import PhasorGAN

gan = PhasorGAN(seq_length=8, num_layers_g=2, num_layers_d=2)
history = gan.fit(X_real, epochs=60, batch_size=10)
fake_samples = gan.generate(num_samples=10)
```

Both Generator and Discriminator are phasor circuits trained adversarially.

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 1 | `1-Circuits.ipynb` | Core circuit operations and gate catalog |
| 2.1 | `2.1-Circuit-Operations.ipynb` | Advanced circuit composition |
| 2.2 | `2.2-Shor's-Algorithm.ipynb` | Period finding via phasor circuits |
| 2.3 | `2.3-Neural-Binding.ipynb` | Phase synchronization for binding |
| 2.4 | `2.4-Associative-Memory.ipynb` | Oscillator-based associative memory |
| 2.5 | `2.5-Finance-Volatility-Phasor.ipynb` | Financial volatility modeling |
| 3.1 | `3.1-VPC-Single.ipynb` | VPC first principles — single stack |
| 3.2 | `3.2-VPC-Stacking-Regular.ipynb` | Multi-stack VPC |
| 3.3 | `3.3-VPC-Stacking-PullBack.ipynb` | VPC with pullback inter-stack |
| 3.4 | `3.4-VPC-Stacking-NonLinear.ipynb` | VPC with threshold gate |
| 3.5 | `3.5-BCI-BrainFlow-Binary.ipynb` | BCI binary classification |
| 3.6 | `3.6-BCI-BrainFlow-MultiClass.ipynb` | BCI multi-class classification |
| 3.7 | `3.7-VPC-Direct.ipynb` | VPC model class API demo |
| 4.1 | `4.1-Phasor-Transformer-Regular.ipynb` | FNet transformer first principles |
| 4.2 | `4.2-Phasor-Transformer-NonLinear.ipynb` | Transformer with threshold gate |
| 4.3 | `4.3-Transformer-Benchmarking.ipynb` | Phasor vs PyTorch transformer |
| 4.4 | `4.4-LPM.ipynb` | Large Phasor Model (autoregressive) |
| 4.5 | `4.5-Transformer-Direct.ipynb` | PhasorTransformer model API demo |
| 5.1 | `5.1-Phasor-GAN.ipynb` | PhasorGAN model API demo |
| 5.2 | `5.2-Phasor-GAN-First-Principle.ipynb` | GAN from raw PhasorCircuit |

---

## How to Cite

If you use **PhasorFlow** in your research, please cite our manuscript:

### BibTeX

```bibtex
@article{sigdel2026phasorflow,
  title={PhasorFlow: A Unitary Computing Framework for Resonant Algorithmic Acceleration},
  author={Sigdel, Dibakar},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/mindverse-computing/phasorflow}
}
```

### APA

Sigdel, D. (2026). *PhasorFlow: A Unitary Computing Framework for Resonant Algorithmic Acceleration*. Mindverse Computing LLC. Retrieved from [https://github.com/mindverse-computing/phasorflow](https://github.com/mindverse-computing/phasorflow)

---

## License

**Copyright (c) 2024-2026 Mindverse Computing LLC**

PhasorFlow is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. This license pertains strictly to the copyright of the source code and associated documentation. **No patent rights are granted, implied, or transferred herein.** See the [LICENSE](LICENSE) file for full details.
