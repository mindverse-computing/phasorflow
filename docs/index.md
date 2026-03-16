<!--
(c) 2026 Mindverse Computing LLC.
Licensed under CC BY-NC 4.0.
See LICENSE file for patent and commercial restrictions.
-->

# Welcome to PhasorFlow

**PhasorFlow** is an open-source Python framework that pioneers **Phase-Based Unit Circle Computing**. 

Unlike standard digital computers that process scale-variant numerical amplitudes (e.g., Euclidean dense layers) or quantum computers reliant on delicate probabilistic wave-function collapses, PhasorFlow evaluates logic deterministically across non-linear complex angles on the unit circle ($\mathbb{T}^N$)—running purely on classical hardware while maintaining the unitary mathematics!

---

## The Core Concept

Information in PhasorFlow is represented purely as a **Phasor**—a complex scalar vector of magnitude $1.0$ rotating in a 2D plane:

$$ \psi = e^{i\phi} $$

By stripping away numerical amplitude scaling ($|x| = 1.0$) and restricting computational operations explicitly to angular phase changes $\phi \in [-\pi, \pi]$, PhasorFlow unlocks an entirely new family of ultra-lightweight structural architectures:

*   **Continuous Machine Learning**: Utilizing the **Variational Phasor Circuit (VPC)**, PhasorFlow trains complex continuous feature classification natively, utilizing 1 to 2 orders of magnitude fewer parameters than standard Deep Neural Networks.
*   **Sequence Forecasting (The Phasor Transformer)**: Replaces massive data-dependent Query-Key Attention mechanics ($Q K^T V$) with parameter-free **Discrete Fourier Transform (DFT)** token mixing—yielding near identical predictive capabilities to standard self-attention mechanisms with exponentially less overhead.
*   **Neuromorphic Computing**: Native mapping of Hopfield Associative Memories and biological Leaky-Integrate-and-Phase (LIP) Neural Binding algorithms through the physics of Kuramoto coupled oscillatory synchronization.
*   **Deterministic Structural Algorithms**: Identically executes the complex wave-interference logic of prime-factoring circuits (e.g. Shor's Period Finding) completely classically.

---

## Installation

PhasorFlow is lightweight, mathematically mathematically rigorous, and relies inherently on highly-optimized matrix simulations driven by `numpy` and `scipy`.

```bash
git clone https://github.com/mindverse-computing/phasorflow.git
cd phasorflow
pip install -e .
```

*Proceed to [Getting Started](getting_started.md) to build your first Phasor Circuit!*


---

**© 2026 Mindverse Computing LLC.**  
Licensed under CC BY-NC 4.0.  
See LICENSE file for patent and commercial restrictions.
