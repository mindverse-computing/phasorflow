<!--
(c) 2026 Mindverse Computing LLC.
Licensed under CC BY-NC 4.0.
See LICENSE file for patent and commercial restrictions.
-->

# Gates & Operations

Computation in PhasorFlow operates strictly through sequential applications of continuous complex unitary matrices mapping onto $N$-dimensional state spaces.

Because vectors are physically bounded to the **Unit Circle** ($|x|^2 = 1.0$), non-linear combinations act as analog inference mechanisms capable of encoding features and dynamically mixing sequences globally.

---

## 1. Shift Gate ($S_{\phi}$)
The **ShiftGate** applies a precise scalar rotation $\phi \in \mathbb{R}$ to a specified Phase element over exactly one logical computational unit.

```python
circuit.shift(thread=0, angle=1.2)
```

Mathematically, it operates as the complex identity matrix by injecting the rotation strictly at index $j$ on the diagonal:

$$
S_{\phi, j} = 
\begin{bmatrix}
1 & 0 & \dots \\
0 & e^{i\phi_j} & \dots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

In AI environments like **Variational Phasor Circuits**, the `Shift` gate acts as the parameterized Multi-Layer Perceptron (MLP). Training Phase boundaries involves altering the angles to dynamically steer physical alignment outcomes.

---

## 2. Mix Gate (Topological Coupling)
The **MixGate** is a two-particle structural connection. It establishes a deterministic topological projection between any two uncoupled thread geometries.

```python
# Symmetrical structural convolution between adjacent nodes
circuit.mix(thread_1=0, thread_2=1)
```

The operator linearly sums the states representing an unparameterized normalized feature exchange mechanism:

$$
M(j, k) = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & \dots & i \\
\vdots & \ddots & \vdots \\
i & \dots & 1
\end{bmatrix}
$$

*(Used heavily establishing recurrent topological graphs mirroring spatial networks, seen in `examples/ex_06_finance_volatility.py`)*

---

## 3. Global DFT Mixing (Self-Attention Alternative)
The crowning mathematical capability of PhasorFlow derives from accessing $N$-Thread phase coherence via the **Discrete Fourier Transform (DFT) Gate**. 

```python
circuit.dft() # O(N Log N) Global Sequence Evaluation
```

Rather than iterating slow pairwise layer connections, the DFT utilizes an unparameterized global $N \times N$ unitary matrix transformation across all threads simultaneously, mixing phases through the discrete Fourier basis:

$$
F_N = \frac{1}{\sqrt{N}} \begin{pmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^2 & \cdots & \omega^{N-1} \\
1 & \omega^2 & \omega^4 & \cdots & \omega^{2(N-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{N-1} & \omega^{2(N-1)} & \cdots & \omega^{(N-1)^2}
\end{pmatrix}
$$

where $\omega = e^{-2\pi i / N}$ is the $N$-th root of unity. The action on the state vector structurally mixes every spatial domain feature into the frequency domain deterministically:

$$
z_k' = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} z_n \, \omega^{kn}, \quad k = 0, 1, \ldots, N-1.
$$

In **Phasor Transformers**, executing the `.dft()` operator natively replaces the hyper-dense $Q K^T V$ weight requirements of traditional Transformer logic, mapping temporal embeddings instantaneously in fractions of the computational footprint!

---

## 4. Discontinuous Constraints
While unitary interference operates continuously, PhasorFlow natively supports physical constraint triggers analogous to discontinuous activation functions:

*   **Saturate Gates**: `circuit.saturate(thread, levels=2)` mathematically "snaps" a continuous phase toward pre-defined discrete angular bins (e.g. exactly `0` or `1.57`). It provides the error-correction foundation necessary for Holographic Memory (`ex_05_associative_memory.py`).
*   **Threshold Gates**: `circuit.threshold(thread, limit=0.5)` zeroes out flow computationally if localized coherence drops beneath a numerical boundary.

---

## 5. Complete Gate Library

PhasorFlow ships with a comprehensive library of 22 primitive gates that organically compose into fully functional algorithmic and machine learning pipelines.

| Category | Gate Name | Operation / Function Description |
| :--- | :--- | :--- |
| **Standard Unitary** | Shift | Phase rotation proportional to input value ($z \mapsto z \cdot e^{i\theta}$) |
| | Invert | Phase flip by $\pi$ radians ($z \mapsto -z$) |
| | Mix | Two-thread interference (beam splitter) |
| | DFT | Global sequence token mixing via Discrete Fourier Transform |
| | Permute | Reordering of computing thread state indices |
| | Reverse | Time-reversal via global complex conjugation ($z \mapsto z^*$) |
| | Accumulate | Cumulative complex wave summation ($z_{n+1} = z_{n+1} + z_n$) |
| | GridPropagate | Wavefront propagation accumulation across a 2D lattice |
| **Non-Linear** | Threshold | Filters low-magnitude phasors and forces outputs to zero |
| | Saturate | Quantizes phase geometry toward discrete binary anchors |
| | Normalize | Pulls any generic $\mathbb{C}^N$ state rigidly back to the $\mathbb{T}^N$ unit circle |
| | LogCompress | Logarithmic amplitude compression ($\mu$-law analog) |
| | CrossCorrelate | Evaluates phase coherence between discrete pattern sequences |
| | Convolve | Sliding continuous spatial convolution along threads |
| **Neuromorphic** | Kuramoto | Global phase synchronization towards a mean alignment field |
| | Hebbian | Associative memory adaptation via nearest-neighbor phase pull |
| | Ising | Anti-ferromagnetic coupling driving bi-modal ($0, \pi$) symmetry |
| | Synaptic | Continuous drag/coupling between targeted neural oscillators |
| | AsymmetricCouple | Non-reciprocal directed phase influence across nodes |
| **Encoding** | EncodePhase | Maps real-valued $x_i$ into the spatial phase domain $[0, 2\pi)$ |
| | EncodeAmplitude | Maps scalar magnitudes physically onto the wave norm |

## 6. Brief Overview of Remaining Operations

### Standard Unitary Operations
- **Invert**: The Invert gate is a special case of the Shift gate with $\phi = \pi$, reflecting the phasor across the origin.
- **Permute**: Reorders thread indices natively without breaking continuous wave topologies.
- **Reverse**: Executes a time-reversal operator by globally conjugating the complex state vector across all $N$ threads.
- **Accumulate**: Performs a local cumulative complex summation sweeping sequentially across adjacent threads.
- **GridPropagate**: Simulates wavefront propagation across a lattice topology, enabling native unit-circle evaluation of localized connectivity graphs.

### Additional Non-Linear Operations
- **Normalize (PullBack)**: Continuously enforces unit-magnitude topology $|z|=1$ iteratively during complex cascade flows.
- **LogCompress**: Attenuates signal extrema for dynamic range adjustment.
- **CrossCorrelate** and **Convolve**: Execute complex sequence sliding operations directly in the spatial phase domain.

### Neuromorphic Operations
- **Kuramoto Gate**: Implements global phase synchronization across continuous populations.
- **Hebbian Gate**: Modifies outer-product associative phase links to store multi-pattern oscillator memories.
- **Ising Gate**: A discrete coupling operator that drives threads to strictly bipartite consensus arrays.
- **Synaptic** & **AsymmetricCouple Gates**: Enact directed phase momentum transfer between distinct computational reservoirs.

### Encoding Operations

- **EncodePhase** & **EncodeAmplitude**: Comprise the native interface for loading external real-valued data structures geometrically onto the $N$-Torus ($\mathbb{T}^N$) manifold or into full continuous $\mathbb{C}^N$ amplitudes.

---

**© 2026 Mindverse Computing LLC.**  
Licensed under CC BY-NC 4.0.  
See LICENSE file for patent and commercial restrictions.
