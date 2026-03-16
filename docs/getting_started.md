<!--
(c) 2026 Mindverse Computing LLC.
Licensed under CC BY-NC 4.0.
See LICENSE file for patent and commercial restrictions.
-->

# Getting Started

In PhasorFlow, the structure of your computation is defined by a `PhasorCircuit`. A circuit allocates a network of $N$ uncoupled "threads," where each thread tracks a localized Phase vector mapped to the Unit Circle.

---

## 1. Initializing the Virtual Circuit

To begin, you initialize a mathematical network. Let's create a minimal network of 3 processing threads:

```python
import phasorflow as pf
import numpy as np

# Create a 3-thread Continuous Unit Circle register
circuit = pf.PhasorCircuit(num_threads=3, name="Demo_Circuit")
```
*(See `examples/ex_01_circuits.py` for full executable implementations!)*

## 2. Applying Operations (Gates)

You map information and entanglement into the circuit by sequentially appending unitary mathematical `Gates`. Every thread starts initialized at exactly $\phi = 0.0$ radians.

```python
# 1. Apply a continuous static phase rotation to Thread 0
circuit.shift(0, 0.79) # ~ PI/4

# 2. Apply a shift to Thread 1
circuit.shift(1, 1.57) # ~ PI/2

# 3. Entangle Thread 0 and Thread 1 structurally
circuit.mix(0, 1)

# 4. Apply a global Discrete Fourier Transform mixing ALL 3 Threads
circuit.dft()
```

## 3. Visualization

You can strictly visualize the chronological sequence of structural operations applied using the native ASCII `Terminal Drawer`. PhasorFlow mimics classical and quantum circuit diagram geometries:

```python
pf.draw(circuit, mode='text')
```

**Output:**
```text
┌─ Circuit Diagram ─┐
T0: ──[S(0.79)]───────┬─────────┬────┤
                    [MIX]     [DFT]  
T1: ─────────────[S(1.57)]────┴─────────┼────┤
                                        │    
T2: ────────────────────────────────────┴────┤
```

## 4. Execution via Backends

A pure `PhasorCircuit` merely strings together symbolic matrix template classes. To actually evaluate the math into explicit physical phase boundaries, you must pass the Circuit into a backend Simulator engine. 

```python
# Initialize the rigorous floating-point physics simulator
backend = pf.Simulator.get_backend('analytic_simulator')

# Simulates the linear algebra of the collective unitary matrices
result = backend.run(circuit)

# The resulting physical vector in the Complex Domain natively bounds at 1.0!
print("Output Phase Angles (Radians):", np.angle(result['state_vector']))
print("Output Amplitudes:", np.abs(result['state_vector'])) # Will always equal 1.0
```

*(For more advanced dynamic non-linear functions like Logarithmic Compression, refer to `examples/ex_02_circuit_operations.py`!)*


---

**© 2026 Mindverse Computing LLC.**  
Licensed under CC BY-NC 4.0.  
See LICENSE file for patent and commercial restrictions.
