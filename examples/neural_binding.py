# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# examples/neural_binding.py

import phasorflow as sn
from phasorflow.neuromorphic.lip_layer import LIPLayer
import numpy as np
import matplotlib.pyplot as plt

def simulate_binding():
    print("--- Simulating Neural Binding via Phase Synchronization ---")
    
    # 1. Setup: 2 Threads (e.g., Visual Neuron and Auditory Neuron)
    n_threads = 2
    
    # Initialize with a large phase difference (T0 at 0, T1 at PI)
    state = np.array([np.exp(1j * 0), np.exp(1j * np.pi)])
    
    # 2. Initialize LIP Layer with strong coupling
    # High coupling strength mimics a strong synaptic connection
    lip = LIPLayer(num_threads=n_threads, leak_rate=0.05)
    lip.weights = np.array([[0, 0.5], 
                           [0.5, 0]]) # T0 pulls T1, T1 pulls T0
    
    # 3. Simulation Loop
    dt = 0.1
    steps = 100
    history = []

    for _ in range(steps):
        # We provide zero external input to see the natural synchronization
        external_in = np.zeros(n_threads)
        state = lip.update(state, external_in, dt=dt)
        history.append(np.angle(state))

    history = np.array(history)

    # 4. Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(history[:, 0], label='Neuron T0 (Visual)', color='blue', lw=2)
    plt.plot(history[:, 1], label='Neuron T1 (Auditory)', color='red', lw=2, linestyle='--')
    plt.axhline(y=np.mean(history[-1]), color='green', linestyle=':', label='Synchronized Phase')
    
    plt.title("Neural Binding: Phase-Locking in the LIP Model")
    plt.xlabel("Time Steps")
    plt.ylabel("Phase (Radians)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('neural_binding_plot.png')
    print("Plot saved to neural_binding_plot.png")

    final_diff = np.abs(history[-1, 0] - history[-1, 1])
    print(f"Final Phase Difference: {final_diff:.4f} radians")
    if final_diff < 0.1:
        print("Status: Signals Bound (Synchronized)")

if __name__ == "__main__":
    simulate_binding()
