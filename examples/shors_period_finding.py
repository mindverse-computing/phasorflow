# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# examples/shors_period_finding.py

import phasorflow as sn
import numpy as np
import matplotlib.pyplot as plt

def run_shors_demonstration():
    print("--- Shor's Period Finding Simulation (N=15, a=7) ---")
    
    # We use 4 threads to represent the repeating sequence
    n_threads = 4
    pc = sn.PhasorCircuit(n_threads)
    
    # 1. Modular Exponentiation Encoding
    # The sequence for 7^x mod 15 is [1, 7, 4, 13]
    # We map these to phases on the unit circle
    phases = [
        (1 % 15) * (2 * np.pi / 15),
        (7 % 15) * (2 * np.pi / 15),
        (4 % 15) * (2 * np.pi / 15),
        (13 % 15) * (2 * np.pi / 15)
    ]
    
    for i, p in enumerate(phases):
        pc.shift(i, p)
        
    # 2. Apply the Global DFT (The 'Quantum' Fourier Step)
    # This gate will cause the phases to interfere and reveal the frequency
    pc.dft()
    
    # 3. Execution
    backend = sn.Simulator.get_backend('analytic_simulator')
    result = backend.run(pc)
    
    # 4. Results Analysis
    magnitudes = np.abs(result['state_vector'])
    print("\nResulting Spectral Magnitudes:")
    for i, m in enumerate(magnitudes):
        print(f"Frequency Component {i}: {m:.4f}")
        
    # Find the peak frequency (excluding DC offset at index 0 if necessary)
    peak_freq = np.argmax(magnitudes[1:]) + 1
    print(f"\nDetected Period Resonance: {peak_freq}")
    print(f"Factors of 15 can now be derived from r={peak_freq}")

    # 5. Visualization
    sn.draw(pc, mode='text')
    
if __name__ == "__main__":
    run_shors_demonstration()
