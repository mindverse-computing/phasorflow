# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# examples/memory_recovery.py

import phasorflow as sn
import numpy as np
from phasorflow.neuromorphic.associative_memory import PhasorFlowMemory

def run_memory_demo():
    print("--- Oscillatory Associative Memory: Pattern Recovery ---")
    
    n = 10
    mem = PhasorFlowMemory(n)
    
    # 1. Define a 'Stored Memory': A binary phase pattern (0 or Pi)
    stored_pattern = np.array([0, np.pi, 0, np.pi, 0, np.pi, 0, np.pi, 0, np.pi])
    mem.store([stored_pattern])
    
    # 2. Create a 'Corrupted' Input: Flip 20% of the phases (2 errors)
    corrupted_pattern = stored_pattern.copy()
    corrupted_pattern[0] = np.pi # was 0
    corrupted_pattern[1] = 0     # was pi
    
    state = np.exp(1j * corrupted_pattern)
    
    def phase_diff(a, b):
        diff = np.abs(a - b)
        return np.minimum(diff, 2*np.pi - diff)
        
    initial_err = np.mean(phase_diff(np.angle(state), stored_pattern))
    print(f"Initial Phase Error: {initial_err:.4f}")
    
    # 3. Converge toward the attractor
    state = mem.converge(state, iterations=100)
    
    # 4. Final Result
    final_phases = np.angle(state)
    
    # Networks are invariant to global Pi shifts, we check both the pattern and its inverse
    err1 = np.mean(phase_diff(final_phases, stored_pattern))
    err2 = np.mean(phase_diff(final_phases, (stored_pattern + np.pi) % (2*np.pi)))
    final_err = min(err1, err2)
    
    print(f"Final Phase Error: {final_err:.4f}")
    
    if final_err < 0.2:
        print("Status: Attractor Recovered Successfully.")
    else:
        print("Status: Failed to Recover Attractor.")

if __name__ == "__main__":
    run_memory_demo()
