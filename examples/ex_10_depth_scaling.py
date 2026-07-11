"""
Example 10: Depth scaling and capacity study.

Reproduces the two central architectural findings of the revised manuscripts:

  1. VPC capacity ceiling -- on the phase-XOR (parity) task the VPC stays at
     chance regardless of stack count, because it is a linear classifier in a
     fixed cos/sin feature lifting. An RBF-SVM and an MLP solve the same task.

  2. Phasor Transformer depth benefit -- on a variable-period continuation task
     the transformer's test MSE decreases monotonically with block count up to
     ~3 blocks, then saturates. This is only meaningful because the v0.3.0
     stacking fix lets gradient reach every block (verified by the script).

Run:
    python ex_10_depth_scaling.py
"""
from phasorflow.benchmarks.depth_study import main

if __name__ == "__main__":
    main(out_path="depth_scaling.json")
