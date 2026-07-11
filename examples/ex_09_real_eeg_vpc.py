"""
Example 09: Real motor-imagery EEG decoding with the VPC.

Downloads the PhysioNet EEG Motor Movement/Imagery dataset (first 10 subjects,
runs 4/8/12), extracts CSP features, and benchmarks the VPC against standard
brain-computer-interface baselines (LDA, logistic regression, RBF-SVM, MLP)
under subject-wise 5-fold cross-validation.

This reproduces Table 1 / Figure (eeg_benchmark) of the VPC manuscript.

Requirements: mne, scikit-learn, scipy  (pip install mne scikit-learn scipy)
Network access to physionet.org is required on first run (data is cached).

Run:
    NUMBA_DISABLE_JIT=1 python ex_09_real_eeg_vpc.py
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from phasorflow.benchmarks.eeg_benchmark import evaluate

if __name__ == "__main__":
    print("Running real-EEG VPC benchmark (downloads ~75 MB on first run)...")
    rows = evaluate(out_path="eeg_results.json")
    print(f"\nDone. Per-subject results written to eeg_results.json "
          f"({len(rows)} subjects).")
