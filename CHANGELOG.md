# PhasorFlow Changelog

## v0.3.0 — Correctness & performance pass

This release fixes three correctness bugs that invalidated the deep-stack and
transformer results, and adds a batched engine that makes depth studies
tractable. All changes preserve the public API and the central phasor-circuit
philosophy; single-block results are numerically unchanged.

### Fixed

- **Transformer readout NaN (critical).** `PhasorTransformer._readout` used
  `arcsin(sin(phase))`, whose autograd derivative diverges to ±inf at
  `phase = ±π/2 + kπ` — exactly the encoding-domain boundary. Training reliably
  produced NaN losses (reproduced: divergence at epoch 51 for a 2-block model).
  Replaced with an `atan2`-based triangle fold that matches `arcsin(sin)` to
  ~3e-5 but has a finite gradient (magnitude 1) everywhere.

- **Stacking detached gradients (critical).** `VPC` and `PhasorTransformer`
  (separate mode) read inter-block phases with `.item()`, detaching them from
  the autograd graph. Consequence: in an N-stack model, only the *last* stack
  received gradient — all earlier stacks were frozen at initialization, so
  "adding blocks" could not improve the model. The batched forward now threads
  a differentiable complex state through every block; gradient reaches all
  stacks (verified in `tests/test_correctness.py`).

- **`inter_stack='pullback'` was a no-op.** `VPC._apply_inter_stack` returned
  the same phases for `'none'` and `'pullback'`, so the deep-stack pull-back
  vs. deep-circuit comparison measured nothing. The batched forward now carries
  the *complex amplitude* between stacks, so `'none'` (amplitude drifts into
  C^N), `'pullback'` (renormalise to T^N), and `'threshold'` are genuinely
  distinct operations.

### Added

- **`phasorflow/engine/vectorized.py`** — batched, fully-differentiable
  primitives (`encode_phase`, `shift_all`, `mix_adjacent`, `dft_all`,
  `pullback`, `threshold_gate`) operating on a `(batch, N)` complex state.
  Numerically identical to `AnalyticEngine` on a single circuit (max abs diff
  ~2e-7), ~1000× faster for model training (VPC binary: 0.1 s vs ~150 s for
  100 epochs / 800 samples).

- **`phasorflow/tests/test_correctness.py`** — gate unitarity, readout
  gradient finiteness, vectorized/reference equivalence, pullback≠none, and
  gradient-reaches-all-stacks regression tests. Run:
  `PYTHONPATH=. pytest phasorflow/tests/ -q`.

### Added — real data and reproducible benchmarks

- **`phasorflow/data/`** — self-contained PhysioNet EEG Motor Movement/Imagery
  pipeline. `download_dataset` / `prepare_dataset` fetch and cache the EDF
  files; `load_subject_epochs` and `epochs_to_bandpassed` produce
  motor-imagery epochs ready for Common Spatial Pattern feature extraction.
  This is the real-data source for the VPC manuscript's headline benchmark.

- **`phasorflow/benchmarks/`** —
  - `tasks.py`: synthetic task families with *known* separability
    (`sum_cosine_task`, `phase_parity_task`, `multifreq_sequence_task`) plus
    honest baselines (`classification_baselines`, `regression_baselines`,
    `SelfAttnRegressor`).
  - `eeg_benchmark.py`: the real-EEG VPC benchmark (CSP + subject-wise 5-fold
    CV) vs. LDA / logistic regression / RBF-SVM / MLP. Reproduces the VPC
    manuscript Table 1 (`python -m phasorflow.benchmarks.eeg_benchmark`).
  - `depth_study.py`: VPC capacity ceiling on phase-XOR + Phasor Transformer
    depth scaling, with a `verify_gradient_flow` regression check.

- **`examples/ex_09_real_eeg_vpc.py`** and **`examples/ex_10_depth_scaling.py`**
  — one-command reproductions of the two studies above.

- **`pyproject.toml`** / **`requirements.txt`** — installable packaging with
  optional extras: `pip install .[eeg]` pulls in `mne`+`scikit-learn`+`scipy`
  for the real-data benchmark; `.[benchmarks]`, `.[test]`, `.[all]` also
  provided.

### Removed

- The dead per-sample multi-stack path (`VPC._run_stacks` /
  `_apply_inter_stack`) that contained the gradient-detachment and pull-back
  no-op defects has been deleted; all training/inference routes through the
  batched engine. `_build_block` / `_build_circuit` are retained solely for
  single-block circuit visualization and for `PhasorGAN`.
- The `VPC._apply_inter_stack_DEPRECATED` stub (a placeholder that only raised
  `NotImplementedError`) has been removed.

### Release hardening

- **Packaging fixed.** `pyproject.toml` previously mis-nested the package tree:
  because this repository uses a directory-is-the-package layout,
  `packages.find` discovered the sub-packages (`engine`, `gates`, `models`,
  `benchmarks`, ...) as *top-level* names rather than under `phasorflow.`, so a
  wheel installed them as colliding top-level modules and `import phasorflow`
  did not resolve correctly. Replaced with an explicit `package-dir =
  {"phasorflow" = "."}` map and an enumerated sub-package list; a clean-venv
  wheel build now installs a single correctly-nested `phasorflow` package
  (verified: `import phasorflow`, all sub-modules, 22 gates, and a VPC forward
  pass all work from an isolated environment).
- **Example bootstraps cleaned.** `examples/ex_01`–`ex_07` carried a stale
  source-checkout bootstrap that used the string literal `'__file__'` (not the
  `__file__` variable), walked a non-existent capital-`PhasorFlow` directory,
  and deleted `phasorflow` entries from `sys.modules` on import. Replaced with a
  minimal, correct `sys.path` insertion using the real `__file__`.
- Added a repository `.gitignore` (Python caches, build/dist, LaTeX
  build artifacts, downloaded EEG data, editor/OS files) and removed committed
  `__pycache__` / `.pytest_cache` directories.

### Notes

- The per-sample `AnalyticEngine` is retained for circuit introspection,
  visualization, and the neuromorphic/DSA examples; the model classes now use
  the vectorized engine for `forward_batch`.
- Example scripts (`examples/ex_0*.py`) reproduce their prior numbers; they
  build circuits inline and so still run at the reference-engine speed. Their
  path-bootstrap headers were cleaned up (see Release hardening) but their
  numerical behavior is unchanged.
- Reproducing the real-EEG benchmark requires network access to
  `physionet.org` on first run (~75 MB, cached thereafter). If MNE's numba
  backend raises a caching error in a restricted sandbox, set
  `NUMBA_DISABLE_JIT=1`.
