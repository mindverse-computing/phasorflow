# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/models/vpc.py

"""
Variational Phasor Circuit (VPC) — High-Level ML Classifier.

A trainable phasor circuit model analogous to Variational Quantum
Circuits (VQCs) in quantum ML, but operating entirely on classical
unit-circle vectors.

Supports:
  - Binary classification (num_classes=2, default)
  - Multi-class classification (num_classes>2)
  - Single-stack and multi-stack architectures

Architecture:
  Single (num_stacks=1):
      [Data Encoding] → [Trainable Shifts → Coupling] × num_layers → Readout

  Stacked (num_stacks>1):
      [Block 1] → [Inter-Stack Op] → [Block 2] → ... → Readout

      Each block is a full VPC with its own trainable weights.
      Between blocks, an inter-stack operation re-projects phasors
      onto the N-torus:

      inter_stack='none'      : pass raw phases (notebook 3.2)
      inter_stack='pullback'  : extract phases only (notebook 3.3)
      inter_stack='threshold' : threshold gate (notebook 3.4)

Coupling strategies per layer:
    'mix_dft'  : alternating Mix pairs + global DFT  (default)
    'mix_only' : alternating Mix pairs only
    'dft_only' : global DFT only

Readout:
  Binary (num_classes=2):
      Thread-0 phase → sine/cosine envelope → [0,1] probability
  Multi-class (num_classes>2):
      First K thread phases → sin() * logit_scale → Softmax → K probabilities

Usage:
    # Binary classification (notebooks 3.1-3.4)
    model = VPC(num_features=12, num_layers=2)
    model.fit(X_train, y_train, epochs=100, lr=0.1)

    # Multi-class classification (notebook 3.6)
    model = VPC(num_features=32, num_layers=2, num_classes=4)
    model.fit(X_train, y_train, epochs=100, lr=0.1)

    # Multi-stack with pullback (notebook 3.3)
    model = VPC(num_features=12, num_layers=2, num_stacks=3,
                inter_stack='pullback')

    accuracy = model.score(X_test, y_test)
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim

from ..circuit import PhasorCircuit
from ..engine.analytic import AnalyticEngine


class VPC(nn.Module):
    """
    Variational Phasor Circuit classifier.

    A torch.nn.Module that wraps PhasorCircuit + AnalyticEngine
    with a scikit-learn-style API (fit / predict / score).

    Supports binary and multi-class classification, as well as
    single-stack and multi-stack architectures.

    Args:
        num_features : int
            Number of input features (= number of oscillator threads).
        num_classes : int, default 2
            Number of target classes.
            - 2: binary classification (MSE loss, sine envelope readout)
            - >2: multi-class classification (cross-entropy loss, Softmax readout)
            Must satisfy num_classes <= num_features.
        num_layers : int, default 2
            Number of variational layers per stack.
        num_stacks : int, default 1
            Number of VPC blocks to stack sequentially.
        coupling : str, default 'mix_dft'
            Coupling strategy per variational layer:
            - 'mix_dft'  : alternating Mix pairs + global DFT
            - 'mix_only' : alternating Mix pairs only
            - 'dft_only' : global DFT only
        inter_stack : str, default 'pullback'
            Operation applied between stacks (only when num_stacks > 1):
            - 'none'      : pass raw output phases
            - 'pullback'  : extract phases only
            - 'threshold' : zero phases where amplitude < threshold_tau
        threshold_tau : float, default 0.5
            Threshold value for the 'threshold' inter-stack operation.
        readout : str, default 'sine'
            Phase-to-probability mapping (binary mode only):
            - 'sine'   : p = (sin(θ) + 1) / 2
            - 'cosine' : p = (cos(θ) + 1) / 2
        logit_scale : float, default 5.0
            Scaling factor for multi-class Softmax logits.
            Amplifies phase differences for sharper class separation.
    """

    COUPLING_STRATEGIES = ('mix_dft', 'mix_only', 'dft_only')
    INTER_STACK_OPS = ('none', 'pullback', 'threshold')
    READOUT_MODES = ('sine', 'cosine')

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        num_layers: int = 2,
        num_stacks: int = 1,
        coupling: str = 'mix_dft',
        inter_stack: str = 'pullback',
        threshold_tau: float = 0.5,
        readout: str = 'sine',
        logit_scale: float = 5.0,
    ):
        super().__init__()

        if coupling not in self.COUPLING_STRATEGIES:
            raise ValueError(
                f"coupling must be one of {self.COUPLING_STRATEGIES}, "
                f"got '{coupling}'"
            )
        if inter_stack not in self.INTER_STACK_OPS:
            raise ValueError(
                f"inter_stack must be one of {self.INTER_STACK_OPS}, "
                f"got '{inter_stack}'"
            )
        if readout not in self.READOUT_MODES:
            raise ValueError(
                f"readout must be one of {self.READOUT_MODES}, "
                f"got '{readout}'"
            )
        if num_stacks < 1:
            raise ValueError(f"num_stacks must be >= 1, got {num_stacks}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if num_classes > num_features:
            raise ValueError(
                f"num_classes ({num_classes}) must be <= num_features "
                f"({num_features}), since each class reads one thread phase"
            )

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_stacks = num_stacks
        self.coupling = coupling
        self.inter_stack = inter_stack
        self.threshold_tau = threshold_tau
        self.readout = readout
        self.logit_scale = logit_scale

        # Derived: is this binary or multi-class?
        self._is_binary = (num_classes == 2)

        # Engine (stateless, shared)
        self._engine = AnalyticEngine()

        # Parameters per stack = num_layers * num_features
        params_per_stack = num_layers * num_features
        total_params = num_stacks * params_per_stack

        # Trainable parameters: one phase-shift per thread per layer per stack
        self.weights = nn.Parameter(
            torch.empty(total_params).uniform_(-math.pi, math.pi)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params_per_stack(self) -> int:
        """Number of trainable parameters per stack."""
        return self.num_layers * self.num_features

    @property
    def total_params(self) -> int:
        """Total number of trainable parameters."""
        return self.num_stacks * self.params_per_stack

    # ------------------------------------------------------------------
    # Circuit Construction
    # ------------------------------------------------------------------

    def _build_block(
        self,
        x: torch.Tensor,
        block_weights: torch.Tensor,
        block_id: int = 0,
    ) -> PhasorCircuit:
        """
        Construct a single VPC block (one stack).

        Args:
            x: Phase input tensor of shape (num_features,).
            block_weights: Trainable weights for this block,
                          shape (num_layers * num_features,).
            block_id: Block index for naming.

        Returns:
            PhasorCircuit instance.
        """
        N = self.num_features
        name = "VPC_Classifier" if self.num_stacks == 1 else f"VPC_Stack_{block_id}"
        pc = PhasorCircuit(N, name=name)

        # ── Data / Phase Encoding ──
        for i in range(N):
            val = x[i].item() if isinstance(x[i], torch.Tensor) else float(x[i])
            pc.shift(i, val)

        # ── Variational Layers ──
        for layer in range(self.num_layers):
            offset = layer * N

            # Trainable phase shifts
            for i in range(N):
                pc.shift(i, block_weights[offset + i])

            # Coupling
            if self.coupling in ('mix_dft', 'mix_only'):
                for i in range(0, N - 1, 2):
                    pc.mix(i, i + 1)

            if self.coupling in ('mix_dft', 'dft_only'):
                pc.dft()

        return pc

    def _apply_inter_stack(
        self,
        phases: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply inter-stack operation to re-project phasors onto N-torus.

        Args:
            phases: Phase tensor from previous stack, shape (N,).
            amplitudes: Amplitude tensor from previous stack, shape (N,).

        Returns:
            Phase tensor to feed into the next stack, shape (N,).
        """
        if self.inter_stack == 'none':
            return phases

        elif self.inter_stack == 'pullback':
            return phases

        elif self.inter_stack == 'threshold':
            mask = (amplitudes >= self.threshold_tau).to(phases.dtype)
            return phases * mask

        else:
            raise ValueError(f"Unknown inter_stack: '{self.inter_stack}'")

    def _build_circuit(self, x: torch.Tensor) -> PhasorCircuit:
        """
        Construct a single-stack VPC circuit (for introspection).
        """
        if self.num_stacks != 1:
            return self._build_block(
                x, self.weights[:self.params_per_stack], block_id=0
            )
        return self._build_block(x, self.weights, block_id=0)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def _phase_to_prob_binary(self, phase: torch.Tensor) -> torch.Tensor:
        """Map a phase value to a [0, 1] probability (binary mode)."""
        if self.readout == 'sine':
            return (torch.sin(phase) + 1.0) / 2.0
        else:  # cosine
            return (torch.cos(phase) + 1.0) / 2.0

    def _phases_to_probs_multiclass(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Map the first num_classes thread phases to class probabilities
        via stable Softmax (multi-class mode).

        Uses sin() to smooth the torus boundaries for continuous
        autograd gradients, then scales by logit_scale.
        """
        K = self.num_classes
        logits = torch.sin(phases[:K]) * self.logit_scale
        # Stable Softmax
        e_x = torch.exp(logits - torch.max(logits))
        return e_x / e_x.sum(dim=0)

    def _run_stacks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run through all stacks and return the final output phases.

        Args:
            x: Feature tensor of shape (num_features,).

        Returns:
            Phase tensor from the final stack.
        """
        pps = self.params_per_stack
        current_phases = x

        for stack in range(self.num_stacks):
            stack_weights = self.weights[stack * pps : (stack + 1) * pps]
            pc = self._build_block(current_phases, stack_weights, block_id=stack)
            result = self._engine.run(pc)

            if stack < self.num_stacks - 1:
                current_phases = self._apply_inter_stack(
                    result['phases'],
                    torch.abs(result['state_vector']),
                )
            else:
                current_phases = result['phases']

        return current_phases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the VPC on a single input sample.

        Args:
            x: Feature tensor of shape (num_features,).

        Returns:
            Binary mode: scalar probability in [0, 1].
            Multi-class mode: probability vector of shape (num_classes,).
        """
        final_phases = self._run_stacks(x)

        if self._is_binary:
            return self._phase_to_prob_binary(final_phases[0])
        else:
            return self._phases_to_probs_multiclass(final_phases)

    def forward_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the VPC on a batch of samples.

        Args:
            X: Feature tensor of shape (batch_size, num_features).

        Returns:
            Binary mode: probability tensor (batch_size,).
            Multi-class mode: probability tensor (batch_size, num_classes).
        """
        return torch.stack([self.forward(X[i]) for i in range(X.shape[0])])

    # ------------------------------------------------------------------
    # Scikit-learn-style API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.1,
        verbose: bool = True,
        print_every: int = 20,
    ) -> dict:
        """
        Train the VPC classifier using Adam optimization.

        Args:
            X: Training features, shape (n_samples, num_features).
            y: Training labels.
               Binary mode: float tensor of 0.0/1.0, shape (n_samples,).
               Multi-class mode: long tensor of class indices, shape (n_samples,).
            epochs: Number of training epochs.
            lr: Learning rate for Adam optimizer.
            verbose: If True, print loss at intervals.
            print_every: Print interval (epochs).

        Returns:
            dict with 'final_loss' and 'loss_history'.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            if self._is_binary:
                y = torch.tensor(y, dtype=torch.float32)
            else:
                y = torch.tensor(y, dtype=torch.long)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward_batch(X)

            if self._is_binary:
                # MSE loss for binary classification
                loss = torch.mean((predictions - y) ** 2)
            else:
                # Cross-entropy loss for multi-class classification
                # Negative log-likelihood on the correct class probability
                loss = -torch.mean(
                    torch.log(predictions[torch.arange(len(y)), y] + 1e-9)
                )

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss_val:.4f}")

        if verbose:
            print(f"Training complete! Final loss: {loss_history[-1]:.4f}")

        return {
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
        }

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return continuous probabilities for each sample.

        Args:
            X: Feature tensor, shape (n_samples, num_features).

        Returns:
            Binary mode: probability tensor (n_samples,).
            Multi-class mode: probability tensor (n_samples, num_classes).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward_batch(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class predictions.

        Args:
            X: Feature tensor, shape (n_samples, num_features).

        Returns:
            Binary mode: float tensor of 0.0/1.0, shape (n_samples,).
            Multi-class mode: long tensor of class indices, shape (n_samples,).
        """
        proba = self.predict_proba(X)
        if self._is_binary:
            return (proba > 0.5).float()
        else:
            return torch.argmax(proba, dim=1)

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Return classification accuracy.

        Args:
            X: Feature tensor, shape (n_samples, num_features).
            y: True labels, shape (n_samples,).

        Returns:
            Accuracy as a float in [0, 1].
        """
        if not isinstance(y, torch.Tensor):
            if self._is_binary:
                y = torch.tensor(y, dtype=torch.float32)
            else:
                y = torch.tensor(y, dtype=torch.long)
        preds = self.predict(X)
        return torch.mean((preds == y).float()).item()

    def get_circuit(self, x: torch.Tensor, stack_id: int = 0) -> PhasorCircuit:
        """
        Return the underlying PhasorCircuit for a given input.

        Useful for inspection, drawing, or manual execution.

        Args:
            x: Feature tensor, shape (num_features,).
            stack_id: Which stack's circuit to return (0-indexed).

        Returns:
            PhasorCircuit instance.
        """
        if stack_id < 0 or stack_id >= self.num_stacks:
            raise IndexError(
                f"stack_id {stack_id} out of range [0, {self.num_stacks})"
            )
        pps = self.params_per_stack
        stack_weights = self.weights[stack_id * pps : (stack_id + 1) * pps]
        return self._build_block(x, stack_weights, block_id=stack_id)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "binary" if self._is_binary else f"{self.num_classes}-class"
        lines = [
            f"VPC(",
            f"  mode='{mode}',",
            f"  num_features={self.num_features},",
            f"  num_classes={self.num_classes},",
            f"  num_layers={self.num_layers},",
            f"  num_stacks={self.num_stacks},",
            f"  coupling='{self.coupling}',",
        ]
        if self.num_stacks > 1:
            lines.append(f"  inter_stack='{self.inter_stack}',")
            if self.inter_stack == 'threshold':
                lines.append(f"  threshold_tau={self.threshold_tau},")
        if self._is_binary:
            lines.append(f"  readout='{self.readout}',")
        else:
            lines.append(f"  logit_scale={self.logit_scale},")
        lines.extend([
            f"  trainable_params={self.total_params}",
            f")",
        ])
        return "\n".join(lines)
