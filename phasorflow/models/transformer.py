# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/models/transformer.py

"""
Phasor Transformer — FNet-Style Sequence Prediction Model.

A trainable phasor circuit model implementing an FNet-style
Transformer architecture for continuous sequence prediction.
Uses the Discrete Fourier Transform (DFT) for token mixing
and trainable phase shifts as position-wise feed-forward
projections.

Supports:
  - Single-circuit mode: all blocks in one PhasorCircuit (notebooks 4.1, 4.3)
  - Stacked (multi-circuit) mode: separate circuits per block with
    inter-block operations (notebook 4.2)
  - Optional readout smoothing layer (notebook 4.4 / LPM)
  - Configurable number of blocks for depth scaling

Architecture per block:
    [Pre-FFN Shift] → [DFT Token Mixing] → [Post-FFN Shift]

    Each block consumes 2*T parameters (T pre-shifts + T post-shifts),
    where T = seq_length (number of oscillator threads).

Readout:
    Thread-0 phase → arcsin(sin(phase))  (autograd-safe wrapping to [-π/2, π/2])

Loss:
    MSE between predicted and target next-step values.

Usage:
    # Single-circuit transformer (notebook 4.1 style)
    model = PhasorTransformer(seq_length=10, num_blocks=2)
    model.fit(X_train, y_train, epochs=100, lr=0.05)

    # Stacked with threshold gate (notebook 4.2 style)
    model = PhasorTransformer(seq_length=10, num_blocks=2,
                              stacking='separate', inter_block='threshold')

    # LPM-style deep stack with readout layer (notebook 4.4 style)
    model = PhasorTransformer(seq_length=16, num_blocks=3,
                              readout_layer=True)

    predictions = model.predict(X_test)
    mse = model.score(X_test, y_test)
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim

from ..circuit import PhasorCircuit
from ..engine.analytic import AnalyticEngine


class PhasorTransformer(nn.Module):
    """
    FNet-style Phasor Transformer for sequence prediction.

    A torch.nn.Module that wraps PhasorCircuit + AnalyticEngine
    with a scikit-learn-style API (fit / predict / score).

    Each FNet block applies:
        1. Pre-FFN trainable phase shifts (T parameters)
        2. Global DFT for token mixing
        3. Post-FFN trainable phase shifts (T parameters)

    The model predicts the next value in a sequence by reading
    thread-0's phase after processing.

    Args:
        seq_length : int
            Length of input sequences (= number of oscillator threads T).
        num_blocks : int, default 2
            Number of FNet blocks (depth of the transformer).
        stacking : str, default 'single'
            How blocks are composed:
            - 'single'   : all blocks in one PhasorCircuit (notebook 4.1)
            - 'separate' : each block is a separate circuit with
                          inter-block operations between them (notebook 4.2)
        inter_block : str, default 'none'
            Operation between blocks (only used when stacking='separate'):
            - 'none'      : pass raw output phases to next block
            - 'threshold' : zero phases where amplitude < threshold_tau
        threshold_tau : float, default 0.5
            Threshold for the 'threshold' inter-block operation.
        readout_layer : bool, default False
            If True, add a final smoothing shift layer (T extra parameters)
            after all blocks (notebook 4.4 / LPM style).
        init_range : float, default π/10
            Range for uniform initialization of trainable weights: [-init_range, init_range].
    """

    STACKING_MODES = ('single', 'separate')
    INTER_BLOCK_OPS = ('none', 'threshold')

    def __init__(
        self,
        seq_length: int,
        num_blocks: int = 2,
        stacking: str = 'single',
        inter_block: str = 'none',
        threshold_tau: float = 0.5,
        readout_layer: bool = False,
        init_range: float = math.pi / 10,
    ):
        super().__init__()

        if stacking not in self.STACKING_MODES:
            raise ValueError(
                f"stacking must be one of {self.STACKING_MODES}, "
                f"got '{stacking}'"
            )
        if inter_block not in self.INTER_BLOCK_OPS:
            raise ValueError(
                f"inter_block must be one of {self.INTER_BLOCK_OPS}, "
                f"got '{inter_block}'"
            )
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if seq_length < 2:
            raise ValueError(f"seq_length must be >= 2, got {seq_length}")

        self.seq_length = seq_length
        self.num_blocks = num_blocks
        self.stacking = stacking
        self.inter_block = inter_block
        self.threshold_tau = threshold_tau
        self.readout_layer = readout_layer
        self.init_range = init_range

        # Engine (stateless, shared)
        self._engine = AnalyticEngine()

        # Total trainable parameters:
        #   num_blocks * 2 * T  (pre-shift + post-shift per block)
        #   + T if readout_layer is True
        self._params_per_block = 2 * seq_length
        total_params = num_blocks * self._params_per_block
        if readout_layer:
            total_params += seq_length

        self.weights = nn.Parameter(
            torch.empty(total_params).uniform_(-init_range, init_range)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params_per_block(self) -> int:
        """Number of trainable parameters per FNet block (2 * seq_length)."""
        return self._params_per_block

    @property
    def total_params(self) -> int:
        """Total number of trainable parameters."""
        return self.weights.numel()

    # ------------------------------------------------------------------
    # Circuit Construction
    # ------------------------------------------------------------------

    def _build_single_circuit(
        self,
        x_seq: torch.Tensor,
    ) -> PhasorCircuit:
        """
        Build a single PhasorCircuit with all FNet blocks inline.

        This matches the architecture in notebooks 4.1 and 4.3:
        all blocks are composed within one circuit.

        Args:
            x_seq: Input sequence tensor of shape (seq_length,).

        Returns:
            PhasorCircuit instance.
        """
        T = self.seq_length
        pc = PhasorCircuit(T, name="PhasorTransformer")

        # Data injection (encode input sequence into thread phases)
        for i in range(T):
            pc.shift(i, x_seq[i])

        weight_idx = 0

        # FNet blocks
        for _block in range(self.num_blocks):
            # Pre-FFN projection (trainable shifts)
            for i in range(T):
                pc.shift(i, self.weights[weight_idx + i])
            weight_idx += T

            # Token mixing (global DFT)
            pc.dft()

            # Post-FFN projection (trainable shifts)
            for i in range(T):
                pc.shift(i, self.weights[weight_idx + i])
            weight_idx += T

        # Optional readout smoothing layer
        if self.readout_layer:
            for i in range(T):
                pc.shift(i, self.weights[weight_idx + i])

        return pc

    def _build_block_circuit(
        self,
        x_phases: torch.Tensor,
        block_weights: torch.Tensor,
        block_id: int = 0,
    ) -> PhasorCircuit:
        """
        Build a single FNet block as its own PhasorCircuit.

        Used in 'separate' stacking mode (notebook 4.2).

        Args:
            x_phases: Input phase tensor, shape (seq_length,).
            block_weights: Trainable weights for this block, shape (2*T,).
            block_id: Block index for naming.

        Returns:
            PhasorCircuit instance.
        """
        T = self.seq_length
        pc = PhasorCircuit(T, name=f"Transformer_Block_{block_id}")

        # Data / Phase injection
        for i in range(T):
            val = x_phases[i].item() if isinstance(x_phases[i], torch.Tensor) else float(x_phases[i])
            pc.shift(i, val)

        # Pre-FFN projection
        for i in range(T):
            pc.shift(i, block_weights[i])

        # Token mixing (global DFT)
        pc.dft()

        # Post-FFN projection
        for i in range(T):
            pc.shift(i, block_weights[T + i])

        return pc

    def _build_readout_circuit(
        self,
        x_phases: torch.Tensor,
        readout_weights: torch.Tensor,
    ) -> PhasorCircuit:
        """
        Build the readout smoothing circuit.

        Args:
            x_phases: Phase tensor from last block, shape (seq_length,).
            readout_weights: Readout layer weights, shape (seq_length,).

        Returns:
            PhasorCircuit instance.
        """
        T = self.seq_length
        pc = PhasorCircuit(T, name="Transformer_Readout")

        for i in range(T):
            val = x_phases[i].item() if isinstance(x_phases[i], torch.Tensor) else float(x_phases[i])
            pc.shift(i, val)

        for i in range(T):
            pc.shift(i, readout_weights[i])

        return pc

    def _apply_inter_block(
        self,
        phases: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply inter-block operation to re-project phasors.

        Args:
            phases: Phase tensor from previous block, shape (T,).
            amplitudes: Amplitude tensor from previous block, shape (T,).

        Returns:
            Phase tensor to feed into the next block, shape (T,).
        """
        if self.inter_block == 'none':
            return phases
        elif self.inter_block == 'threshold':
            mask = (amplitudes >= self.threshold_tau).to(phases.dtype)
            return phases * mask
        else:
            raise ValueError(f"Unknown inter_block: '{self.inter_block}'")

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def _readout(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Autograd-safe phase wrapping readout.

        Maps the raw thread-0 phase to [-π/2, π/2] via arcsin(sin(phase)),
        which is a continuous differentiable approximation of modular
        arithmetic on the unit circle.

        Args:
            phase: Raw phase value (scalar tensor).

        Returns:
            Wrapped prediction (scalar tensor).
        """
        return torch.asin(torch.sin(phase))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Run the Phasor Transformer on a single input sequence.

        Args:
            x_seq: Input sequence tensor of shape (seq_length,).

        Returns:
            Scalar prediction (next-step value).
        """
        T = self.seq_length
        ppb = self._params_per_block

        if self.stacking == 'single':
            # All blocks in one circuit
            pc = self._build_single_circuit(x_seq)
            result = self._engine.run(pc)
            raw_phase = result['phases'][0]
            return self._readout(raw_phase)

        else:  # stacking == 'separate'
            current_phases = x_seq

            for block in range(self.num_blocks):
                block_weights = self.weights[block * ppb : (block + 1) * ppb]
                pc = self._build_block_circuit(
                    current_phases, block_weights, block_id=block
                )
                result = self._engine.run(pc)

                if block < self.num_blocks - 1:
                    current_phases = self._apply_inter_block(
                        result['phases'],
                        torch.abs(result['state_vector']),
                    )
                else:
                    current_phases = result['phases']

            # Optional readout smoothing layer
            if self.readout_layer:
                readout_weights = self.weights[self.num_blocks * ppb:]
                pc = self._build_readout_circuit(current_phases, readout_weights)
                result = self._engine.run(pc)
                current_phases = result['phases']

            return self._readout(current_phases[0])

    def forward_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the Phasor Transformer on a batch of sequences.

        Args:
            X: Input tensor of shape (batch_size, seq_length).

        Returns:
            Prediction tensor of shape (batch_size,).
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
        lr: float = 0.05,
        optimizer_type: str = 'adam',
        verbose: bool = True,
        print_every: int = 10,
    ) -> dict:
        """
        Train the Phasor Transformer using gradient-based optimization.

        Args:
            X: Training sequences, shape (n_samples, seq_length).
               Values should be pre-encoded to phase domain (e.g. [-π/2, π/2]).
            y: Training targets (next-step values), shape (n_samples,).
               Should be in the same phase domain as X.
            epochs: Number of training epochs.
            lr: Learning rate.
            optimizer_type: 'adam' (default) or 'sgd'.
            verbose: If True, print loss at intervals.
            print_every: Print interval (epochs).

        Returns:
            dict with 'final_loss' and 'loss_history'.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(
                f"optimizer_type must be 'adam' or 'sgd', got '{optimizer_type}'"
            )

        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward_batch(X)

            # MSE loss (standard for sequence prediction)
            loss = torch.mean((predictions - y) ** 2)

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

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return next-step predictions for a batch of sequences.

        Args:
            X: Input sequences, shape (n_samples, seq_length).

        Returns:
            Prediction tensor of shape (n_samples,).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward_batch(X)

    def predict_autoregressive(
        self,
        context: torch.Tensor,
        horizon: int = 20,
    ) -> torch.Tensor:
        """
        Generate future steps autoregressively (LPM-style, notebook 4.4).

        Starting from a context window, repeatedly predict the next step
        and roll the context forward.

        Args:
            context: Initial context sequence, shape (seq_length,).
                     Should be phase-encoded.
            horizon: Number of future steps to generate.

        Returns:
            Tensor of generated predictions, shape (horizon,).
        """
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)

        context = context.clone()
        predictions = []

        with torch.no_grad():
            for _ in range(horizon):
                next_step = self.forward(context)
                predictions.append(next_step)

                # Roll context window forward
                context = torch.roll(context, -1)
                context[-1] = next_step

        return torch.stack(predictions)

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Return MSE on test data.

        Args:
            X: Test sequences, shape (n_samples, seq_length).
            y: True next-step values, shape (n_samples,).

        Returns:
            MSE as a float.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        preds = self.predict(X)
        return torch.mean((preds - y) ** 2).item()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_circuit(
        self,
        x_seq: torch.Tensor,
        block_id: int = 0,
    ) -> PhasorCircuit:
        """
        Return the underlying PhasorCircuit for inspection.

        In 'single' stacking mode, returns the full circuit (block_id ignored).
        In 'separate' stacking mode, returns the specified block's circuit.

        Args:
            x_seq: Input sequence, shape (seq_length,).
            block_id: Block index (only used in 'separate' mode).

        Returns:
            PhasorCircuit instance.
        """
        if self.stacking == 'single':
            return self._build_single_circuit(x_seq)
        else:
            if block_id < 0 or block_id >= self.num_blocks:
                raise IndexError(
                    f"block_id {block_id} out of range [0, {self.num_blocks})"
                )
            ppb = self._params_per_block
            block_weights = self.weights[block_id * ppb : (block_id + 1) * ppb]
            return self._build_block_circuit(x_seq, block_weights, block_id=block_id)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [
            f"PhasorTransformer(",
            f"  seq_length={self.seq_length},",
            f"  num_blocks={self.num_blocks},",
            f"  stacking='{self.stacking}',",
        ]
        if self.stacking == 'separate':
            lines.append(f"  inter_block='{self.inter_block}',")
            if self.inter_block == 'threshold':
                lines.append(f"  threshold_tau={self.threshold_tau},")
        lines.extend([
            f"  readout_layer={self.readout_layer},",
            f"  trainable_params={self.total_params}",
            f")",
        ])
        return "\n".join(lines)
