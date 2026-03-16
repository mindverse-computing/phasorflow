# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/models/gan.py

"""
Phasor GAN — Generative Adversarial Network for Timeseries Data.

A GAN where both the Generator and Discriminator are phasor circuits,
trained adversarially on timeseries data.

Architecture:
  Generator:
      z ∈ [-π, π]^T  →  [Shift → DFT] × L  →  sin(phases)  →  fake sample

  Discriminator:
      x ∈ ℝ^T  →  arcsin(x) phase-encode  →  [Shift → Mix → DFT] × L  →  P(real)

Usage:
    from PhasorFlow.models import PhasorGAN

    gan = PhasorGAN(seq_length=8, num_layers_g=2, num_layers_d=2)
    history = gan.fit(X_real, epochs=60, batch_size=10, lr_g=0.01, lr_d=0.01)
    fake_samples = gan.generate(num_samples=10)
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim

from ..circuit import PhasorCircuit
from ..engine.analytic import AnalyticEngine


class PhasorGenerator(nn.Module):
    """
    Generator that maps random latent noise through a phasor circuit
    to produce synthetic timeseries samples.

    Architecture:
        z → [Shift → DFT] × num_layers → sin(output_phases) → sample

    Args:
        seq_length: Length of generated timeseries (= number of threads).
        num_layers: Number of variational layers in the circuit.
    """

    def __init__(self, seq_length, num_layers=3):
        super().__init__()
        self.seq_length = seq_length
        self.num_layers = num_layers
        self._engine = AnalyticEngine()

        total_params = num_layers * seq_length
        self.weights = nn.Parameter(
            torch.empty(total_params).uniform_(-math.pi / 4, math.pi / 4)
        )

    def _build_circuit(self, z):
        """Build generator phasor circuit from latent noise z."""
        T = self.seq_length
        pc = PhasorCircuit(T, name="Generator")

        for i in range(T):
            pc.shift(i, z[i])

        for layer in range(self.num_layers):
            offset = layer * T
            for i in range(T):
                pc.shift(i, self.weights[offset + i])
            pc.dft()

        return pc

    def forward(self, z):
        """
        Generate a single timeseries sample from latent noise.

        Args:
            z: Latent noise tensor, shape (seq_length,), values in [-π, π].

        Returns:
            Generated sample tensor, shape (seq_length,), values in [-1, 1].
        """
        pc = self._build_circuit(z)
        result = self._engine.run(pc)
        return torch.sin(result['phases'])

    def generate_batch(self, batch_size):
        """Generate a batch of fake samples from random noise."""
        z_batch = torch.empty(batch_size, self.seq_length).uniform_(
            -math.pi, math.pi
        )
        return torch.stack([self.forward(z_batch[i]) for i in range(batch_size)])


class PhasorDiscriminator(nn.Module):
    """
    Discriminator that classifies timeseries samples as real or fake
    using a phasor circuit.

    Architecture:
        x → arcsin(x) → [Shift → Mix → DFT] × num_layers → P(real)

    Args:
        seq_length: Length of input timeseries (= number of threads).
        num_layers: Number of variational layers.
    """

    def __init__(self, seq_length, num_layers=3):
        super().__init__()
        self.seq_length = seq_length
        self.num_layers = num_layers
        self._engine = AnalyticEngine()

        total_params = num_layers * seq_length
        self.weights = nn.Parameter(
            torch.empty(total_params).uniform_(-math.pi / 4, math.pi / 4)
        )

    def _build_circuit(self, x_phases):
        """Build discriminator phasor circuit."""
        T = self.seq_length
        pc = PhasorCircuit(T, name="Discriminator")

        for i in range(T):
            pc.shift(i, x_phases[i])

        for layer in range(self.num_layers):
            offset = layer * T
            for i in range(T):
                pc.shift(i, self.weights[offset + i])
            for i in range(0, T - 1, 2):
                pc.mix(i, i + 1)
            pc.dft()

        return pc

    def forward(self, x):
        """
        Classify a single timeseries sample.

        Args:
            x: Timeseries tensor, shape (seq_length,), values in [-1, 1].

        Returns:
            Probability of being real (scalar in [0, 1]).
        """
        x_clamped = torch.clamp(x, -0.999, 0.999)
        x_phases = torch.asin(x_clamped)

        pc = self._build_circuit(x_phases)
        result = self._engine.run(pc)
        return (torch.sin(result['phases'][0]) + 1.0) / 2.0

    def forward_batch(self, X):
        """Classify a batch of samples."""
        return torch.stack([self.forward(X[i]) for i in range(X.shape[0])])


class PhasorGAN(nn.Module):
    """
    Phasor GAN — High-level GAN for timeseries generation.

    Wraps PhasorGenerator and PhasorDiscriminator with a
    scikit-learn-style API (fit / generate).

    Args:
        seq_length: Length of timeseries (= number of phasor threads).
        num_layers_g: Generator circuit layers.
        num_layers_d: Discriminator circuit layers.
    """

    def __init__(
        self,
        seq_length: int,
        num_layers_g: int = 2,
        num_layers_d: int = 2,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_layers_g = num_layers_g
        self.num_layers_d = num_layers_d

        self.generator = PhasorGenerator(seq_length, num_layers=num_layers_g)
        self.discriminator = PhasorDiscriminator(seq_length, num_layers=num_layers_d)

    def fit(
        self,
        X: torch.Tensor,
        epochs: int = 60,
        batch_size: int = 10,
        lr_g: float = 0.01,
        lr_d: float = 0.01,
        verbose: bool = True,
        print_every: int = 10,
    ) -> dict:
        """
        Train the GAN adversarially on real timeseries data.

        Args:
            X: Real timeseries data, shape (n_samples, seq_length).
               Values should be in [-1, 1].
            epochs: Number of training epochs.
            batch_size: Samples per mini-batch.
            lr_g: Generator learning rate.
            lr_d: Discriminator learning rate.
            verbose: Print training progress.
            print_every: Print interval.

        Returns:
            dict with 'd_loss', 'g_loss', 'd_real', 'd_fake' histories.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        n_real = X.shape[0]
        opt_g = optim.Adam(self.generator.parameters(), lr=lr_g)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr_d)

        history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': []}

        if verbose:
            print(f"Phasor GAN Training")
            print(f"  Generator params:     {self.generator.weights.numel()}")
            print(f"  Discriminator params: {self.discriminator.weights.numel()}")
            print(f"  Real samples:         {n_real}")
            print(f"  Seq length:           {self.seq_length}")
            print()

        for epoch in range(epochs):
            # Sample real mini-batch
            idx = torch.randint(0, n_real, (batch_size,))
            real_batch = X[idx]

            # Generate fake mini-batch
            fake_batch = self.generator.generate_batch(batch_size)

            # --- Train Discriminator ---
            opt_d.zero_grad()
            d_real_out = self.discriminator.forward_batch(real_batch)
            d_loss_real = -torch.mean(torch.log(d_real_out + 1e-8))
            d_fake_out = self.discriminator.forward_batch(fake_batch.detach())
            d_loss_fake = -torch.mean(torch.log(1.0 - d_fake_out + 1e-8))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # --- Train Generator ---
            opt_g.zero_grad()
            fake_batch_g = self.generator.generate_batch(batch_size)
            d_fake_for_g = self.discriminator.forward_batch(fake_batch_g)
            g_loss = -torch.mean(torch.log(d_fake_for_g + 1e-8))
            g_loss.backward()
            opt_g.step()

            # Record
            history['d_loss'].append(d_loss.item())
            history['g_loss'].append(g_loss.item())
            history['d_real'].append(d_real_out.mean().item())
            history['d_fake'].append(d_fake_out.mean().item())

            if verbose and (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch+1:4d}/{epochs} | "
                    f"D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | "
                    f"D(real): {d_real_out.mean().item():.3f} | "
                    f"D(fake): {d_fake_out.mean().item():.3f}"
                )

        if verbose:
            print(f"\nTraining complete!")

        return history

    def generate(self, num_samples: int = 10) -> torch.Tensor:
        """
        Generate synthetic timeseries samples.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, seq_length).
        """
        with torch.no_grad():
            return self.generator.generate_batch(num_samples)

    def __repr__(self) -> str:
        return (
            f"PhasorGAN(\n"
            f"  seq_length={self.seq_length},\n"
            f"  generator_layers={self.num_layers_g},\n"
            f"  discriminator_layers={self.num_layers_d},\n"
            f"  generator_params={self.generator.weights.numel()},\n"
            f"  discriminator_params={self.discriminator.weights.numel()}\n"
            f")"
        )
