# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# examples/gan.py

"""
Phasor GAN — Generative Adversarial Network for Timeseries Data.

A GAN where both the Generator and Discriminator are phasor circuits,
trained adversarially on synthetic composite waveform timeseries data.

Generator:
    Random noise z ∈ [-π, π]^T  →  PhasorCircuit (shifts + DFT)  →  sin(phases) → fake sample

Discriminator:
    Timeseries x ∈ ℝ^T  →  phase-encode  →  PhasorCircuit (shifts + mix + DFT)  →  P(real)

Usage:
    cd PhasorFlow/
    python examples/gan.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import PhasorFlow as pf
from PhasorFlow.circuit import PhasorCircuit
from PhasorFlow.engine.analytic import AnalyticEngine


# ======================================================================
# Data Generation
# ======================================================================

def generate_timeseries_data(num_samples, seq_length, seed=42):
    """
    Generate synthetic composite waveform timeseries data.

    Signal: sin(t) + 0.5*cos(3t) + 0.3*sin(5t) + noise

    Returns:
        X: np.ndarray of shape (num_samples, seq_length), values roughly in [-1, 1]
    """
    np.random.seed(seed)
    t_total = np.linspace(0, 8 * np.pi, num_samples + seq_length)
    signal = (
        np.sin(t_total)
        + 0.5 * np.cos(3 * t_total)
        + 0.3 * np.sin(5 * t_total)
    )
    # Normalize to [-1, 1]
    signal = signal / np.max(np.abs(signal))
    # Add small noise
    signal += np.random.normal(0, 0.02, len(signal))
    signal = np.clip(signal, -0.99, 0.99)

    # Window into samples
    X = []
    for i in range(num_samples):
        X.append(signal[i : i + seq_length])
    return np.array(X, dtype=np.float32)


# ======================================================================
# Generator — Phasor Circuit
# ======================================================================

class PhasorGenerator(nn.Module):
    """
    Generator that maps random latent noise through a phasor circuit
    to produce synthetic timeseries samples.

    Architecture:
        z (random phases) → [Shift → DFT] × num_layers → sin(output_phases) → sample

    Args:
        seq_length: Length of generated timeseries (= number of threads).
        num_layers: Number of variational layers in the circuit.
    """

    def __init__(self, seq_length, num_layers=3):
        super().__init__()
        self.seq_length = seq_length
        self.num_layers = num_layers
        self._engine = AnalyticEngine()

        # Trainable parameters: one shift per thread per layer
        total_params = num_layers * seq_length
        self.weights = nn.Parameter(
            torch.empty(total_params).uniform_(-math.pi / 4, math.pi / 4)
        )

    def _build_circuit(self, z):
        """Build generator phasor circuit from latent noise z."""
        T = self.seq_length
        pc = PhasorCircuit(T, name="Generator")

        # Encode latent noise as initial phases
        for i in range(T):
            pc.shift(i, z[i])

        # Variational layers: shift + DFT
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
            Generated sample tensor, shape (seq_length,), values roughly in [-1, 1].
        """
        pc = self._build_circuit(z)
        result = self._engine.run(pc)
        # Map output phases to signal values via sin()
        return torch.sin(result['phases'])

    def generate_batch(self, batch_size):
        """Generate a batch of fake samples."""
        z_batch = torch.empty(batch_size, self.seq_length).uniform_(-math.pi, math.pi)
        return torch.stack([self.forward(z_batch[i]) for i in range(batch_size)])


# ======================================================================
# Discriminator — Phasor Circuit
# ======================================================================

class PhasorDiscriminator(nn.Module):
    """
    Discriminator that classifies timeseries samples as real or fake
    using a phasor circuit.

    Architecture:
        x → arcsin(x) phase-encode → [Shift → Mix → DFT] × num_layers → P(real)

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

        # Encode input signal as phases
        for i in range(T):
            pc.shift(i, x_phases[i])

        # Variational layers: shift + mix pairs + DFT
        for layer in range(self.num_layers):
            offset = layer * T
            for i in range(T):
                pc.shift(i, self.weights[offset + i])
            # Alternating mix pairs for entanglement
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
        # Phase-encode the signal: x ∈ [-1,1] → θ ∈ [-π/2, π/2]
        x_clamped = torch.clamp(x, -0.999, 0.999)
        x_phases = torch.asin(x_clamped)

        pc = self._build_circuit(x_phases)
        result = self._engine.run(pc)

        # Readout: thread-0 phase → [0, 1] probability via sine envelope
        return (torch.sin(result['phases'][0]) + 1.0) / 2.0

    def forward_batch(self, X):
        """Classify a batch of samples."""
        return torch.stack([self.forward(X[i]) for i in range(X.shape[0])])


# ======================================================================
# GAN Training
# ======================================================================

def train_phasor_gan(
    real_data,
    seq_length=8,
    num_layers_g=3,
    num_layers_d=3,
    epochs=100,
    batch_size=16,
    lr_g=0.01,
    lr_d=0.01,
    print_every=10,
):
    """
    Train a Phasor GAN on timeseries data.

    Args:
        real_data: np.ndarray of real samples, shape (n_samples, seq_length).
        seq_length: Timeseries / thread count.
        num_layers_g: Generator circuit layers.
        num_layers_d: Discriminator circuit layers.
        epochs: Training epochs.
        batch_size: Samples per batch.
        lr_g: Generator learning rate.
        lr_d: Discriminator learning rate.
        print_every: Print interval.

    Returns:
        generator, discriminator, training_history
    """
    # Convert data
    real_tensor = torch.tensor(real_data, dtype=torch.float32)
    n_real = real_tensor.shape[0]

    # Initialize models
    generator = PhasorGenerator(seq_length, num_layers=num_layers_g)
    discriminator = PhasorDiscriminator(seq_length, num_layers=num_layers_d)

    opt_g = optim.Adam(generator.parameters(), lr=lr_g)
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d)

    history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': []}

    print(f"Phasor GAN Training")
    print(f"  Generator params:     {generator.weights.numel()}")
    print(f"  Discriminator params: {discriminator.weights.numel()}")
    print(f"  Real samples:         {n_real}")
    print(f"  Seq length:           {seq_length}")
    print()

    for epoch in range(epochs):

        # --- Sample mini-batch of real data ---
        idx = torch.randint(0, n_real, (batch_size,))
        real_batch = real_tensor[idx]

        # --- Generate fake data ---
        fake_batch = generator.generate_batch(batch_size)

        # =============================================
        # Train Discriminator
        # =============================================
        opt_d.zero_grad()

        # D(real) should be close to 1
        d_real_out = discriminator.forward_batch(real_batch)
        d_loss_real = -torch.mean(torch.log(d_real_out + 1e-8))

        # D(fake) should be close to 0
        d_fake_out = discriminator.forward_batch(fake_batch.detach())
        d_loss_fake = -torch.mean(torch.log(1.0 - d_fake_out + 1e-8))

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_d.step()

        # =============================================
        # Train Generator
        # =============================================
        opt_g.zero_grad()

        # Re-generate (need fresh computation graph)
        fake_batch_g = generator.generate_batch(batch_size)
        d_fake_for_g = discriminator.forward_batch(fake_batch_g)

        # G wants D(G(z)) → 1
        g_loss = -torch.mean(torch.log(d_fake_for_g + 1e-8))
        g_loss.backward()
        opt_g.step()

        # Record
        history['d_loss'].append(d_loss.item())
        history['g_loss'].append(g_loss.item())
        history['d_real'].append(d_real_out.mean().item())
        history['d_fake'].append(d_fake_out.mean().item())

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | "
                f"D(real): {d_real_out.mean().item():.3f} | "
                f"D(fake): {d_fake_out.mean().item():.3f}"
            )

    print("\nTraining complete!")
    return generator, discriminator, history


# ======================================================================
# Visualization
# ======================================================================

def visualize_results(real_data, generator, history, num_samples=5):
    """Plot real vs generated samples and training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Real vs Generated Samples
    ax = axes[0]
    for i in range(num_samples):
        ax.plot(real_data[i], color='tab:blue', alpha=0.5, linewidth=1.5)
    with torch.no_grad():
        fake = generator.generate_batch(num_samples)
    for i in range(num_samples):
        ax.plot(fake[i].numpy(), color='tab:red', alpha=0.5, linewidth=1.5, linestyle='--')
    ax.set_title('Real (blue) vs Generated (red) Timeseries')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Amplitude')
    ax.legend(['Real', '', '', '', '', 'Generated'], loc='upper right')

    # 2. D & G Loss Curves
    ax = axes[1]
    ax.plot(history['d_loss'], label='D Loss', color='tab:blue')
    ax.plot(history['g_loss'], label='G Loss', color='tab:red')
    ax.set_title('GAN Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # 3. D(real) and D(fake) over time
    ax = axes[2]
    ax.plot(history['d_real'], label='D(real)', color='tab:green')
    ax.plot(history['d_fake'], label='D(fake)', color='tab:orange')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Discriminator Confidence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Probability')
    ax.legend()

    plt.tight_layout()
    plt.savefig('phasor_gan_results.png', dpi=150)
    print("Results saved to phasor_gan_results.png")
    plt.show()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    SEQ_LENGTH = 8
    NUM_SAMPLES = 100
    EPOCHS = 60
    BATCH_SIZE = 10

    print("=" * 60)
    print("  Phasor GAN — Timeseries Generation Experiment")
    print("=" * 60)
    print()

    # 1. Generate real data
    print("Generating synthetic timeseries data...")
    real_data = generate_timeseries_data(NUM_SAMPLES, SEQ_LENGTH)
    print(f"  Shape: {real_data.shape}")
    print(f"  Range: [{real_data.min():.3f}, {real_data.max():.3f}]")
    print()

    # 2. Train the GAN
    generator, discriminator, history = train_phasor_gan(
        real_data,
        seq_length=SEQ_LENGTH,
        num_layers_g=2,
        num_layers_d=2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr_g=0.01,
        lr_d=0.01,
        print_every=10,
    )

    # 3. Visualize
    print()
    visualize_results(real_data, generator, history)

    # 4. Summary statistics
    print("\n--- Final Statistics ---")
    print(f"Generator circuit:     {generator.num_layers} layers × {SEQ_LENGTH} threads = {generator.weights.numel()} params")
    print(f"Discriminator circuit: {discriminator.num_layers} layers × {SEQ_LENGTH} threads = {discriminator.weights.numel()} params")
    print(f"Final D(real): {history['d_real'][-1]:.4f}")
    print(f"Final D(fake): {history['d_fake'][-1]:.4f}")
