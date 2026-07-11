# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/engine/__init__.py

from .analytic import AnalyticEngine


class Simulator:
    """
    High-level simulator interface for PhasorFlow circuits.

    Usage (fluent/notebook style)::

        sim = Simulator()
        out = sim.run(circuit)   # returns complex state_vector tensor
        amp = out.abs()          # amplitudes
        phi = out.angle()        # phases

    The returned tensor is a standard ``torch.Tensor`` (complex64) so all
    PyTorch operations work on it directly.
    """

    def __init__(self, backend: str = 'analytic'):
        """
        Args:
            backend: Which simulation backend to use.
                     Currently only ``'analytic'`` is supported.
        """
        self._engine = AnalyticEngine()

    def run(self, circuit, initial_state=None):
        """
        Execute a PhasorCircuit and return the final state as a complex tensor.

        Args:
            circuit:       A :class:`~phasorflow.PhasorCircuit` instance.
            initial_state: Optional initial state tensor (complex64, shape n).
                           If ``None``, all threads start at z = 1+0j.

        Returns:
            torch.Tensor: Complex64 tensor of shape (num_threads,) representing
            the final phasor state vector.  Take ``.abs()`` for amplitudes,
            ``.angle()`` for phases.
        """
        result = self._engine.run(circuit, initial_state=initial_state)
        return result['state_vector']

    def run_full(self, circuit, initial_state=None):
        """
        Execute a PhasorCircuit and return the full result dict.

        Returns:
            dict with keys:
            - ``'state_vector'`` – complex64 tensor
            - ``'phases'``       – float tensor of phase angles
            - ``'amplitudes'``   – float tensor of amplitudes
            - ``'measurements'`` – dict of labelled measurement snapshots
        """
        return self._engine.run(circuit, initial_state=initial_state)

    @staticmethod
    def get_backend(name: str):
        """Legacy factory method."""
        if name == 'analytic_simulator':
            return AnalyticEngine()
        else:
            raise ValueError(f"Backend '{name}' not recognized.")
