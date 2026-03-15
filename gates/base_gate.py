# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/gates/base_gate.py

from abc import ABC, abstractmethod
import torch

class BaseGate(ABC):
    def __init__(self, name: str, num_threads: int):
        """
        Abstract base class for all PhasorFlow gates.
        
        Args:
            name: The display name of the gate.
            num_threads: The number of threads the gate acts upon.
        """
        self.name = name
        self.num_threads = num_threads

    @abstractmethod
    def get_matrix(self, **kwargs) -> torch.Tensor:
        """
        Returns the unitary matrix representation of the gate.
        """
        pass

    def __repr__(self):
        return f"Gate(name='{self.name}', threads={self.num_threads})"
