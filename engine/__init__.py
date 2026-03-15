# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/engine/__init__.py

from .analytic import AnalyticEngine

class Simulator:
    @staticmethod
    def get_backend(name: str):
        if name == 'analytic_simulator':
            return AnalyticEngine()
        else:
            raise ValueError(f"Backend '{name}' not recognized.")
