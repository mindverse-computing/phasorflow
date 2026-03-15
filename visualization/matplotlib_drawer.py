# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/visualization/matplotlib_drawer.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MatplotlibDrawer:
    def __init__(self, circuit):
        self.circuit = circuit
        self.fig, self.ax = plt.subplots(figsize=(len(circuit.data) + 2, circuit.num_threads + 1))
        self._setup_canvas()

    def _setup_canvas(self):
        """Sets up the grid and labels for the threads."""
        self.ax.set_ylim(-1, self.circuit.num_threads)
        self.ax.set_xlim(-0.5, len(self.circuit.data) + 0.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Draw the horizontal thread lines
        for i in range(self.circuit.num_threads):
            self.ax.text(-0.8, i, f"T{i}", va='center', ha='right', fontsize=12, fontweight='bold')
            self.ax.plot([-0.5, len(self.circuit.data) + 0.5], [i, i], color='black', lw=1.5, zorder=1)

    def draw(self):
        """Iterates through circuit data and draws the gates."""
        for x, (gate_name, targets, params) in enumerate(self.circuit.data):
            if gate_name == 'shift':
                self._draw_box(x, targets[0], f"S\n({params['phi']:.2f})", color='#6495ED')
            elif gate_name == 'mix':
                # Draw a vertical link between the two threads
                self.ax.plot([x, x], [targets[0], targets[1]], color='#FF6347', lw=3, zorder=2)
                self._draw_box(x, targets[0], "M", color='#FF6347')
                self._draw_box(x, targets[1], "M", color='#FF6347')
            elif gate_name == 'dft':
                # Draw a large box spanning all targets
                y_min, y_max = min(targets), max(targets)
                rect = patches.Rectangle((x - 0.35, y_min - 0.35), 0.7, (y_max - y_min) + 0.7, 
                                         facecolor='#9370DB', edgecolor='black', zorder=3)
                self.ax.add_patch(rect)
                self.ax.text(x, (y_min + y_max) / 2, "DFT", ha='center', va='center', color='white', fontweight='bold')

        return self.fig

    def _draw_box(self, x, y, label, color):
        """Helper to draw a standardized gate box."""
        rect = patches.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, facecolor=color, edgecolor='black', zorder=3)
        self.ax.add_patch(rect)
        self.ax.text(x, y, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
