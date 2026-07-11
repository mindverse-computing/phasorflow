"""
Shared publication-quality figure style for all PhasorFlow figures.

Rationale: figures were previously generated at 12x8 inches with 10 pt fonts and
then downscaled to \\textwidth (~6.5 in) in the manuscripts, which shrank the
effective font size to ~5 pt. This module fixes a single consistent style with
large fonts and print-appropriate figure sizes, so that axis names, tick labels,
and legends remain legible at the size the figure is actually printed.

Usage:
    from phasorflow.visualization.pubstyle import apply_pub_style, FIGSIZE
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["multipanel"],
                             constrained_layout=True)
"""

import matplotlib as mpl

# Standard figure sizes (inches). Multi-panel widths (9.5 in) exceed a
# single-column \textwidth (~6.5 in), so \includegraphics[width=\textwidth]{...}
# downscales them by ~0.68x; the font sizes below are deliberately enlarged
# (see _PUB_RC) so that they still print at >= ~9 pt after that downscale. The
# 'single' size (7.0 in) is close to \textwidth and prints at near-nominal size.
FIGSIZE = {
    "multipanel": (9.5, 6.6),   # 2x2 grids
    "wide2":      (9.5, 4.0),   # 1x2 wide panels
    "single":     (7.0, 4.4),   # single panel
    "tall3":      (9.5, 8.4),   # 3-row stacks
}

# Font sizes are deliberately large so that after any modest downscale to
# \textwidth they remain >= ~9 pt.
_PUB_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "legend.fontsize": 13,
    "legend.frameon": False,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
}


def apply_pub_style():
    """Apply the shared publication rcParams globally."""
    mpl.rcParams.update(_PUB_RC)
    return _PUB_RC


# A small, colour-blind-safe categorical palette used across figures.
PALETTE = {
    "phasor":   "#1f77b4",   # blue  -- the phasor/VPC model (focal)
    "baseline": "#ff7f0e",   # orange
    "accent2":  "#2ca02c",   # green
    "accent3":  "#9467bd",   # purple
    "neutral":  "#7f7f7f",   # grey
    "alarm":    "#d62728",   # red -- reserved for error/reference only
}
