"""Visualisation utilities for AOPP experiments.

Generates publication-quality figures for the RA-L paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import List, Tuple, Optional, Dict
from pathlib import Path


# Consistent colour scheme for the paper
COLORS = {
    'baseline_path': '#2166AC',   # blue
    'attacked_path': '#B2182B',   # red
    'obstacle': '#1A1A1A',        # near-black
    'adv_object': '#E69F00',      # amber
    'free': '#F7F7F7',            # off-white
    'occupied': '#404040',        # dark grey
    'unknown': '#CCCCCC',         # light grey
    'start': '#4DAF4A',           # green
    'goal': '#984EA3',            # purple
}


def plot_map_with_paths(
    grid: np.ndarray,
    baseline_path: Optional[List[Tuple[int, int]]] = None,
    attacked_path: Optional[List[Tuple[int, int]]] = None,
    adv_placements: Optional[List[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    obj_w: int = 3,
    obj_h: int = 3,
    title: str = '',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
):
    """Plot occupancy grid with paths and adversarial objects.

    This produces figures suitable for Fig. 2 of the paper.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Map background
    display = np.ones((*grid.shape, 3))
    free_mask = grid == 0
    occ_mask = grid >= 100
    unk_mask = ~free_mask & ~occ_mask

    for i, c in enumerate(
        [int(x, 16) / 255 for x in [
            COLORS['free'][i:i+2] for i in (1, 3, 5)
        ]]
    ):
        display[:, :, i] = np.where(free_mask, c, display[:, :, i])

    hex_occ = COLORS['occupied'].lstrip('#')
    occ_rgb = [int(hex_occ[i:i+2], 16) / 255 for i in (0, 2, 4)]
    for i in range(3):
        display[:, :, i] = np.where(occ_mask, occ_rgb[i], display[:, :, i])

    hex_unk = COLORS['unknown'].lstrip('#')
    unk_rgb = [int(hex_unk[i:i+2], 16) / 255 for i in (0, 2, 4)]
    for i in range(3):
        display[:, :, i] = np.where(unk_mask, unk_rgb[i], display[:, :, i])

    ax.imshow(display, origin='upper')

    # Adversarial objects
    if adv_placements:
        for (r, c) in adv_placements:
            rect = mpatches.Rectangle(
                (c - 0.5, r - 0.5), obj_w, obj_h,
                linewidth=1.5, edgecolor=COLORS['adv_object'],
                facecolor=COLORS['adv_object'], alpha=0.8,
            )
            ax.add_patch(rect)

    # Paths
    if baseline_path:
        rows = [p[0] for p in baseline_path]
        cols = [p[1] for p in baseline_path]
        ax.plot(cols, rows, color=COLORS['baseline_path'],
                linewidth=2.0, label='Baseline path', zorder=3)

    if attacked_path:
        rows = [p[0] for p in attacked_path]
        cols = [p[1] for p in attacked_path]
        ax.plot(cols, rows, color=COLORS['attacked_path'],
                linewidth=2.0, linestyle='--',
                label='Path under attack', zorder=3)

    # Start and goal
    if start:
        ax.plot(start[1], start[0], 'o', color=COLORS['start'],
                markersize=10, zorder=5, label='Start')
    if goal:
        ax.plot(goal[1], goal[0], '*', color=COLORS['goal'],
                markersize=14, zorder=5, label='Goal')

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Column (cells)')
    ax.set_ylabel('Row (cells)')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_delay_comparison(
    results: List[Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
):
    """Bar chart comparing delay across conditions (Fig. 3)."""
    envs = sorted(set(r['env'] for r in results))
    budgets = sorted(set(r['k'] for r in results))

    fig, axes = plt.subplots(1, len(envs), figsize=figsize, sharey=True)
    if len(envs) == 1:
        axes = [axes]

    bar_colors = {
        'random': '#A6CEE3',
        'pathblock': '#FB9A99',
        'greedy': '#E31A1C',
    }

    for ax_idx, env in enumerate(envs):
        ax = axes[ax_idx]
        env_results = [r for r in results if r['env'] == env]
        env_results.sort(key=lambda r: r['k'])

        x = np.arange(len(env_results))
        width = 0.25

        random_vals = [r['random_delay_mean'] for r in env_results]
        pb_vals = [r['pathblock_delay'] for r in env_results]
        greedy_vals = [r['greedy_delay'] for r in env_results]
        random_errs = [r['random_delay_std'] for r in env_results]

        ax.bar(x - width, random_vals, width, yerr=random_errs,
               label='Random', color=bar_colors['random'], capsize=3)
        ax.bar(x, pb_vals, width,
               label='Path-block', color=bar_colors['pathblock'])
        ax.bar(x + width, greedy_vals, width,
               label='Greedy (ours)', color=bar_colors['greedy'])

        ax.set_xticks(x)
        ax.set_xticklabels([f"k={r['k']}" for r in env_results])
        ax.set_title(env.capitalize(), fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel('Navigation delay (cells)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_marginal_gains(
    env_gains: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
):
    """Line plot of marginal gains vs object index (Fig. 4)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    markers = ['o', 's', '^', 'D']
    for i, (env, gains) in enumerate(env_gains.items()):
        ax.plot(
            range(1, len(gains) + 1), gains,
            marker=markers[i % len(markers)],
            label=env.capitalize(), linewidth=1.5, markersize=6,
        )

    ax.set_xlabel('Object index $i$')
    ax.set_ylabel('Marginal gain $\\Delta f_i$ (cells)')
    ax.set_title('Marginal gains (submodularity evidence)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig
