"""
Visualization utilities for FoveatedGraphMemory experiments (improved clarity).

Key improvements:
- High-DPI rendering and no rescaling of the original image (pixel-perfect display)
- Use of imshow(interpolation='nearest') to avoid blur
- Explicit legend outside the image explaining colors/symbols
- Attention overlay normalized with gamma correction to avoid gray wash
"""

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image
import torch

# ----------------------------
# Utilities
# ----------------------------

def _to_numpy_image(img: Any) -> np.ndarray:
    """Convert input to uint8 HxWx3 numpy array without implicit smoothing."""
    if isinstance(img, np.ndarray):
        arr = img
    elif hasattr(img, 'detach'):
        t = img.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = np.transpose(t.numpy(), (1, 2, 0))
        else:
            t = t.numpy()
        arr = t
    else:
        arr = np.array(img)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.dtype in (np.float32, np.float64):
        # assume 0..1
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _normalize_attention(att: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Normalize and apply gamma correction to avoid gray overlay."""
    att = np.clip(att, 0.0, 1.0)
    att = att ** gamma
    return att

# ----------------------------
# Drawing helpers
# ----------------------------

def overlay_attention(ax, attn_map: torch.Tensor, img_shape: Tuple[int, int], alpha: float = 0.35):
    if isinstance(attn_map, torch.Tensor):
        att = attn_map.detach().cpu().numpy()
    else:
        att = np.array(attn_map)

    H, W = img_shape
    if att.shape != (H, W):
        att = np.array(
            Image.fromarray((att * 255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
        ).astype(np.float32) / 255.0

    att = _normalize_attention(att)
    ax.imshow(att, cmap='Reds', alpha=alpha, interpolation='nearest')


def draw_graph0_nodes(ax, mem, node_ids=None, color='lightgray', size=12, alpha=0.8):
    xs, ys = [], []
    if node_ids is None:
        nodes = mem.graph0.nodes.values()
    else:
        nodes = (mem.graph0.nodes[nid] for nid in node_ids if nid in mem.graph0.nodes)

    for n in nodes:
        xs.append(n.pos[0])
        ys.append(n.pos[1])

    ax.scatter(xs, ys, s=size, c=color, alpha=alpha, edgecolors='black', linewidths=0.4, zorder=3)


def draw_graph0_edges(ax, mem, highlight_node_set=None):
    segs, colors, widths = [], [], []
    for e in mem.graph0.edges.values():
        n1 = mem.graph0.nodes.get(e.src)
        n2 = mem.graph0.nodes.get(e.dst)
        if n1 is None or n2 is None:
            continue
        segs.append([(n1.pos[0], n1.pos[1]), (n2.pos[0], n2.pos[1])])
        if highlight_node_set and e.src in highlight_node_set and e.dst in highlight_node_set:
            colors.append((1.0, 0.0, 0.0, 0.9))
            widths.append(2.5)
        else:
            w = float(getattr(e, 'weight', 1.0))
            widths.append(0.5 + min(2.0, w))
            colors.append((0.3, 0.3, 0.3, 0.25))

    if segs:
        lc = LineCollection(segs, colors=colors, linewidths=widths, zorder=2)
        ax.add_collection(lc)


def draw_labeled_nodes(ax, mem, node_ids, facecolor, radius=2):
    for nid in node_ids:
        if nid not in mem.graph0.nodes:
            continue
        n = mem.graph0.nodes[nid]
        c = Circle((n.pos[0], n.pos[1]), radius=radius, facecolor=facecolor,
                   edgecolor='black', linewidth=0.8, zorder=4)
        ax.add_patch(c)


def draw_saccade_arrow(ax, start, end):
    arr = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=18,
                            color='cyan', linewidth=2.5, zorder=5)
    ax.add_patch(arr)

# ----------------------------
# Main API
# ----------------------------

def visualize_step(image: Any,
                   mem: Any,
                   result: Dict,
                   current_fix: Optional[Tuple[float, float]] = None,
                   figsize: Tuple[int, int] = (8, 8),
                   save_path: Optional[str] = None,
                   show: bool = True):

    img = _to_numpy_image(image)
    H, W = img.shape[:2]

    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_axes([0.05, 0.05, 0.7, 0.9])  # leave space for legend

    ax.imshow(img, interpolation='nearest')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('on')

    # Attention
    if hasattr(mem, 'attn'):
        overlay_attention(ax, mem.attn.get_map(), (H, W))

    # Graph edges and nodes
    draw_graph0_edges(ax, mem)
    draw_graph0_nodes(ax, mem)

    match_info = result.get('match_info', {}) if result else {}
    if match_info is not None:
        matched_nodes = match_info.get('matched_mem_ids', [])
        if matched_nodes:
            draw_labeled_nodes(ax, mem, matched_nodes, facecolor='lime', radius=2)
            draw_graph0_edges(ax, mem, highlight_node_set=set(matched_nodes))
        
    learned_nodes = result.get('learned_nodes', []) if result else []
    
    if learned_nodes:
        draw_labeled_nodes(ax, mem, learned_nodes, facecolor='dodgerblue', radius=2)

    # Fixation and saccade
    if current_fix is not None:
        ax.plot(current_fix[0], current_fix[1], marker='+', color='yellow', markersize=2, mew=2)

    mv = result.get('move') if result else None
    if mv and len(mv) >= 3 and current_fix is not None:
        target = mv[2]
        draw_saccade_arrow(ax, current_fix, target)
        ax.plot(target[0], target[1], marker='o', color='magenta', markersize=2)

    # Legend
    legend_ax = fig.add_axes([0.78, 0.1, 0.2, 0.8])
    legend_ax.axis('off')
    legend_items = [
        Patch(facecolor='lime', edgecolor='black', label='Matched nodes'),
        Patch(facecolor='dodgerblue', edgecolor='black', label='Newly learned nodes'),
        Line2D([0], [0], color='red', lw=2.5, label='Reinforced edges'),
        Line2D([0], [0], color='gray', lw=0.1, alpha=0.4, label='Existing edges'),
        Line2D([0], [0], marker='+', color='yellow', lw=0, markersize=12, label='Current fixation'),
        Line2D([0], [0], marker='o', color='magenta', lw=0, markersize=8, label='Saccade target'),
        Line2D([0], [0], color='cyan', lw=2.5, label='Saccade path'),
        Patch(facecolor='red', alpha=0.35, label='Attention map')
    ]
    legend_ax.legend(handles=legend_items, loc='upper left', framealpha=0.9)

    title = f"matched={result.get('matched', False)}"
    ax.set_title(title, fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
