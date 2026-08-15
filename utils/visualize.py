# Overlay, comparison plots
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def heatmap_to_overlay(img_pil: Image.Image,
                       heatmap: np.ndarray,
                       alpha: float = 0.45) -> np.ndarray:
    """
    Blend a normalized heatmap onto the original image.

    Args:
        img_pil:  PIL image (H, W, 3), already resized to match heatmap
        heatmap:  numpy array (H, W) with values in [0, 1]
        alpha:    heatmap opacity. 0 = invisible, 1 = no original image.

    Returns:
        overlay:  numpy array (H, W, 3) in [0, 1], ready for imshow()

    How the blending works:
        pixel = (1 - alpha) * original_pixel + alpha * colormap_pixel
    At alpha=0.45, you see ~55% original image, 45% colored heatmap.
    """
    img_np = np.array(img_pil, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]

    # 'jet' colormap: maps scalar [0,1] → RGB color
    # Low activation (0.0) → blue, high activation (1.0) → red
    colormap = cm.get_cmap('jet')
    heatmap_rgb = colormap(heatmap)[..., :3]               # (H, W, 4) → (H, W, 3)

    overlay = (1 - alpha) * img_np + alpha * heatmap_rgb
    return overlay.clip(0, 1)


def show_result(img_pil: Image.Image,
                heatmap: np.ndarray,
                class_name: str,
                method_name: str,
                top_k_scores: list = None,
                save_path: str = None):
    """
    Display: [original] [raw heatmap] [overlay] side by side.
    Optionally show top-k predictions and save the figure.

    Args:
        img_pil:       Original PIL image
        heatmap:       Normalized (H, W) numpy array [0, 1]
        class_name:    Predicted class name string
        method_name:   e.g. 'CAM', 'Grad-CAM'
        top_k_scores:  Optional list of (class_name, score) tuples for top-5
        save_path:     If provided, save figure to this path
    """
    overlay = heatmap_to_overlay(img_pil, heatmap)
    if top_k_scores:
        fig = plt.figure(figsize=(16, 4))
        ax_orig    = fig.add_subplot(1, 4, 1)
        ax_heatmap = fig.add_subplot(1, 4, 2)
        ax_overlay = fig.add_subplot(1, 4, 3)
        ax_scores  = fig.add_subplot(1, 4, 4)
    else:
        fig, (ax_orig, ax_heatmap, ax_overlay) = plt.subplots(1, 3, figsize=(12, 4))

    fig.suptitle(f'{method_name}  ·  Predicted: "{class_name}"',
                 fontsize=13, fontweight='bold', y=1.01)

    # Original image
    ax_orig.imshow(img_pil)
    ax_orig.set_title('Original', fontsize=11)
    ax_orig.axis('off')

    # Raw heatmap with colorbar
    im = ax_heatmap.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    ax_heatmap.set_title('Heatmap (7×7 → upsampled)', fontsize=11)
    ax_heatmap.axis('off')
    plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)

    # Overlay
    ax_overlay.imshow(overlay)
    ax_overlay.set_title('Overlay', fontsize=11)
    ax_overlay.axis('off')

    # Top-k bar chart
    if top_k_scores:
        names  = [s[0][:20] for s in top_k_scores]   # truncate long names
        scores = [s[1] for s in top_k_scores]
        bars = ax_scores.barh(names[::-1], scores[::-1], color='steelblue')
        ax_scores.set_xlabel('Softmax probability')
        ax_scores.set_title('Top-5 Predictions', fontsize=11)
        ax_scores.set_xlim(0, 1)
        # Annotate bars with values
        for bar, score in zip(bars, scores[::-1]):
            ax_scores.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                           f'{score:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")

    plt.show()
    plt.close(fig)
    
def _add_topk_bar(ax, top_k_scores: list, highlight_name: str = None):
    """
    Horizontal bar chart of top-k softmax scores.
    Highlights the predicted class bar in tomato red, others in steelblue.
    Extracted as a helper so both show_result and compare_methods can use it
    without duplicating the bar-drawing logic.
    """
    names  = [s[0][:22] for s in top_k_scores]
    scores = [s[1]       for s in top_k_scores]

    colors = [
        'tomato' if n.strip() == (highlight_name or '').strip() else 'steelblue'
        for n in names
    ]

    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel('Softmax probability', fontsize=9)
    ax.set_title('Top-5 Predictions', fontsize=10)
    ax.set_xlim(0, max(scores) * 1.25)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            score + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'{score:.3f}',
            va='center', fontsize=8
        )


def compare_methods(img_pil: Image.Image,
                    heatmaps: dict,
                    class_name: str,
                    top_k_scores: list = None,
                    save_path: str = None):
    """
    Side-by-side overlay comparison for multiple CAM methods.

    Layout:
        [Original] [Method-1 overlay] [Method-2 overlay] ... [Top-k bar]

    Args:
        img_pil:      Original PIL image.
        heatmaps:     Ordered dict of {method_display_name: heatmap_array}.
                      e.g. {'CAM': arr1, 'Grad-CAM': arr2}
                      Column order follows key insertion order (Python 3.7+).
        class_name:   Predicted class label (same for all methods).
        top_k_scores: Optional list of (class_name, score) tuples.
        save_path:    If set, saves the figure here.

    Example:
        compare_methods(
            img_pil,
            heatmaps     = {'CAM': cam_map, 'Grad-CAM': gcam_map},
            class_name   = 'golden retriever',
            top_k_scores = top5,
            save_path    = 'results/compare_dog.png'
        )
    """
    method_names = list(heatmaps.keys())
    n_methods    = len(method_names)
    has_scores   = top_k_scores is not None

    # columns: 1 original + one overlay per method + optional bar chart
    n_cols = 1 + n_methods + (1 if has_scores else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # axes is always a list — but if n_cols == 1 matplotlib returns a bare Axes.
    # Wrapping ensures we can always index with axes[i].
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        f'Method Comparison  ·  Predicted: "{class_name}"',
        fontsize=13, fontweight='bold'
    )

    # Panel 0: original image 
    axes[0].imshow(img_pil)
    axes[0].set_title('Original', fontsize=10)
    axes[0].axis('off')

    # Panels 1…N: one overlay per method 
    for col, method_name in enumerate(method_names, start=1):
        overlay = heatmap_to_overlay(img_pil, heatmaps[method_name])
        axes[col].imshow(overlay)
        axes[col].set_title(method_name, fontsize=10, fontweight='bold')
        axes[col].axis('off')

    # Last panel (optional): top-k bar chart
    if has_scores:
        _add_topk_bar(axes[-1], top_k_scores, highlight_name=class_name)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")

    plt.show()
    plt.close(fig)