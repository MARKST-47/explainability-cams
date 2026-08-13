# Main compare file, runs all methods on n imagesdef compare_all(img_pil, cam_map, gradcam_map, gradcampp_map, class_name):
import matplotlib.pyplot as plt
import numpy as np

def compare_all(img_pil, cam_map, gradcam_map, gradcampp_map, class_name):
    img_np = np.array(img_pil.resize((224, 224))) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Predicted: {class_name}', fontsize=13)

    axes[0].imshow(img_np)
    axes[0].set_title('Original')

    for ax, heatmap, title in zip(
        axes[1:],
        [cam_map, gradcam_map, gradcampp_map],
        ['CAM', 'Grad-CAM', 'Grad-CAM++']
    ):
        overlay = img_np.copy()
        heatmap_colored = plt.cm.jet(heatmap)[..., :3]
        blended = 0.5 * overlay + 0.5 * heatmap_colored
        ax.imshow(blended)
        ax.set_title(title)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'comparison_{class_name}.png', dpi=150)
    plt.show()
    