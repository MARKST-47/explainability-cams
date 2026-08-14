"""
Entry point — runs CAM on every image in ./images/ and saves results to ./results/

Usage:
    python run.py                    # run on all images in ./images/
    python run.py --img images/dog.jpg           # single image
    python run.py --img images/dog.jpg --class 243  # force class index
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

from utils.preprocess import load_image, get_class_names
from utils.visualize import show_result
from methods.cam import CAM

# Model setup
def load_model() -> torch.nn.Module:
    print("Loading pretrained ResNet-18 (ImageNet weights)...")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    print("  └── Model ready.\n")
    return model

# Single image pipeline 
def process_image(image_path: str, cam_engine: CAM, class_names: list, class_idx: int = None, top_k: int = 5):
    """
    Full pipeline for one image:
      load → forward → CAM → top-k predictions → visualize → save
    """
    print(f"Processing: {Path(image_path).name}")

    # Load
    img_pil, img_tensor = load_image(image_path)

    # Generate CAM
    heatmap, predicted_idx, logits = cam_engine.generate(
        img_tensor,
        class_idx=class_idx
    )

    # Top-k predictions from softmax probabilities
    probs = F.softmax(logits[0], dim=0)
    top_probs, top_indices = probs.topk(top_k)
    top_k_scores = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    # Print summary
    class_name = class_names[predicted_idx]
    print(f"  Predicted  : [{predicted_idx:4d}] {class_name}")
    print(f"  Confidence : {probs[predicted_idx]:.4f}")
    print(f"  Top-{top_k} predictions:")
    for rank, (name, score) in enumerate(top_k_scores, 1):
        marker = " ← target" if name == class_name else ""
        print(f"    {rank}. {score:.4f}  {name}{marker}")

    # Save and display
    stem = Path(image_path).stem
    save_path = f"results/CAM_{stem}.png"

    show_result(
        img_pil=img_pil,
        heatmap=heatmap,
        class_name=class_name,
        method_name='CAM',
        top_k_scores=top_k_scores,
        save_path=save_path
    )
    print()


def main():
    parser = argparse.ArgumentParser(description='CAM visualization')
    parser.add_argument('--img', type=str, default=None,
                        help='Path to a single image. Omit to run on all images/ files.')
    parser.add_argument('--class_idx', type=int, default=None,
                        help='Force a specific ImageNet class index (0-999).')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to display.')
    args = parser.parse_args()

    # Resolve images to process
    if args.img:
        image_paths = [args.img]
    else:
        img_dir = Path('images')
        image_paths = sorted(
            list(img_dir.glob('*.jpg')) +
            list(img_dir.glob('*.jpeg')) +
            list(img_dir.glob('*.png'))
        )
        if not image_paths:
            print("No images found in ./images/")
            print("Add .jpg or .png files there and re-run, or use --img path/to/image.jpg")
            return

    model      = load_model()
    class_names = get_class_names()

    # Use context manager so hook is always cleaned up
    with CAM(model) as cam_engine:
        for path in image_paths:
            process_image(
                image_path=str(path),
                cam_engine=cam_engine,
                class_names=class_names,
                class_idx=args.class_idx,
                top_k=args.top_k
            )

    print(f"Done. All results saved to ./results/")


if __name__ == '__main__':
    main()