"""
Visualize CAM-family attribution methods on ImageNet classifiers.

Usage:
    python run.py [--method cam|gradcam|all] [--arch resnet18|resnet50|vgg16]
                  [--img PATH] [--class_idx N] [--top_k N]

Examples:
    python run.py --method all --arch resnet18
    python run.py --method gradcam --arch vgg16 --img images/dog.jpg
    python run.py --method cam --img images/cat.jpg --class_idx 281
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_Weights,
)

from utils.preprocess import load_image, get_class_names
from utils.visualize import show_result, compare_methods
from methods.cam import CAM
from methods.gradcam import GradCAM


# Architecture registry.
# target_layer: callable(model) -> the conv layer to hook for gradient methods.
# cam_support:  CAM requires GAP -> single FC, which VGG-style archs lack.
ARCH_REGISTRY = {
    "resnet18": {
        "loader":       lambda: models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
        "target_layer": lambda m: m.layer4[-1],
        "cam_support":  True,
    },
    "resnet50": {
        "loader":       lambda: models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        "target_layer": lambda m: m.layer4[-1],
        "cam_support":  True,
    },
    "vgg16": {
        "loader":       lambda: models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
        "target_layer": lambda m: m.features[-3],  # last Conv2d before ReLU + MaxPool
        "cam_support":  False,
    },
}

# Method registry.
# Add new methods here when implemented — nothing else in this file needs to change.
METHOD_REGISTRY = {
    "cam":     CAM,
    "gradcam": GradCAM,
}

ALL_METHODS = ["cam", "gradcam"]


def load_model(arch_name: str):
    if arch_name not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch '{arch_name}'. Choose from: {list(ARCH_REGISTRY)}")
    cfg = ARCH_REGISTRY[arch_name]
    print(f"Loading {arch_name}...")
    model = cfg["loader"]()
    model.eval()
    target_layer = cfg["target_layer"](model)
    return model, target_layer, cfg["cam_support"]


def create_engine(method_name: str, model, target_layer):
    if method_name == "cam":
        return CAM(model)
    # Gradient-based methods receive the target layer so they work
    # across architectures where the hook location differs.
    return METHOD_REGISTRY[method_name](model, target_layer=target_layer)


def get_top_k(logits: torch.Tensor, class_names: list, k: int):
    probs = F.softmax(logits[0], dim=0)
    top_probs, top_idx = probs.topk(k)
    return [(class_names[i.item()], p.item()) for i, p in zip(top_idx, top_probs)]


def process_image(image_path, method_names, model, target_layer, class_names,
                  forced_class=None, top_k=5):
    print(f"  {Path(image_path).name}")

    img_pil, img_tensor = load_image(image_path)

    heatmaps    = {}
    active_class = forced_class
    main_logits  = None

    for name in method_names:
        with create_engine(name, model, target_layer) as engine:
            heatmap, pred_class, logits = engine.generate(
                img_tensor, class_idx=active_class
            )
        heatmaps[name.upper()] = heatmap

        if active_class is None:
            active_class = pred_class
            main_logits  = logits

    class_name   = class_names[active_class]
    top_k_scores = get_top_k(main_logits, class_names, top_k)

    print(f"  [{active_class}] {class_name}  ({top_k_scores[0][1]:.3f})")

    stem      = Path(image_path).stem
    tag       = "_".join(method_names)
    save_path = f"results/{tag}_{stem}.png"

    if len(method_names) == 1:
        show_result(
            img_pil      = img_pil,
            heatmap      = heatmaps[method_names[0].upper()],
            class_name   = class_name,
            method_name  = method_names[0].upper(),
            top_k_scores = top_k_scores,
            save_path    = save_path,
        )
    else:
        compare_methods(
            img_pil      = img_pil,
            heatmaps     = heatmaps,
            class_name   = class_name,
            top_k_scores = top_k_scores,
            save_path    = save_path,
        )


def resolve_methods(method_arg: str, cam_support: bool) -> list:
    methods = ALL_METHODS if method_arg == "all" else [method_arg]

    if not cam_support and "cam" in methods:
        print("Note: CAM requires a GAP->FC architecture — skipping for this arch.")
        methods = [m for m in methods if m != "cam"]

    if not methods:
        raise RuntimeError("No valid methods remain for this architecture.")

    unknown = [m for m in methods if m not in METHOD_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Choose from: {list(METHOD_REGISTRY)} or 'all'")

    return methods


def resolve_images(img_arg: str) -> list:
    if img_arg:
        p = Path(img_arg)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {img_arg}")
        return [str(p)]

    img_dir = Path("images")
    img_dir.mkdir(exist_ok=True)
    paths = sorted(img_dir.glob("*.jpg")) + \
            sorted(img_dir.glob("*.jpeg")) + \
            sorted(img_dir.glob("*.png"))

    if not paths:
        print("No images found in ./images/ — add .jpg/.png files or use --img.")
        return []
    return [str(p) for p in paths]


def main():
    parser = argparse.ArgumentParser(description="CAM-family visualization")
    parser.add_argument("--method",    default="gradcam",
                        help="cam | gradcam | all  (default: gradcam)")
    parser.add_argument("--arch",      default="resnet18",
                        help=f"Architecture: {list(ARCH_REGISTRY)}  (default: resnet18)")
    parser.add_argument("--img",       default=None,
                        help="Single image path. Omit to run on all images/")
    parser.add_argument("--class_idx", default=None, type=int,
                        help="Force a specific ImageNet class index (0-999)")
    parser.add_argument("--top_k",     default=5, type=int,
                        help="Top-k predictions in bar chart (default: 5)")
    args = parser.parse_args()

    model, target_layer, cam_support = load_model(args.arch)
    method_names = resolve_methods(args.method, cam_support)
    image_paths  = resolve_images(args.img)

    if not image_paths:
        return

    print(f"arch={args.arch}  methods={method_names}  images={len(image_paths)}\n")

    class_names = get_class_names()
    Path("results").mkdir(exist_ok=True)

    for path in image_paths:
        process_image(
            image_path   = path,
            method_names = method_names,
            model        = model,
            target_layer = target_layer,
            class_names  = class_names,
            forced_class = args.class_idx,
            top_k        = args.top_k,
        )

    print(f"\nResults saved to ./results/")


if __name__ == "__main__":
    main()
