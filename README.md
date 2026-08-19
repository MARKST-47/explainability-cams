# Explainability CAMs

Implementations of CAM-family saliency methods for CNN classifiers, built from scratch using PyTorch. Supports multiple architectures and produces overlay visualizations with top-k prediction breakdowns.

## Methods

| Method   | Paper                       | Gradient-free | Architecture-agnostic   |
| -------- | --------------------------- | ------------- | ----------------------- |
| CAM      | Zhou et al., CVPR 2016      | Yes           | No (requires GAP -> FC) |
| Grad-CAM | Selvaraju et al., ICCV 2017 | No            | Yes                     |

## Architectures

| Arch     | CAM | Grad-CAM |
| -------- | --- | -------- |
| resnet18 | Yes | Yes      |
| resnet50 | Yes | Yes      |
| vgg16    | No  | Yes      |

## Setup

```bash
pip install -r requirements.txt
```

Place test images (`.jpg` / `.png`) in `./images/`.

## Usage

```bash
# Grad-CAM on all images with ResNet-18 (default)
python run.py

# Specific method and architecture
python run.py --method gradcam --arch vgg16

# Side-by-side comparison of all methods
python run.py --method all --arch resnet18

# Single image, forced class
python run.py --img images/dog.jpg --class_idx 243

# Full options
python run.py --method all --arch resnet50 --img images/dog.jpg --top_k 5
```

Results are saved to `./results/`.

## Project Structure

```
methods/
    cam.py          CAM implementation
    gradcam.py      Grad-CAM implementation
    gradcampp.py    Grad-CAM++ implementation
utils/
    preprocess.py   Image loading and ImageNet normalization
    visualize.py    Heatmap overlay and comparison figures
images/             Input images
results/            Output figures
run.py              Entry point
requirements.txt
```

## Adding a Method

1. Implement `methods/yourmethod.py` with a `generate(img_tensor, class_idx, upsample_size)` method returning `(heatmap, class_idx, logits)` and context manager support.
2. Import it in `run.py` and add it to `METHOD_REGISTRY` and `ALL_METHODS`.

## References

- Zhou et al. "Learning Deep Features for Discriminative Localization." CVPR 2016. arXiv:1512.04150
- Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017. arXiv:1610.02391
