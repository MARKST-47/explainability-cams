# Imagenet Transforms
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(image_path: str, size: int = 224):
    """
    Load an image from disk and return:
      - img_pil  : PIL image resized to (size, size) — used for visualization
      - tensor   : Preprocessed torch tensor (1, 3, size, size) — model input

    The tensor is normalized for ImageNet. img_pil is NOT normalized —
    it's the raw pixel image you'll overlay the heatmap onto.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path).convert('RGB')   # force RGB 
    img_pil = img.resize((size, size), Image.LANCZOS)  # keep for visualization

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),              # [0,255] PIL → [0,1] float tensor
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tensor = transform(img).unsqueeze(0)   # (3, H, W) → (1, 3, H, W)
    return img_pil, tensor


def get_class_names():
    """
    Get the 1000 ImageNet class names from torchvision metadata.
    Returns a list where index i = class name for class i.
    Requires torchvision >= 0.13.
    """
    return ResNet18_Weights.IMAGENET1K_V1.meta['categories']


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized model-input tensor back to a displayable numpy image.
    Useful for debugging — lets you verify the tensor actually looks right.

    Args:
        tensor: shape (1, 3, H, W) or (3, H, W), normalized with ImageNet stats
    Returns:
        numpy array (H, W, 3) with values in [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)          # (1, 3, H, W) → (3, H, W)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    img = tensor * std + mean               # undo normalization
    img = img.permute(1, 2, 0).numpy()     # (3, H, W) → (H, W, 3)
    return img.clip(0, 1)