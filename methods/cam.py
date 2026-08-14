import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self._feature_maps: torch.Tensor = None
        self._hook_handle = None
        self._register_hook()
    
    def _register_hook(self):
        target_layer = self.model.layer4[-1]
        def _hook_fn(module, input, output):
            self._feature_maps = output.detach()
        self._hook_handle = target_layer.register_forward_hook(_hook_fn)
    
    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None, upsample_size: tuple = (224, 224)):
        with torch.no_grad():
            logits = self.model(img_tensor)             # (1, 1000)

        if self._feature_maps is None:
            raise RuntimeError(
                "Feature maps not captured. "
                "Ensure the hook is registered and a forward pass has run."
            )

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Extract FC weights for target class 
        # model.fc.weight shape: (1000, 512)
        # We take the row for class_idx: shape (512,)
        # These are the importance weights for each of the 512 channels.
        class_weights = self.model.fc.weight[class_idx]    # (512,)

        # Weighted sum of feature maps
        # self._feature_maps[0]: drop batch dim → (512, 7, 7)
        # einsum 'c,chw->hw':
        #   For each spatial position (h, w):
        #     cam[h,w] = sum over c: class_weights[c] * feature_maps[c, h, w]
        # Result: (7, 7)
        fmaps = self._feature_maps[0]                      # (512, 7, 7)
        cam = torch.einsum('c,chw->hw', class_weights, fmaps)

        # ReLU 
        # Keep only positive contributions.
        # Negative values mean "this region suppresses class c" — not useful
        # for localizing where the class IS in the image.
        cam = F.relu(cam)                                  # (7, 7)

        # Normalize to [0, 1] 
        # Subtract min so the lowest point becomes 0.
        # Divide by max so the highest point becomes 1.
        # eps prevents division by zero (e.g., all-zero cam after ReLU)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample: (7, 7) → (224, 224)
        # F.interpolate needs shape (N, C, H, W), so we add two dims,
        # then squeeze them back off after.
        # mode='bilinear': smooth upsampling (vs. 'nearest' which is blocky)
        # align_corners=False: the standard convention for vision models
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),                 # (1, 1, 7, 7)
            size=upsample_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()                                        # (224, 224)

        return cam_upsampled.detach().numpy(), class_idx, logits

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hook()