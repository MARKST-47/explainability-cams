import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMPlusPlus:
    """
    Grad-CAM++ — Chattopadhay et al., WACV 2018.
    arXiv: 1710.11063

    Extends Grad-CAM with pixel-wise alpha weights from second and third
    order gradient terms, producing tighter localisation when a class
    appears multiple times or occupies a small image region.

    Per-channel weight:
        alpha_{ij}^k  =  grad²_{ij}^k  /  ( 2*grad²_{ij}^k
                          + sum_{a,b}( A_{ab}^k * grad³_{ab}^k ) )

        w_k  =  sum_{i,j}( alpha_{ij}^k * ReLU(grad_{ij}^k) )

    Final map:
        L  =  ReLU( sum_k( w_k * A_k ) )
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()

        self._activations = None
        self._gradients   = None
        self._fwd_handle  = None
        self._bwd_handle  = None

        target = target_layer if target_layer is not None else model.layer4[-1]
        self._register_hooks(target)

    def _register_hooks(self, layer: nn.Module):
        # Inplace ops conflict with full_backward_hook — disable across model.
        for m in self.model.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False

        def _save_activation(module, input, output):
            self._activations = output.detach()

        def _save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._fwd_handle = layer.register_forward_hook(_save_activation)
        self._bwd_handle = layer.register_full_backward_hook(_save_gradient)

    def remove_hooks(self):
        if self._fwd_handle:
            self._fwd_handle.remove()
        if self._bwd_handle:
            self._bwd_handle.remove()

    def generate(self,
                 img_tensor: torch.Tensor,
                 class_idx: int = None,
                 upsample_size: tuple = (224, 224)):
        """
        Returns:
            heatmap   (H, W) numpy float32 in [0, 1]
            class_idx int
            logits    (1, 1000) detached tensor
        """
        self.model.zero_grad()
        logits = self.model(img_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads = self._gradients[0]    # (C, H, W)
        acts  = self._activations[0]  # (C, H, W)

        grads_sq = grads ** 2         # (C, H, W)
        grads_cu = grads ** 3         # (C, H, W)

        # The spatial sum term is constant for all (i,j) within each channel k,
        # computed as sum_{a,b}( A_{ab}^k * grad³_{ab}^k ) — shape (C, 1, 1).
        spatial_sum = (acts * grads_cu).sum(dim=(1, 2), keepdim=True)
        denom       = 2.0 * grads_sq + spatial_sum   # (C, H, W) via broadcast

        # Where denom is non-positive or near-zero set alpha to 0.
        # Prevents sign flips when the third-order term dominates negatively.
        alpha = torch.where(
            denom > 1e-7,
            grads_sq / denom.clamp(min=1e-7),
            torch.zeros_like(grads_sq),
        )

        # Only positive gradient positions contribute to channel weights.
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))  # (C,)

        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)   # (H, W)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=upsample_size,
            mode='bilinear',
            align_corners=False,
        ).squeeze()

        self.model.zero_grad()
        return cam.detach().numpy(), class_idx, logits.detach()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()