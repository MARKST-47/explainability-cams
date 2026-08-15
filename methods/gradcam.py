import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping 
    Works on any CNN architecture. Requires a target_layer to be specified
    when the default (ResNet's layer4[-1]) is not appropriate.

    Importance weight per channel k:
        alpha_k = (1/Z) * sum_{i,j}( d(y^c) / d(A^k_{ij}) )

    Final map:
        L = ReLU( sum_k( alpha_k * A_k ) )
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        """
        Args:
            model:        Pretrained model in eval mode.
            target_layer: Conv layer to hook. Defaults to model.layer4[-1]
                          for ResNet. Pass the correct layer for other archs
                          (e.g. model.features[-3] for VGG).
        """
        self.model = model
        self.model.eval()

        self._activations = None
        self._gradients   = None
        self._fwd_handle  = None
        self._bwd_handle  = None

        target = target_layer if target_layer is not None else model.layer4[-1]
        self._register_hooks(target)

    def _register_hooks(self, layer: nn.Module):
        def _save_activation(module, input, output):
            self._activations = output.detach()

        def _save_gradient(module, grad_input, grad_output):
            # grad_output[0]: gradient of the loss w.r.t. this layer's output
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
        Args:
            img_tensor:    (1, 3, H, W) — must not be inside torch.no_grad().
            class_idx:     Target class index. None = top-1 predicted.
            upsample_size: Output resolution.

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

        grads  = self._gradients[0]    # (C, H, W)
        acts   = self._activations[0]  # (C, H, W)

        alpha  = grads.mean(dim=(1, 2))                         # (C,)
        cam    = (alpha.view(-1, 1, 1) * acts).sum(dim=0)      # (H, W)
        cam    = F.relu(cam)
        cam    = (cam - cam.min()) / (cam.max() + 1e-8)

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