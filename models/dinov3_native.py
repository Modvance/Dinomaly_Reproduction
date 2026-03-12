import torch
import torch.nn as nn
from typing import Iterable, List, Sequence, Tuple, Union


class DinoV3NativeTokenEncoder(nn.Module):
    """
    Adapter for official DINOv3 backbones (DinoVisionTransformer).

    Exposes `get_multi_scale_features(x)` returning a list of token tensors shaped
    like ViT outputs: (B, 1 + num_register_tokens + N, C).

    Internally uses the *official* `get_intermediate_layers()` API, so the tokens
    are produced by the native DINOv3 tokenization + RoPE + transformer blocks.
    """

    def __init__(self, backbone: nn.Module, target_layers: Sequence[int]):
        super().__init__()
        self.backbone = backbone
        self.target_layers = list(target_layers)

        # ViTill expects this attribute. In DINOv3 these are "storage tokens".
        n_storage = int(getattr(self.backbone, "n_storage_tokens", 0))
        self.num_register_tokens = n_storage

    @torch.no_grad()
    def get_multi_scale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns tokens for each target layer as a list of (B, 1+R+N, C),
        where R = num_register_tokens (storage tokens in DINOv3).
        """
        outs = self.backbone.get_intermediate_layers(
            x,
            n=self.target_layers,
            reshape=False,
            return_class_token=True,
            return_extra_tokens=True,
            norm=False,
        )
        tokens: List[torch.Tensor] = []
        for patch_tokens, cls_token, extra_tokens in outs:
            # patch_tokens: (B, N, C)
            # cls_token: (B, C)
            # extra_tokens: (B, R, C)
            full = torch.cat([cls_token.unsqueeze(1), extra_tokens, patch_tokens], dim=1)
            tokens.append(full)
        return tokens


def load_dinov3_backbone(
    hub_model: str = "dinov3_vitb16",
    source: str = "github",
    repo_or_dir: str = "facebookresearch/dinov3",
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Load official DINOv3 backbone via torch.hub.

    - source='github': downloads dinov3 code from GitHub (requires GitHub access)
    - source='local' : repo_or_dir must point to a local clone of dinov3
    """
    if source not in {"github", "local"}:
        raise ValueError("source must be 'github' or 'local'")

    model = torch.hub.load(repo_or_dir, hub_model, source=source, pretrained=pretrained, **kwargs)
    model.eval()
    return model

