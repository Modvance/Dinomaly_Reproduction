"""
ResNet encoder wrapper for Dinomaly/ViTill.
Exposes get_multi_scale_features(x) so ViTill can use ResNet as backbone
instead of ViT. All layer outputs are projected to embed_dim and resized
to a common spatial size (same as ViT patch grid for 392 crop: 28x28).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# 使用项目内的 ResNet（返回多阶段特征）
from .resnet import ResNet, Bottleneck, BasicBlock, _resnet, model_urls


# ResNet50 各 stage 输出通道: layer1=256, layer2=512, layer3=1024, layer4=2048
RESNET_CHANNELS = {
    'resnet18': [64, 64, 128, 256],
    'resnet34': [64, 64, 128, 256],
    'resnet50': [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
    'resnet152': [256, 512, 1024, 2048],
    'wide_resnet50_2': [256, 512, 1024, 2048],
    'wide_resnet101_2': [256, 512, 1024, 2048],
}


class ResNetEncoderWrapper(nn.Module):
    """
    Wraps a ResNet backbone so it matches the interface expected by ViTill
    when using get_multi_scale_features: returns a list of (B, 1+N, C) tensors
    with same N and C (embed_dim), so downstream fuse/decoder logic is unchanged.
    """
    def __init__(
        self,
        backbone: nn.Module,
        out_channels: List[int],
        embed_dim: int = 768,
        target_side: int = 28,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_register_tokens = 0
        self.embed_dim = embed_dim
        self.target_side = target_side
        self.num_patches = target_side * target_side
        # 1x1 投影：各 stage 通道 -> embed_dim
        self.proj = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in out_channels
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def get_multi_scale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return list of (B, 1+N, embed_dim), N = target_side^2."""
        B = x.shape[0]
        # ResNet 前向，逐 stage 取特征（不调用 backbone.forward，因其实装可能只返回 3 个）
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        f1 = self.backbone.layer1(x)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)
        feats = [f1, f2, f3, f4]
        out_list = []
        for i, f in enumerate(feats):
            f = self.proj[i](f)
            f = F.interpolate(f, size=(self.target_side, self.target_side), mode='bilinear', align_corners=False)
            f = f.flatten(2).permute(0, 2, 1)
            cls_t = self.cls_token.expand(B, 1, self.embed_dim)
            f = torch.cat([cls_t, f], dim=1)
            out_list.append(f)
        return out_list


def _get_resnet_backbone(name: str, pretrained: bool = True):
    """Build ResNet backbone (full model with layer4), optionally pretrained."""
    if name == 'resnet18':
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress=True)
    if name == 'resnet34':
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress=True)
    if name == 'resnet50':
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress=True)
    if name == 'resnet101':
        return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress=True)
    if name == 'resnet152':
        return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress=True)
    if name == 'wide_resnet50_2':
        return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress=True, width_per_group=64 * 2)
    if name == 'wide_resnet101_2':
        return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress=True, width_per_group=64 * 2)
    raise ValueError("Unknown resnet arch: {}".format(name))


def load_resnet_encoder(
    name: str,
    embed_dim: int = 768,
    target_side: int = 28,
    pretrained: bool = True,
) -> ResNetEncoderWrapper:
    """
    name: e.g. 'resnet50', 'wide_resnet50_2'
    """
    name = name.lower().strip()
    if name not in RESNET_CHANNELS:
        raise ValueError("Unsupported ResNet: {}. Choose from {}".format(
            name, list(RESNET_CHANNELS.keys())))
    backbone = _get_resnet_backbone(name, pretrained=pretrained)
    out_channels = RESNET_CHANNELS[name]
    return ResNetEncoderWrapper(
        backbone=backbone,
        out_channels=out_channels,
        embed_dim=embed_dim,
        target_side=target_side,
    )
