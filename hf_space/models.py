"""
Stage D — Model definitions.

BaselineCNN      Small conv-net; fast to iterate on a new dataset (pure PyTorch).
build_coatnet    CoAtNet-0. Prefers the pretrained timm model (as in the 2023
                 arXiv paper); if timm / torchvision are unavailable on this
                 Python build it transparently falls back to NativeCoAtNet.
NativeCoAtNet    A self-contained conv-plus-transformer ("CoAtNet-style") network
                 in pure PyTorch — no timm, no torchvision, no pretrained
                 weights. Lets `--model coatnet` run anywhere torch runs
                 (e.g. Python 3.14, where torchvision has no compiled wheel yet).

NOTE on `import timm`: timm imports torchvision at package-load time, and the
current torchvision wheel for some Python versions (3.14) ships without its
compiled ops, so merely importing timm raises at import. We therefore import
timm *lazily*, inside build_coatnet, and catch any exception — so the baseline
path never depends on it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Baseline CNN ──────────────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """
    Three-block conv-net for 1×64×64 log-mel spectrograms.
    Gets the full pipeline end-to-end before introducing CoAtNet.
    """

    def __init__(self, num_classes=36):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 64×64 -> 32×32
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 2 — 32×32 -> 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 3 — 16×16 -> 4×4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Native CoAtNet-style model (pure PyTorch, no external deps) ─────────────────

class _MBConv(nn.Module):
    """Mobile inverted bottleneck conv block with squeeze-excite and residual.
    This is the convolutional half of CoAtNet's C-C-T-T stage design."""

    def __init__(self, in_ch, out_ch, stride=1, expand=4):
        super().__init__()
        mid = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
        )
        # Squeeze-and-excite
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, max(1, mid // 16), 1), nn.GELU(),
            nn.Conv2d(max(1, mid // 16), mid, 1), nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.block(x)
        out = out * self.se(out)
        out = self.project(out)
        return out + x if self.use_res else out


class _TransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block operating on H*W tokens.
    This is the attention half of CoAtNet's design (the two 'T' stages)."""

    def __init__(self, dim, heads=4, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):                    # x: (B, N, dim)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class NativeCoAtNet(nn.Module):
    """
    Compact CoAtNet-style hybrid (conv stem -> 2 conv stages -> 2 transformer
    stages) for 1×64×64 spectrograms. ~2–3 M params; trains comfortably on CPU.
    No pretraining, so on real cross-session data it will trail the timm
    ImageNet-pretrained CoAtNet-0 — but it fully exercises the architecture and
    the training/eval/demo pipeline, and learns the synthetic set to high
    accuracy.
    """

    def __init__(self, num_classes=36, dims=(64, 96, 192, 256)):
        super().__init__()
        c0, c1, c2, c3 = dims
        # Stem: 64×64 -> 32×32
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.GELU(),
        )
        # Conv stage 1: 32×32 -> 16×16
        self.s1 = nn.Sequential(_MBConv(c0, c1, stride=2), _MBConv(c1, c1))
        # Conv stage 2: 16×16 -> 8×8
        self.s2 = nn.Sequential(_MBConv(c1, c2, stride=2), _MBConv(c2, c2))
        # Downsample to transformer resolution 8×8 -> 4×4, project to c3
        self.down = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.GELU(),
        )
        # Transformer stages on the 4×4 = 16 tokens
        self.pos = nn.Parameter(torch.zeros(1, 16, c3))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.t1 = _TransformerBlock(c3, heads=4)
        self.t2 = _TransformerBlock(c3, heads=4)
        self.norm = nn.LayerNorm(c3)
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.down(x)                      # (B, c3, 4, 4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # (B, 16, c3)
        x = x + self.pos[:, : H * W]
        x = self.t1(x)
        x = self.t2(x)
        x = self.norm(x).mean(dim=1)          # global token pool
        return self.head(x)


# ── timm CoAtNet (preferred when available) ─────────────────────────────────────

class _ResizeWrapper(nn.Module):
    """Upsamples input to size×size before forwarding (timm CoAtNet-0 expects
    224×224 for its pretrained relative-position bias tables)."""
    def __init__(self, model, size=224):
        super().__init__()
        self.model = model
        self.size  = size

    def forward(self, x):
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size),
                              mode='bilinear', align_corners=False)
        return self.model(x)


def _try_build_timm_coatnet(num_classes, pretrained, model_name):
    """Return a wrapped timm CoAtNet, or None if timm/torchvision are unusable."""
    try:
        import timm   # noqa: F401  (imports torchvision internally)
    except Exception as e:                    # ImportError OR torchvision RuntimeError
        print(f'[models] timm/torchvision unavailable ({type(e).__name__}: {e}).')
        return None
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        _patch_first_conv(model, in_channels=1, pretrained=pretrained)
        return _ResizeWrapper(model, size=224)
    except Exception as e:
        print(f'[models] timm CoAtNet build failed ({type(e).__name__}: {e}).')
        return None


def build_coatnet(num_classes=36, pretrained=True, model_name='coatnet_0_rw_224'):
    """
    CoAtNet-0. Preference order:
      1. timm pretrained CoAtNet-0 (matches the 2023 paper) — if importable.
      2. NativeCoAtNet (pure-PyTorch fallback) — always works.
    """
    wrapped = _try_build_timm_coatnet(num_classes, pretrained, model_name)
    if wrapped is not None:
        print('[models] Using timm CoAtNet-0'
              f'{" (ImageNet-pretrained)" if pretrained else ""}.')
        return wrapped
    print('[models] Falling back to NativeCoAtNet (pure PyTorch, no pretraining).')
    return NativeCoAtNet(num_classes=num_classes)


def _patch_first_conv(module, in_channels=1, pretrained=True):
    """Replace the first 3-channel Conv2d with a 1-channel equivalent, averaging
    the pretrained RGB weights across channels."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels, child.out_channels,
                kernel_size=child.kernel_size, stride=child.stride,
                padding=child.padding, bias=child.bias is not None,
            )
            if pretrained and child.weight is not None:
                with torch.no_grad():
                    new_conv.weight.copy_(child.weight.mean(dim=1, keepdim=True))
                    if child.bias is not None:
                        new_conv.bias.copy_(child.bias)
            setattr(module, name, new_conv)
            return True
        if _patch_first_conv(child, in_channels, pretrained):
            return True
    return False


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(model_type, num_classes=36, pretrained=True):
    if model_type == 'coatnet':
        return build_coatnet(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'baseline':
        return BaselineCNN(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model_type: {model_type!r}  (choose baseline or coatnet)')
