"""SegFormer (Xie et al., NeurIPS 2021) in plain PyTorch, with NVIDIA's released weights.

Mix Transformer encoder (MiT) + all-MLP decoder, written to the same module names as Hugging Face's
SegformerForSemanticSegmentation, so `nvidia/mit-b0` (ImageNet encoder) and
`nvidia/segformer-b0-finetuned-ade-512-512` load with strict key matching. Plain PyTorch because
transformers 5 needs torch >= 2.6 and transformers 4 conflicts with gradio's huggingface_hub.
test_segmenter.py checks it numerically against the transformers implementation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

B0 = {"hidden_sizes": [32, 64, 160, 256], "depths": [2, 2, 2, 2], "heads": [1, 2, 5, 8],
      "sr_ratios": [8, 4, 2, 1], "patch_sizes": [7, 3, 3, 3], "strides": [4, 2, 2, 2],
      "mlp_ratio": 4, "decoder_hidden_size": 256}


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, patch: int, stride: int, cin: int, cout: int):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, patch, stride, padding=patch // 2)
        self.layer_norm = nn.LayerNorm(cout)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2:]
        return self.layer_norm(x.flatten(2).transpose(1, 2)), h, w


class EfficientSelfAttention(nn.Module):
    """Multi-head attention whose keys/values come from a sr x sr strided (reduced) sequence."""

    def __init__(self, hidden: int, heads: int, sr: int):
        super().__init__()
        self.heads, self.sr_ratio = heads, sr
        self.query, self.key, self.value = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        if sr > 1:
            self.sr = nn.Conv2d(hidden, hidden, sr, sr)
            self.layer_norm = nn.LayerNorm(hidden)

    def split(self, x):
        b, n, c = x.shape
        return x.view(b, n, self.heads, c // self.heads).transpose(1, 2)

    def forward(self, x, h, w):
        q = self.split(self.query(x))
        kv = x
        if self.sr_ratio > 1:
            b, n, c = x.shape
            kv = self.sr(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
            kv = self.layer_norm(kv)
        k, v = self.split(self.key(kv)), self.split(self.value(kv))
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        out = (attn @ v).transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], -1)


class SelfOutput(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.dense(x)


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, sr: int):
        super().__init__()
        self.self = EfficientSelfAttention(hidden, heads, sr)
        self.output = SelfOutput(hidden)

    def forward(self, x, h, w):
        return self.output(self.self(x, h, w))


class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        return self.dwconv(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, hidden: int, ratio: int):
        super().__init__()
        self.dense1 = nn.Linear(hidden, hidden * ratio)
        self.dwconv = DWConv(hidden * ratio)
        self.dense2 = nn.Linear(hidden * ratio, hidden)

    def forward(self, x, h, w):
        return self.dense2(F.gelu(self.dwconv(self.dense1(x), h, w)))


class Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, sr: int, ratio: int):
        super().__init__()
        self.layer_norm_1, self.layer_norm_2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.attention = Attention(hidden, heads, sr)
        self.mlp = MixFFN(hidden, ratio)

    def forward(self, x, h, w):
        x = x + self.attention(self.layer_norm_1(x), h, w)
        return x + self.mlp(self.layer_norm_2(x), h, w)


class Encoder(nn.Module):
    """(B, 3, H, W) -> four feature maps at 1/4, 1/8, 1/16, 1/32 resolution."""

    def __init__(self, cfg: dict):
        super().__init__()
        sizes, cins = cfg["hidden_sizes"], [3] + cfg["hidden_sizes"][:-1]
        self.patch_embeddings = nn.ModuleList(
            OverlapPatchEmbeddings(p, s, ci, co) for p, s, ci, co in zip(cfg["patch_sizes"], cfg["strides"], cins, sizes))
        self.block = nn.ModuleList(
            nn.ModuleList(Layer(c, n, sr, cfg["mlp_ratio"]) for _ in range(d))
            for c, n, sr, d in zip(sizes, cfg["heads"], cfg["sr_ratios"], cfg["depths"]))
        self.layer_norm = nn.ModuleList(nn.LayerNorm(c) for c in sizes)

    def forward(self, x):
        features = []
        for embed, layers, norm in zip(self.patch_embeddings, self.block, self.layer_norm):
            x, h, w = embed(x)
            for layer in layers:
                x = layer(x, h, w)
            x = norm(x).transpose(1, 2).reshape(x.shape[0], -1, h, w)
            features.append(x)
        return features


class SegformerBody(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = Encoder(cfg)


class MLP(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.proj = nn.Linear(cin, cout)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


class DecodeHead(nn.Module):
    """All-MLP decoder: project each scale to 256 channels, upsample to 1/4, fuse, classify."""

    def __init__(self, cfg: dict, num_labels: int):
        super().__init__()
        d = cfg["decoder_hidden_size"]
        self.linear_c = nn.ModuleList(MLP(c, d) for c in cfg["hidden_sizes"])
        self.linear_fuse = nn.Conv2d(4 * d, d, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(d)
        self.classifier = nn.Conv2d(d, num_labels, 1)

    def forward(self, features):
        size = features[0].shape[2:]
        fused = []
        for feature, mlp in zip(features, self.linear_c):
            x = mlp(feature).transpose(1, 2).reshape(feature.shape[0], -1, *feature.shape[2:])
            fused.append(F.interpolate(x, size=size, mode="bilinear", align_corners=False))
        x = F.relu(self.batch_norm(self.linear_fuse(torch.cat(fused[::-1], dim=1))))
        return self.classifier(x)


class Segformer(nn.Module):
    """(B, 3, H, W) ImageNet-normalised -> (B, num_labels, H/4, W/4) logits."""

    def __init__(self, num_labels: int, cfg: dict = B0):
        super().__init__()
        self.segformer = SegformerBody(cfg)
        self.decode_head = DecodeHead(cfg, num_labels)

    def forward(self, x):
        return self.decode_head(self.segformer.encoder(x))


def load_released(repo: str, num_labels: int) -> Segformer:
    """Segformer with weights from the Hugging Face Hub (downloaded once to ~/.cache/huggingface).

    `nvidia/segformer-b0-finetuned-ade-512-512`: the full model (150 ADE20K classes), strict load.
    `nvidia/mit-b0`: ImageNet encoder only; the decode head stays randomly initialised (for fine-tuning).
    """
    model = Segformer(num_labels)
    state = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}   # mit-b0's ImageNet head
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all(k.startswith("decode_head.") for k in missing), f"missing encoder keys: {missing[:5]}"
    assert not missing or repo.endswith("mit-b0"), f"missing keys: {missing[:5]}"
    return model
