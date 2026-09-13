"""Backbone / model construction for the re-ID benchmark.

Two model families:

* ``EmbeddingNet`` - generic: ``backbone -> [BNNeck] -> Linear(feat, 512)``.
  Used by ``proposed`` (MobileNetV2), ``convnext_tiny``, ``resnet50_arcface``
  and ``dinov2_vits14``.  The MobileNetV2 path reproduces the published model
  exactly (Dropout(0.2) kept, single ``Linear(1280, 512)`` head).

* ``ReIDResNet`` - dedicated person/vehicle re-ID architecture: ResNet50 with
  last-stride-1, optional Non-local blocks + GeM pooling, a BNNeck and an ID
  classifier head.  Used by the two strict-ReID methods ``bot_reid``
  (Bag of Tricks, CVPRW 2019) and ``agw_reid`` (AGW, TPAMI 2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Generic backbones (final classifier stripped)
# --------------------------------------------------------------------------- #
def _mobilenet_v2(pretrained: bool):
    from torchvision import models
    from torchvision.models import MobileNet_V2_Weights

    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    feat_dim = m.classifier[1].in_features          # 1280
    m.classifier[1] = nn.Identity()                 # keep Dropout(0.2) in classifier[0]
    return m, feat_dim


def _resnet50(pretrained: bool):
    from torchvision import models
    from torchvision.models import ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    feat_dim = m.fc.in_features                      # 2048
    m.fc = nn.Identity()
    return m, feat_dim


def _convnext_tiny(pretrained: bool):
    from torchvision import models
    from torchvision.models import ConvNeXt_Tiny_Weights

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.convnext_tiny(weights=weights)
    feat_dim = m.classifier[2].in_features           # 768
    m.classifier[2] = nn.Identity()                  # keep LayerNorm2d + Flatten
    return m, feat_dim


class _DinoV2Wrapper(nn.Module):
    """timm DINOv2 ViT-S/14 with a 128->126 bilinear resize (patch-14 divisibility)."""

    def __init__(self, pretrained: bool):
        super().__init__()
        import timm

        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.out_dim = self.model.num_features       # 384

    def forward(self, x):
        if x.shape[-1] % 14 != 0 or x.shape[-2] % 14 != 0:
            new_h = (x.shape[-2] // 14) * 14
            new_w = (x.shape[-1] // 14) * 14
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return self.model(x)


def _dinov2_vits14(pretrained: bool):
    m = _DinoV2Wrapper(pretrained=pretrained)
    return m, m.out_dim


_BACKBONES = {
    "mobilenet_v2": _mobilenet_v2,
    "resnet50": _resnet50,
    "convnext_tiny": _convnext_tiny,
    "dinov2_vits14": _dinov2_vits14,
}


# --------------------------------------------------------------------------- #
#  Generic embedding network
# --------------------------------------------------------------------------- #
class EmbeddingNet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, embedding_dim: int = 512,
                 bnneck: bool = False):
        super().__init__()
        self.backbone = backbone
        self.bnneck = None
        if bnneck:
            self.bnneck = nn.BatchNorm1d(feat_dim)
            self.bnneck.bias.requires_grad_(False)
        self.head = nn.Linear(feat_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        if self.bnneck is not None:
            feats = self.bnneck(feats)
        return self.head(feats)


# --------------------------------------------------------------------------- #
#  Dedicated re-ID building blocks
# --------------------------------------------------------------------------- #
class GeM(nn.Module):
    """Generalized-mean pooling (Radenovic et al. 2018), learnable ``p`` - AGW."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        return x.flatten(1)


class NonLocalBlock(nn.Module):
    """Embedded-Gaussian non-local block (Wang et al. 2018), as used by AGW."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.inter = max(in_channels // 2, 1)
        self.g = nn.Conv2d(in_channels, self.inter, 1)
        self.theta = nn.Conv2d(in_channels, self.inter, 1)
        self.phi = nn.Conv2d(in_channels, self.inter, 1)
        self.W = nn.Sequential(
            nn.Conv2d(self.inter, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)   # identity init -> safe to insert
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x):
        b, _, h, w = x.shape
        g = self.g(x).view(b, self.inter, -1).permute(0, 2, 1)
        theta = self.theta(x).view(b, self.inter, -1).permute(0, 2, 1)
        phi = self.phi(x).view(b, self.inter, -1)
        f = torch.softmax(torch.bmm(theta, phi), dim=-1)
        y = torch.bmm(f, g).permute(0, 2, 1).contiguous().view(b, self.inter, h, w)
        return x + self.W(y)


def _weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ReIDResNet(nn.Module):
    """ResNet50 re-ID backbone with BNNeck + ID head (BoT / AGW).

    ``forward``        -> post-BNNeck feature  (inference embedding, 2048-d)
    ``forward_train``  -> (pre-BNNeck feature, ID logits)   for the combined loss
    """

    def __init__(self, num_classes: int, pretrained: bool = True,
                 last_stride1: bool = True, non_local: bool = False,
                 pooling: str = "avg"):
        super().__init__()
        from torchvision import models
        from torchvision.models import ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        r = models.resnet50(weights=weights)
        if last_stride1:
            r.layer4[0].conv2.stride = (1, 1)
            r.layer4[0].downsample[0].stride = (1, 1)

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            r.layer1, r.layer2, r.layer3, r.layer4,
        )
        self.nl2 = NonLocalBlock(512) if non_local else None
        self.nl3 = NonLocalBlock(1024) if non_local else None
        self.pool = GeM() if pooling == "gem" else None  # None -> adaptive avg pool

        feat_dim = 2048
        self.bnneck = nn.BatchNorm1d(feat_dim)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(_weights_init_classifier)
        self.embedding_dim = feat_dim

    def _features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.nl2 is not None:
            x = self.nl2(x)
        x = self.layer3(x)
        if self.nl3 is not None:
            x = self.nl3(x)
        x = self.layer4(x)
        if self.pool is not None:
            return self.pool(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.bnneck(self._features(x))

    def forward_train(self, x):
        feat = self._features(x)
        logits = self.classifier(self.bnneck(feat))
        return feat, logits


# --------------------------------------------------------------------------- #
def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


# --------------------------------------------------------------------------- #
#  Generic timm backbone under a dedicated re-ID head (BoT / AGW recipe)
# --------------------------------------------------------------------------- #
_TIMM_REID_NAMES = {
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in1k",
}


class ReIDTimmBackbone(nn.Module):
    """Any timm CNN backbone + GeM/avg pool + BNNeck + ID classifier head.

    ``forward``        -> post-BNNeck feature (inference embedding)
    ``forward_train``  -> (pre-BNNeck feature, ID logits)
    """

    def __init__(self, timm_name: str, num_classes: int, pretrained: bool = True,
                 pooling: str = "gem"):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        feat_dim = self.backbone.num_features
        self.pool = GeM() if pooling == "gem" else None
        self.bnneck = nn.BatchNorm1d(feat_dim)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(_weights_init_classifier)
        self.embedding_dim = feat_dim

    def _features(self, x):
        f = self.backbone.forward_features(x)          # (B, C, H, W) for conv nets
        if f.dim() == 4:
            if self.pool is not None:
                return self.pool(f)
            return F.adaptive_avg_pool2d(f, 1).flatten(1)
        if f.dim() == 3:                               # (B, N, C) token grid
            return f.mean(dim=1)
        return f

    def forward(self, x):
        return self.bnneck(self._features(x))

    def forward_train(self, x):
        feat = self._features(x)
        return feat, self.classifier(self.bnneck(feat))


# --------------------------------------------------------------------------- #
#  CLIP-ReID (Li et al., AAAI 2023) - learnable ID prompts + 2-stage training
# --------------------------------------------------------------------------- #
_CLIP_CFG = {
    "ViT-B/16": dict(embed_dim=512, image_resolution=224, vision_layers=12,
                     vision_width=768, vision_patch_size=16, context_length=77,
                     vocab_size=49408, transformer_width=512, transformer_heads=8,
                     transformer_layers=12),
    "RN50": dict(embed_dim=1024, image_resolution=224, vision_layers=(3, 4, 6, 3),
                 vision_width=64, vision_patch_size=None, context_length=77,
                 vocab_size=49408, transformer_width=512, transformer_heads=8,
                 transformer_layers=12),
}


def _build_clip(arch: str, pretrained: bool):
    import clip

    if pretrained:
        model, _ = clip.load(arch, device="cpu")
        return model.float()
    from clip.model import CLIP

    return CLIP(**_CLIP_CFG[arch]).float()


class PromptLearner(nn.Module):
    """Per-ID learnable context tokens: "A photo of a [X]x4 <id> beetle." (CoOp-style)."""

    def __init__(self, num_classes: int, clip_model, n_ctx: int = 4):
        super().__init__()
        import clip

        ctx_dim = clip_model.ln_final.weight.shape[0]
        prompt = "A photo of a " + " ".join(["X"] * n_ctx) + " beetle."
        tokenized = clip.tokenize(prompt)               # (1, 77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).float()   # (1, 77, ctx_dim)

        self.n_ctx = n_ctx
        self.num_classes = num_classes
        # learnable context, one set per identity
        self.cls_ctx = nn.Parameter(torch.empty(num_classes, n_ctx, ctx_dim).normal_(std=0.02))
        # fixed pieces: [SOS] + "A photo of a"  (n_ctx+1 tokens), then suffix
        self.register_buffer("token_prefix", embedding[:, : n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_ctx :, :])
        self.register_buffer("tokenized_prompt", tokenized)             # (1, 77)

    def forward(self, class_ids=None):
        ctx = self.cls_ctx if class_ids is None else self.cls_ctx[class_ids]
        k = ctx.shape[0]
        prefix = self.token_prefix.expand(k, -1, -1)
        suffix = self.token_suffix.expand(k, -1, -1)
        return torch.cat([prefix, ctx, suffix], dim=1)                  # (k, 77, ctx_dim)


class CLIPReID(nn.Module):
    """CLIP-ReID: CLIP image encoder + PromptLearner, BNNeck + ID head for stage 2.

    ``forward(x)``            -> image embedding (inference)
    ``encode_image_norm(x)``  -> L2-normalised image features (stage 1)
    ``text_features(ids)``    -> L2-normalised prompt/text features
    ``forward_stage2(x)``     -> (image_feat, id_logits, i2t_logits)
    """

    INPUT_RESOLUTION = 224

    def __init__(self, num_classes: int, arch: str = "ViT-B/16", pretrained: bool = True,
                 n_ctx: int = 4):
        super().__init__()
        clip_model = _build_clip(arch, pretrained)
        self.prompt_learner = PromptLearner(num_classes, clip_model, n_ctx=n_ctx)

        # keep only the pieces we need (no duplicate `self.clip` submodule)
        self.visual = clip_model.visual
        self.text_transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.positional_embedding = nn.Parameter(clip_model.positional_embedding.data.clone())
        self.text_projection = nn.Parameter(clip_model.text_projection.data.clone())
        self.logit_scale = nn.Parameter(clip_model.logit_scale.data.clone())

        embed_dim = self.text_projection.shape[1]
        self.embedding_dim = embed_dim
        self.bnneck = nn.BatchNorm1d(embed_dim)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.classifier.apply(_weights_init_classifier)

        # the text encoder is never trained
        set_requires_grad(self.text_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)
        self.positional_embedding.requires_grad_(False)
        self.text_projection.requires_grad_(False)
        self.logit_scale.requires_grad_(False)

        # names of the frozen text-tower tensors (excluded from checkpoints)
        self.frozen_text_prefixes = (
            "text_transformer.", "token_embedding.", "ln_final.",
            "positional_embedding", "text_projection",
        )

        self._stage = None
        self._text_cache = None
        self.set_stage(1)

    # -- input adaptation: shared 128x128 tensor -> CLIP native 224 --
    def _resize(self, x):
        if x.shape[-1] != self.INPUT_RESOLUTION:
            x = F.interpolate(x, size=(self.INPUT_RESOLUTION,) * 2, mode="bilinear",
                              align_corners=False)
        return x

    def _encode_text(self, prompts):
        x = prompts + self.positional_embedding.float()
        x = x.permute(1, 0, 2)
        x = self.text_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).float()
        eot = self.prompt_learner.tokenized_prompt.argmax(dim=-1).item()
        x = x[:, eot, :] @ self.text_projection.float()
        return x

    def encode_image(self, x):
        return self.visual(self._resize(x).type(self.visual.conv1.weight.dtype)).float()

    def encode_image_norm(self, x):
        return F.normalize(self.encode_image(x), dim=1)

    def text_features(self, class_ids=None, normed=True):
        feats = self._encode_text(self.prompt_learner(class_ids))
        return F.normalize(feats, dim=1) if normed else feats

    def set_stage(self, stage: int):
        if stage == self._stage:
            return
        self._stage = stage
        train_visual = stage == 2
        set_requires_grad(self.visual, train_visual)
        self.prompt_learner.cls_ctx.requires_grad_(stage == 1)
        self.bnneck.weight.requires_grad_(stage == 2)
        set_requires_grad(self.classifier, stage == 2)
        self._text_cache = None
        if stage == 2:
            with torch.no_grad():
                self._text_cache = self.text_features(None, normed=True).detach()

    def forward(self, x):                               # inference embedding
        return self.encode_image(x)

    def forward_stage2(self, x):
        img = self.encode_image(x)
        id_logits = self.classifier(self.bnneck(img))
        text = self._text_cache
        if text is None or text.device != img.device:
            with torch.no_grad():
                text = self.text_features(None, normed=True).detach()
            self._text_cache = text
        i2t_logits = self.logit_scale.exp() * F.normalize(img, dim=1) @ text.t()
        return img, id_logits, i2t_logits


def build_model(method_cfg, device: str = "cuda", num_classes: int | None = None):
    """Return the model for ``method_cfg`` (already moved to ``device``)."""
    if method_cfg.loss == "clip_reid":
        if num_classes is None:
            raise ValueError(f"method '{method_cfg.name}' needs num_classes")
        model = CLIPReID(
            num_classes=num_classes,
            arch=method_cfg.clip_arch,
            pretrained=method_cfg.pretrained,
            n_ctx=method_cfg.clip_n_ctx,
        )
        return model.to(device)

    if method_cfg.loss in ("bot", "agw"):
        if num_classes is None:
            raise ValueError(f"method '{method_cfg.name}' needs num_classes")
        if method_cfg.backbone == "resnet50":
            model = ReIDResNet(
                num_classes=num_classes,
                pretrained=method_cfg.pretrained,
                last_stride1=method_cfg.last_stride1,
                non_local=method_cfg.non_local,
                pooling=method_cfg.pooling,
            )
        else:
            model = ReIDTimmBackbone(
                _TIMM_REID_NAMES[method_cfg.backbone],
                num_classes=num_classes,
                pretrained=method_cfg.pretrained,
                pooling=method_cfg.pooling,
            )
        if method_cfg.feature_extract:
            for name, p in model.named_parameters():
                if not (name.startswith("bnneck") or name.startswith("classifier")):
                    p.requires_grad_(False)
        model.embedding_dim = method_cfg.embedding_dim  # keep config/inventory in sync
        return model.to(device)

    if method_cfg.backbone not in _BACKBONES:
        raise KeyError(f"unknown backbone '{method_cfg.backbone}'. available: {sorted(_BACKBONES)}")
    backbone, feat_dim = _BACKBONES[method_cfg.backbone](method_cfg.pretrained)
    if method_cfg.feature_extract:
        set_requires_grad(backbone, False)
    model = EmbeddingNet(
        backbone=backbone,
        feat_dim=feat_dim,
        embedding_dim=method_cfg.embedding_dim,
        bnneck=method_cfg.bnneck,
    )
    return model.to(device)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def checkpoint_state_dict(model: nn.Module) -> dict:
    """state_dict for saving; drops always-frozen tensors restored from pretrained
    weights on load (currently the CLIP-ReID text tower)."""
    sd = model.state_dict()
    prefixes = tuple(getattr(model, "frozen_text_prefixes", ()))
    if prefixes:
        sd = {k: v for k, v in sd.items() if not k.startswith(prefixes)}
    return sd


def load_checkpoint(model: nn.Module, state: dict) -> nn.Module:
    strict = not getattr(model, "frozen_text_prefixes", None)
    model.load_state_dict(state, strict=strict)
    return model
