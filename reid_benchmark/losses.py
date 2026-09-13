"""Loss / miner / (optional) loss-optimiser / LR-scheduler construction.

Recipes
-------
* ``triplet``            TripletMarginLoss + TripletMarginMiner (published recipe:
                         cosine distance, ThresholdReducer(low=0), type from config)
* ``subcenter_arcface``  SubCenterArcFaceLoss (parametric -> own optimiser), no miner
* ``bot``    Bag of Tricks (Luo et al., CVPRW 2019): label-smoothed ID
             cross-entropy + batch-hard TripletMarginLoss, LR warm-up
* ``agw``    AGW (Ye et al., TPAMI 2021): label-smoothed ID cross-entropy +
             Weighted Regularization Triplet loss, LR warm-up

``bot`` / ``agw`` run in ``mode="reid_combined"``: the engine calls
``model.forward_train(x) -> (pre_bnneck_feat, id_logits)`` and the loss consumes
both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners, reducers


@dataclass
class LossBundle:
    loss_func: object
    miner: Optional[object]
    loss_optimizer: Optional[torch.optim.Optimizer]
    mode: str = "metric"          # "metric" | "reid_combined" | "clip_reid"
    stage1_epochs: int = 0        # clip_reid: epochs of prompt-only training


def _distance(name: str):
    if name == "cosine":
        return distances.CosineSimilarity()
    if name == "euclidean":
        return distances.LpDistance(normalize_embeddings=True, p=2, power=1)
    raise ValueError(f"unknown distance '{name}'")


# --------------------------------------------------------------------------- #
#  AGW: Weighted Regularization Triplet loss (port of the official implementation)
# --------------------------------------------------------------------------- #
def _euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def _softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    return torch.exp(diff) * mask / Z


class WeightedRegularizationTriplet(nn.Module):
    def __init__(self, normalize_feature: bool = True):
        super().__init__()
        self.normalize_feature = normalize_feature
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, embedding, labels):
        if self.normalize_feature:
            embedding = F.normalize(embedding, p=2, dim=1)
        dist_mat = _euclidean_dist(embedding, embedding)
        n = dist_mat.size(0)
        is_pos = labels.expand(n, n).eq(labels.expand(n, n).t()).float()
        is_neg = labels.expand(n, n).ne(labels.expand(n, n).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg
        w_ap = _softmax_weights(dist_ap, is_pos)
        w_an = _softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * w_ap, dim=1)
        closest_negative = torch.sum(dist_an * w_an, dim=1)

        y = torch.ones_like(furthest_positive)
        return self.ranking_loss(closest_negative - furthest_positive, y)


class BoTLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, triplet_margin: float = 0.3,
                 id_weight: float = 1.0, triplet_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet = losses.TripletMarginLoss(
            margin=triplet_margin, distance=distances.LpDistance(normalize_embeddings=False)
        )
        self.miner = miners.BatchHardMiner()
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight

    def forward(self, feat_triplet, id_logits, labels):
        l_id = self.ce(id_logits, labels)
        hard = self.miner(feat_triplet, labels)
        l_tri = self.triplet(feat_triplet, labels, hard)
        return self.id_weight * l_id + self.triplet_weight * l_tri


class AGWLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, id_weight: float = 1.0,
                 triplet_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.wrt = WeightedRegularizationTriplet(normalize_feature=True)
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight

    def forward(self, feat_triplet, id_logits, labels):
        return (
            self.id_weight * self.ce(id_logits, labels)
            + self.triplet_weight * self.wrt(feat_triplet, labels)
        )


# --------------------------------------------------------------------------- #
#  CLIP-ReID (Li et al., AAAI 2023)
# --------------------------------------------------------------------------- #
class _SupConLoss(nn.Module):
    """Multi-positive contrastive loss between two L2-normalised feature sets."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, anchor, contrast, anchor_labels, contrast_labels):
        mask = torch.eq(anchor_labels.unsqueeze(1), contrast_labels.unsqueeze(0)).float()
        logits = anchor @ contrast.t() / self.t
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        exp = torch.exp(logits)
        log_prob = logits - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return -mean_log_prob_pos.mean()


class CLIPReIDLoss(nn.Module):
    """Stage 1: image<->text SupCon on the learnable prompts.
    Stage 2: label-smoothed ID CE + batch-hard triplet + image->text CE."""

    def __init__(self, label_smoothing: float = 0.1, triplet_margin: float = 0.3):
        super().__init__()
        self.supcon = _SupConLoss(temperature=1.0)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.i2t_ce = nn.CrossEntropyLoss()
        self.triplet = losses.TripletMarginLoss(
            margin=triplet_margin, distance=distances.LpDistance(normalize_embeddings=False)
        )
        self.miner = miners.BatchHardMiner()

    def stage1(self, text_feat, image_feat, text_labels, image_labels):
        return (
            self.supcon(image_feat, text_feat, image_labels, text_labels)
            + self.supcon(text_feat, image_feat, text_labels, image_labels)
        )

    def stage2(self, image_feat, id_logits, i2t_logits, labels):
        l_id = self.ce(id_logits, labels)
        hard = self.miner(image_feat, labels)
        l_tri = self.triplet(image_feat, labels, hard)
        l_i2t = self.i2t_ce(i2t_logits, labels)
        return l_id + l_tri + l_i2t


# --------------------------------------------------------------------------- #
def build_loss(method_cfg, num_classes: int, device: str = "cuda",
               num_epochs: Optional[int] = None) -> LossBundle:
    if method_cfg.loss == "clip_reid":
        loss_func = CLIPReIDLoss(
            label_smoothing=method_cfg.id_label_smoothing,
            triplet_margin=method_cfg.triplet_margin_reid,
        ).to(device)
        stage1 = max(1, round(method_cfg.clip_stage1_frac * (num_epochs or 4)))
        return LossBundle(loss_func, None, None, mode="clip_reid", stage1_epochs=stage1)

    if method_cfg.loss == "triplet":
        distance = _distance(method_cfg.distance)
        loss_func = losses.TripletMarginLoss(
            margin=method_cfg.margin_loss,
            distance=distance,
            reducer=reducers.ThresholdReducer(low=0),
            triplets_per_anchor="all",
        )
        miner = miners.TripletMarginMiner(
            margin=method_cfg.margin_miner,
            distance=distance,
            type_of_triplets=method_cfg.type_of_triplets,
        )
        return LossBundle(loss_func, miner, None, mode="metric")

    if method_cfg.loss == "subcenter_arcface":
        loss_func = losses.SubCenterArcFaceLoss(
            num_classes=num_classes,
            embedding_size=method_cfg.embedding_dim,
            margin=method_cfg.arcface_margin,
            scale=method_cfg.arcface_scale,
            sub_centers=method_cfg.arcface_sub_centers,
        ).to(device)
        loss_optimizer = torch.optim.AdamW(loss_func.parameters(), lr=method_cfg.loss_lr)
        return LossBundle(loss_func, None, loss_optimizer, mode="metric")

    if method_cfg.loss == "bot":
        loss_func = BoTLoss(
            label_smoothing=method_cfg.id_label_smoothing,
            triplet_margin=method_cfg.triplet_margin_reid,
        ).to(device)
        return LossBundle(loss_func, None, None, mode="reid_combined")

    if method_cfg.loss == "agw":
        loss_func = AGWLoss(label_smoothing=method_cfg.id_label_smoothing).to(device)
        return LossBundle(loss_func, None, None, mode="reid_combined")

    raise ValueError(f"unknown loss '{method_cfg.loss}'")


def build_optimizer(method_cfg, model) -> torch.optim.Optimizer:
    if method_cfg.loss == "clip_reid":
        # separate LRs: prompt / image-encoder / re-ID head.  Params are all
        # included; requires_grad is toggled per stage, Adam skips grad-less ones.
        groups = [
            {"params": list(model.prompt_learner.parameters()), "lr": method_cfg.clip_prompt_lr},
            {"params": list(model.visual.parameters()), "lr": method_cfg.clip_image_lr},
            {
                "params": list(model.bnneck.parameters()) + list(model.classifier.parameters()),
                "lr": method_cfg.clip_head_lr,
            },
        ]
        return torch.optim.Adam(groups, weight_decay=method_cfg.weight_decay)

    params = [p for p in model.parameters() if p.requires_grad]
    if method_cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=method_cfg.lr, weight_decay=method_cfg.weight_decay)
    if method_cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=method_cfg.lr, weight_decay=method_cfg.weight_decay)
    raise ValueError(f"unknown optimizer '{method_cfg.optimizer}'")


def build_scheduler(method_cfg, optimizer):
    """Linear LR warm-up for ``warmup_epochs`` epochs, then constant (BoT / AGW).

    Stepped once per epoch.  Returns ``None`` when no warm-up is configured, so
    every other method keeps the published constant-LR schedule.
    """
    w = getattr(method_cfg, "warmup_epochs", 0)
    if not w:
        return None

    def lr_lambda(epoch):  # epoch is 0-indexed by LambdaLR
        return float(epoch + 1) / float(w) if epoch < w else 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
