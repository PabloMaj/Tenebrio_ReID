"""Method registry and experiment configuration for the re-ID benchmark.

One "method" = one backbone + one metric-learning recipe.  Method ``proposed`` is
the previously published best model (MobileNetV2 fine-tuned with a triplet
margin loss, exactly the settings from ``params.txt``).  The other three are
newer SOTA options; they share the *training scheme* (5 splits, same
pre-processing, same triplet mining protocol / eval protocol, same epoch budget
and best-epoch selection) and only adapt the optimiser hyper-parameters to the
backbone, plus - for ``resnet50_arcface`` - the loss family.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from data import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class MethodConfig:
    name: str
    description: str

    # --- backbone ---
    backbone: str                     # key understood by models.build_model
    weights_source: str               # human readable, for the inventory
    pretrained: bool = True
    feature_extract: bool = False     # False -> full fine-tuning (preferred)
    embedding_dim: int = 512
    bnneck: bool = False
    input_adaptation: str = "none"    # e.g. bilinear resize for patch-14 ViTs

    # --- loss / miner ---
    loss: str = "triplet"             # triplet | subcenter_arcface | bot | agw
    distance: str = "cosine"          # "cosine" | "euclidean"
    margin_loss: float = 0.2
    margin_miner: float = 0.2
    type_of_triplets: str = "all"     # all | hard | semihard | easy
    arcface_margin: float = 28.6
    arcface_scale: float = 64.0
    arcface_sub_centers: int = 3

    # --- dedicated re-ID knobs (bot / agw / convnextv2 / clip) ---
    last_stride1: bool = False        # ResNet last stride 1 -> larger final map
    non_local: bool = False           # insert Non-local blocks (AGW)
    pooling: str = "avg"              # "avg" | "gem"  (GeM -> AGW / ConvNeXtV2)
    id_label_smoothing: float = 0.1
    triplet_margin_reid: float = 0.3
    warmup_epochs: int = 0            # linear LR warm-up (BoT / AGW)

    # --- CLIP-ReID knobs ---
    clip_arch: str = "ViT-B/16"
    clip_n_ctx: int = 4              # learnable context tokens per identity
    clip_stage1_frac: float = 0.3   # fraction of the epoch budget spent on prompt learning
    clip_prompt_lr: float = 3.5e-4
    clip_image_lr: float = 5e-6
    clip_head_lr: float = 3.5e-4

    # --- optimiser ---
    optimizer: str = "adam"           # "adam" | "adamw"
    lr: float = 0.01
    weight_decay: float = 0.0
    loss_lr: float = 1e-3             # optimiser LR for a parametric loss (ArcFace)

    # download hint surfaced to the user
    download_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    dataset_root: str
    part_for_analysis: str = "thorax"
    splits: tuple = (1, 2, 3, 4, 5)
    num_epochs: int = 200
    eval_every: int = 10
    batch_size: int = 128
    use_oversampling: bool = False           # the published split-CV run used False
    best_metric: str = "mean_average_precision_at_r"   # selected on the val set
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    num_workers: int = 0
    seed: int = 0
    output_dir: str = "runs"
    limit_classes: Optional[int] = None
    device: str = "cuda"


# --------------------------------------------------------------------------- #
#  The benchmarked methods: 1 proposed baseline + 5 comparison methods
#    - 3 modern general-purpose backbones (convnext_tiny, resnet50_arcface,
#      dinov2_vits14), the recipe kept as close to `proposed` as sensible
#    - 2 dedicated re-identification methods (bot_reid, agw_reid)
# --------------------------------------------------------------------------- #
METHODS: dict[str, MethodConfig] = {
    # 1) the proposed / previously-best model - exact published settings
    "proposed": MethodConfig(
        name="proposed",
        description="Proposed: MobileNetV2 backbone fine-tuned with TripletMarginLoss "
        "(published best settings from params.txt).",
        backbone="mobilenet_v2",
        weights_source="torchvision MobileNet_V2_Weights.IMAGENET1K_V1 (~14 MB, auto)",
        pretrained=True,
        feature_extract=False,
        embedding_dim=1024,
        loss="triplet",
        distance="cosine",
        margin_loss=0.2,
        margin_miner=0.2,
        type_of_triplets="all",
        optimizer="adam",
        lr=0.01,
        weight_decay=0.0,
    ),
    # 2) modern ConvNet, same triplet recipe
    "convnext_tiny": MethodConfig(
        name="convnext_tiny",
        description="ConvNeXt-Tiny (modern CNN, 2022) fine-tuned with the same "
        "TripletMarginLoss recipe as the proposed method.",
        backbone="convnext_tiny",
        weights_source="torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1 (~110 MB, auto)",
        pretrained=True,
        feature_extract=False,
        embedding_dim=1024,
        loss="triplet",
        distance="cosine",
        margin_loss=0.2,
        margin_miner=0.2,
        type_of_triplets="all",
        optimizer="adamw",
        lr=1e-4,
        weight_decay=0.05,
        download_note="ConvNeXt-Tiny ImageNet weights (~110 MB) downloaded automatically on first run.",
    ),
    # 3) strong ReID baseline: ResNet50 + BNNeck + SubCenter ArcFace
    "resnet50_arcface": MethodConfig(
        name="resnet50_arcface",
        description="ResNet50 + BNNeck fine-tuned with SubCenterArcFaceLoss "
        "(angular-margin metric learning, 2019/2020).",
        backbone="resnet50",
        weights_source="torchvision ResNet50_Weights.IMAGENET1K_V2 (~100 MB, auto)",
        pretrained=True,
        feature_extract=False,
        embedding_dim=512,
        bnneck=True,
        loss="subcenter_arcface",
        distance="cosine",
        arcface_margin=28.6,
        arcface_scale=64.0,
        arcface_sub_centers=3,
        optimizer="adam",
        lr=3.5e-4,
        weight_decay=5e-4,
        loss_lr=1e-3,
        download_note="ResNet50 ImageNet weights (~100 MB) downloaded automatically on first run.",
    ),
    # 4) strict re-ID: Bag of Tricks (Luo et al., CVPRW 2019)
    "bot_reid": MethodConfig(
        name="bot_reid",
        description="Strict re-ID: ResNet50 Bag-of-Tricks (CVPRW 2019) - last-stride-1, "
        "BNNeck, label-smoothed ID cross-entropy + batch-hard triplet, LR warm-up.",
        backbone="resnet50",
        weights_source="torchvision ResNet50_Weights.IMAGENET1K_V2 (~100 MB, auto)",
        pretrained=True,
        feature_extract=False,
        embedding_dim=2048,          # BNNeck feature used directly, no 512 projection
        bnneck=True,
        last_stride1=True,
        pooling="avg",
        loss="bot",
        distance="euclidean",
        id_label_smoothing=0.1,
        triplet_margin_reid=0.3,
        warmup_epochs=10,
        optimizer="adam",
        lr=3.5e-4,
        weight_decay=5e-4,
        download_note="ResNet50 ImageNet weights (~100 MB) downloaded automatically on first run.",
    ),
    # 5) strict re-ID: AGW (Ye et al., TPAMI 2021)
    "agw_reid": MethodConfig(
        name="agw_reid",
        description="Strict re-ID: AGW (TPAMI 2021) - ResNet50 + Non-local blocks + GeM "
        "pooling + BNNeck, label-smoothed ID cross-entropy + Weighted Regularization "
        "Triplet loss, LR warm-up.",
        backbone="resnet50",
        weights_source="torchvision ResNet50_Weights.IMAGENET1K_V2 (~100 MB, auto)",
        pretrained=True,
        feature_extract=False,
        embedding_dim=2048,
        bnneck=True,
        last_stride1=True,
        non_local=True,
        pooling="gem",
        loss="agw",
        distance="euclidean",
        id_label_smoothing=0.1,
        warmup_epochs=10,
        optimizer="adam",
        lr=3.5e-4,
        weight_decay=5e-4,
        download_note="ResNet50 ImageNet weights (~100 MB) downloaded automatically on first run.",
    ),
    # 6) strict re-ID, newest backbone: ConvNeXt V2 (CVPR 2023) + AGW-style recipe
    "convnextv2_reid": MethodConfig(
        name="convnextv2_reid",
        description="Strict re-ID: ConvNeXt V2-Tiny (CVPR 2023, FCMAE self-supervised "
        "pretraining) + GeM + BNNeck, label-smoothed ID cross-entropy + Weighted "
        "Regularization Triplet loss, LR warm-up.",
        backbone="convnextv2_tiny",
        weights_source="timm 'convnextv2_tiny.fcmae_ft_in1k' (~112 MB, auto from HF)",
        pretrained=True,
        feature_extract=False,
        embedding_dim=768,
        bnneck=True,
        pooling="gem",
        loss="agw",
        distance="euclidean",
        id_label_smoothing=0.1,
        warmup_epochs=10,
        optimizer="adamw",
        lr=1e-4,
        weight_decay=0.05,
        download_note="ConvNeXt V2-Tiny FCMAE weights (~112 MB) downloaded automatically on first run.",
    ),
    # 7) dedicated re-ID method: CLIP-ReID (Li et al., AAAI 2023)
    "clip_reid": MethodConfig(
        name="clip_reid",
        description="Strict re-ID: CLIP-ReID (AAAI 2023) - CLIP ViT-B/16 image encoder "
        "+ per-identity learnable text prompts, 2-stage training (prompt learning, then "
        "image-encoder fine-tuning with image->text CE + ID + batch-hard triplet).",
        backbone="clip_vitb16",
        weights_source="OpenAI CLIP 'ViT-B/16' (~335 MB, auto to ~/.cache/clip)",
        pretrained=True,
        feature_extract=False,
        embedding_dim=512,
        bnneck=True,
        loss="clip_reid",
        distance="cosine",
        input_adaptation="bilinear resize 128->224 (CLIP native resolution)",
        id_label_smoothing=0.1,
        triplet_margin_reid=0.3,
        clip_arch="ViT-B/16",
        clip_n_ctx=4,
        clip_stage1_frac=0.3,
        optimizer="adam",
        lr=5e-6,
        weight_decay=1e-4,
        download_note="OpenAI CLIP ViT-B/16 (~335 MB) downloaded automatically on first run "
        "(needs internet). Memory-heavy: use --batch-size 64 on GPUs with <16 GB.",
    ),
    # 8) self-supervised ViT backbone (DINOv2), same triplet recipe
    "dinov2_vits14": MethodConfig(
        name="dinov2_vits14",
        description="DINOv2 ViT-S/14 (self-supervised backbone, 2023) fine-tuned "
        "with the same TripletMarginLoss recipe as the proposed method.",
        backbone="dinov2_vits14",
        weights_source="timm 'vit_small_patch14_dinov2.lvd142m' (~85 MB, auto from HF)",
        pretrained=True,
        feature_extract=False,
        embedding_dim=1024,
        input_adaptation="bilinear resize 128->126 (patch-14 divisibility)",
        loss="triplet",
        distance="cosine",
        margin_loss=0.2,
        margin_miner=0.2,
        type_of_triplets="all",
        optimizer="adamw",
        lr=1e-5,
        weight_decay=0.05,
        download_note="DINOv2 ViT-S/14 weights (~85 MB) downloaded automatically on first run "
        "(needs internet). Set pretrained=False to skip.",
    ),
}

ALL_METHODS = list(METHODS.keys())
