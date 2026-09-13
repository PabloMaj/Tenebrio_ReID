"""Every model must build, embed to its declared dimension, and receive
gradients; every loss (metric / dedicated-re-ID combined / CLIP-ReID 2-stage)
must produce a finite, differentiable scalar that steps the optimiser."""

import copy
import importlib.util

import pytest
import torch

from config import METHODS
from losses import build_loss, build_optimizer, build_scheduler
from models import build_model, count_parameters

NUM_CLASSES = 9

# clip_reid needs the OpenAI CLIP package, installed separately (see requirements.txt):
#   pip install git+https://github.com/openai/CLIP.git
# Skip rather than fail when it isn't there - it's an optional extra, not a core dep.
HAS_CLIP = importlib.util.find_spec("clip") is not None
_NO_CLIP_REASON = "requires `pip install git+https://github.com/openai/CLIP.git`"
_skip_clip = pytest.mark.skipif(not HAS_CLIP, reason=_NO_CLIP_REASON)

ALL = [
    pytest.param(m, marks=_skip_clip) if m == "clip_reid" else m
    for m in METHODS.keys()
]


@pytest.fixture
def cpu_method(request):
    cfg = copy.deepcopy(METHODS[request.param])
    cfg.pretrained = False  # no downloads in tests
    return cfg


@pytest.mark.parametrize("cpu_method", ALL, indirect=True)
def test_model_forward_and_grad(cpu_method):
    model = build_model(cpu_method, device="cpu", num_classes=NUM_CLASSES)
    if cpu_method.loss == "clip_reid":
        model.set_stage(2)  # stage 1 freezes the image encoder by design
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    assert out.shape == (4, cpu_method.embedding_dim)
    assert torch.isfinite(out).all()

    counts = count_parameters(model)
    assert counts["trainable"] > 0
    if cpu_method.loss == "clip_reid":
        assert counts["frozen"] > 0        # frozen CLIP text tower
    else:
        bnneck = getattr(model, "bnneck", None)
        assert counts["frozen"] == (bnneck.bias.numel() if bnneck is not None else 0)

    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("cpu_method", ALL, indirect=True)
def test_loss_is_finite_and_trains(cpu_method):
    torch.manual_seed(0)
    model = build_model(cpu_method, device="cpu", num_classes=NUM_CLASSES)
    bundle = build_loss(cpu_method, num_classes=NUM_CLASSES, device="cpu", num_epochs=4)
    opt = build_optimizer(cpu_method, model)
    sched = build_scheduler(cpu_method, opt)

    x = torch.randn(8, 3, 128, 128)
    labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])

    losses_seen = []
    stages = [1, 1, 2, 2] if bundle.mode == "clip_reid" else [None] * 3
    for i, stg in enumerate(stages):
        opt.zero_grad()
        if bundle.loss_optimizer is not None:
            bundle.loss_optimizer.zero_grad()

        if bundle.mode == "clip_reid":
            model.set_stage(stg)
            if stg == 1:
                img = model.encode_image_norm(x)
                uniq = torch.unique(labels)
                txt = model.text_features(uniq, normed=True)
                loss = bundle.loss_func.stage1(txt, img, uniq, labels)
            else:
                img, id_logits, i2t = model.forward_stage2(x)
                loss = bundle.loss_func.stage2(img, id_logits, i2t, labels)
        elif bundle.mode == "reid_combined":
            feat, logits = model.forward_train(x)
            loss = bundle.loss_func(feat, logits, labels)
        else:
            emb = model(x)
            if bundle.miner is not None:
                it = bundle.miner(emb, labels)
                loss = bundle.loss_func(emb, labels, it)
            else:
                loss = bundle.loss_func(emb, labels)

        assert torch.isfinite(loss), f"non-finite loss at step {i}"
        loss.backward()
        opt.step()
        if bundle.loss_optimizer is not None:
            bundle.loss_optimizer.step()
        if sched is not None:
            sched.step()
        losses_seen.append(float(loss))

    assert all(v == v for v in losses_seen)  # not NaN


@pytest.mark.parametrize("cpu_method", ["bot_reid", "agw_reid", "convnextv2_reid"], indirect=True)
def test_reid_warmup_schedule(cpu_method):
    model = build_model(cpu_method, device="cpu", num_classes=NUM_CLASSES)
    opt = build_optimizer(cpu_method, model)
    sched = build_scheduler(cpu_method, opt)
    assert sched is not None
    lr0 = sched.get_last_lr()[0]
    for _ in range(cpu_method.warmup_epochs):
        sched.step()
    assert lr0 < sched.get_last_lr()[0] == pytest.approx(cpu_method.lr)


@pytest.mark.parametrize("cpu_method", [pytest.param("clip_reid", marks=_skip_clip)], indirect=True)
def test_clip_reid_two_stage(cpu_method):
    model = build_model(cpu_method, device="cpu", num_classes=NUM_CLASSES)

    model.set_stage(1)
    assert model.prompt_learner.cls_ctx.requires_grad
    assert not any(p.requires_grad for p in model.visual.parameters())

    model.set_stage(2)
    assert not model.prompt_learner.cls_ctx.requires_grad
    assert any(p.requires_grad for p in model.visual.parameters())
    assert model._text_cache is not None  # frozen prompt -> text features cached
