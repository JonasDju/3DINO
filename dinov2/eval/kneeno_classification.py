#!/usr/bin/env python
"""Standalone KneeNo classification evaluation of a 3DINO checkpoint.

Loads the frozen teacher backbone from a 3DINO checkpoint -- the official weights or a
``teacher_checkpoint.pth`` written by ``dinov2/train/train3d.py::do_test`` (both are
``{"teacher": {"backbone.*": ..., "dino_head.*": ..., ...}}``) -- and hands it to
``kneeno.evaluation.ClassificationEvaluator``, which logs each task's metrics once per test set
(``eval.data.test_sets``; heads early-stopped on the val split). The counterpart of vjepa2's ``app/vjepa_2_1/eval_classification.py``.
Every task with a non-null ``eval.freq`` > 0 runs once (as in curia's and lightly-train's standalone evaluations).

The config file has three blocks:

- ``model:`` the architecture, with the key names of 3DINO's ``student:`` config block (``arch``,
  ``patch_size``, ``block_chunks``, ...). Omitted keys fall back to ``dinov2/configs/ssl3d_default_config.yaml``,
  which matches the official ViT-L weights. ``img_size`` is not a key: it is read from the checkpoint's
  positional embedding.
- ``transform:`` ``image_size`` (cube edge length every volume is resized to; null -> the checkpoint's
  pretraining size) and ``crop_foreground``; see ``DINO3DAdapter``.
- ``eval:`` KneeNo's evaluation config (see ``kneeno/evaluation/config.py::DEFAULT_EVAL_CONFIG``).

Next to ``eval.logging.per_label_dir`` the script writes the expanded config and command line (``params.yaml``, once
the checkpoint, dataset and split are loaded, before evaluating) and the final metrics (``results.json``, the format
of curia's and lightly-train's standalone evaluations); without a ``per_label_dir``, only ``results.json``, next to
the checkpoint. It refuses to start if earlier results exist.

``FSDPCheckpointer``'s ``model_*.rank_*.pth`` training checkpoints are sharded and cannot be loaded here; use the
``<output_dir>/eval/training_<iteration>/teacher_checkpoint.pth`` written next to them.

Usage (from the repository root)::

    .venv/bin/python -m dinov2.eval.kneeno_classification \\
        --config dinov2/configs/eval/kneeno-vitl-official.yaml \\
        --checkpoint /path/to/3dino_vit_weights.pth \\
        --device cuda:0
"""

import argparse
import json
import logging
import os
import pprint
from pathlib import Path

import torch
import yaml
from kneeno.config import expand_env_vars
from kneeno.evaluation import ClassificationEvaluator, enabled_tasks, load_eval_config
from omegaconf import OmegaConf

from dinov2.configs import dinov2_default_config_3d
from dinov2.data.kneeno_adapter import DINO3DAdapter

logger = logging.getLogger("dinov2")

# transform: keys and their defaults
TRANSFORM_DEFAULTS = {"image_size": None, "crop_foreground": True}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="config file (model + transform + eval blocks)")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a 3DINO teacher checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cpu", help="e.g. 'cpu' or 'cuda:0'")
    return parser.parse_args(argv)


def load_backbone_state_dict(checkpoint_path):
    """The teacher backbone's weights from a 3DINO checkpoint, without the ``backbone.`` prefix.

    The DINO / iBOT heads stored next to it are dropped: evaluation only needs the backbone.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "teacher" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} has no 'teacher' entry, only {list(checkpoint)}. Expected the official weights or a "
            "teacher_checkpoint.pth written by train3d.py (FSDP training checkpoints are sharded and not supported)"
        )
    state_dict = {k.removeprefix("backbone."): v for k, v in checkpoint["teacher"].items() if k.startswith("backbone.")}
    if not state_dict:
        raise KeyError(f"{checkpoint_path}'s 'teacher' entry has no 'backbone.*' weights")
    return state_dict


def infer_img_size(state_dict, patch_size):
    """The pretraining crop size, from the positional embedding: one cls token plus a cubic patch grid."""
    num_patches = state_dict["pos_embed"].shape[1] - 1
    grid = round(num_patches ** (1 / 3))
    if grid**3 != num_patches:
        raise ValueError(f"pos_embed has {num_patches} patch positions, which is not a cubic grid")
    return grid * patch_size


def build_backbone(cfg_model, img_size):
    """Build the teacher backbone the way 3DINO does (``dinov2.models.build_model``).

    ``cfg_model`` is merged over the ``student:`` block of 3DINO's default config, in struct mode, so a key that
    does not exist there (e.g. a typo) raises instead of being silently ignored.
    """
    # Imported here, not at the top: dinov2.layers decides at import time whether to use xFormers, and main()
    # disables it for CPU runs first
    from dinov2.models import build_model

    student_cfg = OmegaConf.create(OmegaConf.to_container(dinov2_default_config_3d.student))
    OmegaConf.set_struct(student_cfg, True)
    student_cfg = OmegaConf.merge(student_cfg, cfg_model or {})
    backbone, _ = build_model(student_cfg, only_teacher=True, img_size=img_size)
    return backbone


def load_frozen_backbone(backbone, state_dict):
    """Load ``state_dict`` into ``backbone`` and freeze it.

    Strict, unlike 3DINO's own ``load_pretrained_weights``: a missing, unexpected or differently shaped parameter
    means the ``model:`` block does not describe the checkpoint, and a partly loaded backbone would silently be
    evaluated with (partly) random weights.
    """
    try:
        backbone.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"The checkpoint does not match the architecture of the config's model: block. {e}") from e
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone


def get_output_paths(cfg_eval, checkpoint_path):
    """``results.json`` and ``params.yaml`` in the parent of ``eval.logging.per_label_dir``.

    Without a ``per_label_dir`` there is no evaluation folder: ``results.json`` goes next to the checkpoint and no
    ``params.yaml`` is written (as in lightly-train's standalone evaluation).
    """
    label_dir = (cfg_eval.get("logging") or {}).get("per_label_dir")
    if label_dir is None:
        results_path = Path(checkpoint_path).resolve().parent / "results.json"
        logger.warning(
            f"eval.logging.per_label_dir not set: writing the metrics to {results_path} and not saving the "
            "evaluation params to disk"
        )
        return results_path, None
    out_dir = Path(label_dir).resolve().parent
    return out_dir / "results.json", out_dir / "params.yaml"


def check_no_previous_results(cfg_eval, results_path, params_path):
    """Abort if a previous evaluation already wrote to the configured outputs."""
    cfg_logging = cfg_eval.get("logging") or {}
    paths = [results_path, params_path, cfg_logging.get("tensorboard_dir"), cfg_logging.get("per_label_dir")]
    existing = [str(path) for path in paths if path is not None and Path(path).exists()]
    if existing:
        raise ValueError(f"Evaluation results already exist: {existing}")


def save_params(config, args, params_path):
    """Save the config and the command line as ``params.yaml`` (nothing without a ``per_label_dir``)."""
    if params_path is None:
        return
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params = {**config, "cli": {"checkpoint": args.checkpoint, "device": args.device}}
    with open(params_path, "w") as f:
        yaml.dump(params, f)


def save_results(metrics, results_path):
    """Save the final metrics as ``results.json``, in the format of curia's and lightly-train's standalone evals."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info(f"wrote the metrics to {results_path}")


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    device = torch.device(args.device)
    if device.type == "cpu":
        # xFormers' attention kernels are CUDA-only; this makes 3DINO's MemEffAttention fall back to plain attention
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    with open(args.config, "r") as f:
        config = expand_env_vars(yaml.safe_load(f))
    logger.info("loaded config:\n%s", pprint.pformat(config))

    cfg_eval = config.get("eval")
    if cfg_eval is None:
        raise ValueError(f"{args.config} has no 'eval:' block")
    freq = load_eval_config(cfg_eval)["freq"]
    tasks = enabled_tasks(freq)
    if not tasks:
        raise ValueError(f"No task is enabled: every task in eval.freq is null or <= 0 ({freq}).")
    cfg_transform = {**TRANSFORM_DEFAULTS, **(config.get("transform") or {})}
    unknown = set(cfg_transform) - set(TRANSFORM_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown transform: keys {sorted(unknown)}, expected a subset of {list(TRANSFORM_DEFAULTS)}")

    results_path, params_path = get_output_paths(cfg_eval, args.checkpoint)
    check_no_previous_results(cfg_eval, results_path, params_path)

    state_dict = load_backbone_state_dict(args.checkpoint)
    patch_size = (config.get("model") or {}).get("patch_size", dinov2_default_config_3d.student.patch_size)
    img_size = infer_img_size(state_dict, patch_size)
    backbone = build_backbone(config.get("model"), img_size)
    backbone = load_frozen_backbone(backbone, state_dict).to(device)
    logger.info(f"loaded the teacher backbone from {args.checkpoint} (pretraining crop size {img_size})")
    logger.info(f"evaluating tasks {tasks}")

    image_size = cfg_transform["image_size"] or img_size
    if image_size != img_size:
        logger.info(
            f"transform.image_size {image_size} differs from the pretraining crop size {img_size}: the positional "
            "embedding is interpolated"
        )

    adapter = DINO3DAdapter(
        embed_dim=backbone.embed_dim, image_size=image_size, crop_foreground=cfg_transform["crop_foreground"]
    )

    evaluator = ClassificationEvaluator(config=cfg_eval, adapter=adapter, device=device)
    # After everything that fails fast (the checks, the strict load, and the dataset and split the evaluator builds):
    # a config they reject must not leave a params.yaml that makes the corrected rerun refuse. But before evaluating,
    # so a run that crashes still records its settings
    save_params(config, args, params_path)
    try:
        metrics = evaluator.evaluate(backbone, tasks=tasks, epoch=0)
    finally:
        # Flush+close the TensorBoard writer so buffered scalars are not lost if the process exits right after
        evaluator.cleanup()

    logger.info("final metrics: %s", metrics)
    pprint.pprint(metrics)
    save_results(metrics, results_path)
    return metrics


if __name__ == "__main__":
    main()
