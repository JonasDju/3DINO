"""End-to-end CPU check for the standalone KneeNo classification evaluation
(``dinov2/eval/kneeno_classification.py``): builds a small checkpoint in the layout of the official weights /
``train3d.py::do_test``'s ``teacher_checkpoint.pth``, then drives the script's ``main()`` -- exercising config
parsing, architecture reconstruction, strict checkpoint loading and the standalone call into
``ClassificationEvaluator`` in one pass.

Run from the repository root: ``.venv/bin/python -m unittest tests.test_kneeno_eval_classification -v``
"""

import os

# xFormers' kernels are CUDA-only; must be set before dinov2.layers is imported
os.environ["XFORMERS_DISABLED"] = "1"

import sys
import tempfile
import unittest
from pathlib import Path

import kneeno
import torch
import yaml

from dinov2.eval.kneeno_classification import (
    build_backbone,
    infer_img_size,
    load_backbone_state_dict,
    load_frozen_backbone,
    main,
)

# KneeNo's synthetic labeled-dataset fixtures, from the sibling (editable) KneeNo checkout
sys.path.insert(0, str(Path(kneeno.__file__).resolve().parents[1] / "tests"))
from labeled_fixtures import (  # noqa: E402
    INTERNAL_SEQUENCES,
    full_exam_spec,
    make_internal_labeled_dataset,
    make_labeled_dataset,
)

ARCH = "vit_base_3d"
IMG_SIZE = 32  # pretraining crop size of the fake checkpoint: a 2^3 patch grid
SPEC = full_exam_spec(10, INTERNAL_SEQUENCES)  # 10 complete exams of native depths 6, 5, 4, 7
EXTERNAL_SPEC = full_exam_spec(10)


class EvalClassificationCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        torch.manual_seed(0)
        cls.backbone_state = build_backbone({"arch": ARCH}, IMG_SIZE).state_dict()
        # do_test()'s layout: the whole teacher (backbone + heads) under "teacher"
        teacher = {f"backbone.{k}": v for k, v in cls.backbone_state.items()}
        teacher["dino_head.mlp.0.weight"] = torch.randn(4, 768)
        cls.checkpoint_path = cls.root / "teacher_checkpoint.pth"
        torch.save({"teacher": teacher}, cls.checkpoint_path)
        cls.labeled_root = cls.root / "labeled"
        cls.labeled_meta_path = make_internal_labeled_dataset(cls.labeled_root, SPEC, h=20, w=24)
        cls.external_root = cls.root / "external"
        cls.external_meta_path = make_labeled_dataset(cls.external_root, EXTERNAL_SPEC, h=20, w=24)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(dir=self.root))

    def _write_config(self, model=None, transform=None, data=None):
        config = {
            "model": {"arch": ARCH} if model is None else model,
            "eval": {
                "seed": 1,
                "split": {"test_fraction": 0.3},
                "data": {
                    "data_root": str(self.labeled_root),
                    "label_meta": str(self.labeled_meta_path),
                    "dataset_type": "internal",
                    "series_depth": 0,
                    "num_workers": 0,
                    **(data or {}),
                },
                "logging": {
                    "tensorboard_dir": str(self.out / "tb"),
                    "per_label_dir": str(self.out / "per_label"),
                },
                "knn": {"k": [3], "batch_size": 4},
                "linear": {"epochs": 2, "batch_size": 4},
            },
        }
        if transform is not None:
            config["transform"] = transform
        path = self.out / "config.yaml"
        path.write_text(yaml.dump(config))
        return path

    def _run(self, config_path, tasks=("knn", "linear")):
        return main(["--config", str(config_path), "--checkpoint", str(self.checkpoint_path), "--tasks", *tasks])

    def test_cli_runs_and_refuses_to_overwrite_results(self):
        config_path = self._write_config()
        metrics = self._run(config_path)
        self.assertTrue(any(k.startswith("knn") for k in metrics), metrics)
        self.assertTrue(any(k.startswith("linear") for k in metrics), metrics)
        params = yaml.safe_load((self.out / "params.yaml").read_text())
        self.assertEqual(params["cli"]["checkpoint"], str(self.checkpoint_path))
        self.assertEqual(params["model"], {"arch": ARCH})
        with self.assertRaisesRegex(ValueError, "already exist"):
            self._run(config_path)

    def test_image_size_override_interpolates_positional_embedding(self):
        config_path = self._write_config(transform={"image_size": 48, "crop_foreground": False})
        metrics = self._run(config_path, tasks=("knn",))
        self.assertTrue(any(k.startswith("knn") for k in metrics), metrics)

    def test_external_dataset_runs(self):
        # Every external sequence is stored in its own orientation; the adapter reorients each to RAS.
        data = {
            "data_root": str(self.external_root),
            "label_meta": str(self.external_meta_path),
            "dataset_type": "external",
        }
        metrics = self._run(self._write_config(data=data), tasks=("knn",))
        self.assertTrue(any(k.startswith("knn") for k in metrics), metrics)

    def test_loads_every_backbone_weight_and_drops_the_heads(self):
        state_dict = load_backbone_state_dict(self.checkpoint_path)
        self.assertEqual(set(state_dict), set(self.backbone_state))
        self.assertEqual(infer_img_size(state_dict, 16), IMG_SIZE)
        backbone = load_frozen_backbone(build_backbone({"arch": ARCH}, IMG_SIZE), state_dict)
        for k, v in backbone.state_dict().items():
            self.assertTrue(torch.equal(v, self.backbone_state[k]), k)
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))
        self.assertFalse(backbone.training)

    def test_non_cubic_positional_embedding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a cubic grid"):
            infer_img_size({"pos_embed": torch.zeros(1, 1 + 12, 8)}, 16)

    def test_unknown_config_keys_raise(self):
        with self.assertRaises(Exception) as ctx:
            build_backbone({"arch": ARCH, "patch_sise": 16}, IMG_SIZE)
        self.assertIn("patch_sise", str(ctx.exception))
        with self.assertRaisesRegex(ValueError, "unknown transform"):
            self._run(self._write_config(transform={"img_size": 64}))

    def test_wrong_architecture_fails_the_strict_load(self):
        with self.assertRaisesRegex(RuntimeError, "does not match the architecture"):
            self._run(self._write_config(model={"arch": "vit_large_3d"}))

    def test_missing_teacher_entry_is_rejected(self):
        path = self.out / "not_a_teacher_checkpoint.pth"
        torch.save({"model": {}}, path)
        with self.assertRaisesRegex(KeyError, "no 'teacher' entry"):
            load_backbone_state_dict(path)


if __name__ == "__main__":
    unittest.main()
