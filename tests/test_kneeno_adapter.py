"""``DINO3DAdapter.prepare_input`` against 3DINO's own classification validation transform.

The labeled KneeNo datasets return plain ``(1, D, H, W)`` tensors plus one orientation code per sequence. On
the external (NIfTI) data, the adapter must produce exactly what
``make_classification_transform_3d``'s validation pipeline (``LoadImaged`` -> ``Orientationd("RAS")`` ->
percentile scaling -> foreground crop -> resize) produces from the same file. This holds for every
sequence, each stored in a different orientation.

Run from the repository root: ``.venv/bin/python -m unittest tests.test_kneeno_adapter -v``
"""

import os

# xFormers' kernels are CUDA-only; must be set before dinov2.layers is imported
os.environ["XFORMERS_DISABLED"] = "1"

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import kneeno
import numpy as np
import torch
from kneeno.evaluation.dataset import LabeledExternalKneeMRIDataset
from monai.transforms import Compose, CropForegroundd

from dinov2.data.kneeno_adapter import DINO3DAdapter, affine_from_axcodes
from dinov2.data.transforms import make_classification_transform_3d

# KneeNo's synthetic labeled-dataset fixtures, from the sibling (editable) KneeNo checkout
sys.path.insert(0, str(Path(kneeno.__file__).resolve().parents[1] / "tests"))
from labeled_fixtures import EXTERNAL_SEQUENCES, full_exam_spec, make_labeled_dataset  # noqa: E402

IMAGE_SIZE = 16
H, W = 10, 7  # depths come from the fixture (6, 5, 4, 7): every axis of every volume has its own length


class PrepareInputMatches3DINOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        meta = make_labeled_dataset(cls.root, full_exam_spec(2), h=H, w=W, dtype=np.int16)
        cls.dataset = LabeledExternalKneeMRIDataset(cls.root, meta)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _reference(self, case_id, series, crop_foreground=True):
        """3DINO's validation pipeline straight from the file (without its foreground crop if not ``crop_foreground``)."""
        _, reference = make_classification_transform_3d("ICBM", IMAGE_SIZE, min_int=-1)
        if not crop_foreground:
            reference = Compose([t for t in reference.transforms if not isinstance(t, CropForegroundd)])
        # The ICBM label map needs an age-like label.
        sample = reference({"image": str(self.root / case_id / f"{series}.nii.gz"), "label": 25})
        return sample["image"].as_tensor()

    def test_every_sequence_matches_3dinos_pipeline_on_the_same_file(self):
        for crop_foreground in (True, False):
            adapter = DINO3DAdapter("external", embed_dim=8, image_size=IMAGE_SIZE, crop_foreground=crop_foreground)
            for index, (case_id, _, _) in enumerate(self.dataset.samples):
                volumes, _ = self.dataset[index]
                for series, volume, orientation in zip(EXTERNAL_SEQUENCES, volumes, self.dataset.orientations):
                    with self.subTest(crop_foreground=crop_foreground, case=case_id, series=series):
                        out = adapter.prepare_input(volume, orientation=orientation)
                        self.assertEqual(tuple(out.shape), (1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))
                        reference = self._reference(case_id, series, crop_foreground)
                        torch.testing.assert_close(out, reference, rtol=0, atol=1e-6)

    def test_the_orientation_decides_the_layout(self):
        # Without its own orientation (here: another sequence's), a volume no longer matches the reference.
        adapter = DINO3DAdapter("external", embed_dim=8, image_size=IMAGE_SIZE, crop_foreground=False)
        volumes, _ = self.dataset[0]
        case_id = self.dataset.samples[0][0]
        sag = adapter.prepare_input(volumes[0], orientation=self.dataset.orientations[2])  # coronal's code
        self.assertFalse(torch.allclose(sag, self._reference(case_id, "sag", crop_foreground=False), atol=1e-3))

    def test_prepare_input_requires_an_orientation(self):
        adapter = DINO3DAdapter("external", embed_dim=8, image_size=IMAGE_SIZE)
        volumes, _ = self.dataset[0]
        with self.assertRaisesRegex(ValueError, "orientation"):
            adapter.prepare_input(volumes[0])

    def test_adapter_pickles(self):
        # prepare_input runs in DataLoader workers
        adapter = DINO3DAdapter("external", embed_dim=8, image_size=IMAGE_SIZE)
        restored = pickle.loads(pickle.dumps(adapter))
        volumes, _ = self.dataset[0]
        code = self.dataset.orientations[0]
        self.assertTrue(torch.equal(restored.prepare_input(volumes[0], code), adapter.prepare_input(volumes[0], code)))


class AffineFromAxcodesTest(unittest.TestCase):
    def test_ras_is_the_identity(self):
        self.assertTrue(torch.equal(affine_from_axcodes("RAS"), torch.eye(4, dtype=torch.float64)))

    def test_each_axis_points_toward_its_letter(self):
        affine = affine_from_axcodes("LIP")  # axis 0 toward L (-x), 1 toward I (-z), 2 toward P (-y)
        expected = torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float64)
        self.assertTrue(torch.equal(affine[:3, :3], expected))

    def test_invalid_codes_raise(self):
        for code in ("LLP", "RA", "XYZ", "RASR"):
            with self.subTest(code=code), self.assertRaises(ValueError):
                affine_from_axcodes(code)


if __name__ == "__main__":
    unittest.main()
