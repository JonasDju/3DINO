import torch
from kneeno.evaluation.adapter import EncoderAdapter
from monai.data import MetaTensor
from monai.transforms import Compose, CropForeground, Resize, ScaleIntensity, ScaleIntensityRangePercentiles, \
    Orientation
from torch.nn.parallel import DistributedDataParallel

from dinov2.fsdp import reshard_fsdp_model

#: World axis and sign of each anatomical direction in MONAI's (nibabel's) RAS world space.
_RAS_DIRECTIONS = {"R": (0, 1.0), "L": (0, -1.0), "A": (1, 1.0), "P": (1, -1.0), "S": (2, 1.0), "I": (2, -1.0)}


def _foreground(x):
    return x > -1


def affine_from_axcodes(axcodes):
    """RAS-space affine of a volume whose spatial array axes increase toward ``axcodes``, e.g. ``"LIP"``.

    The affine has unit spacing and a zero origin. ``Orientation`` only reads the direction of each axis, and
    none of the later transforms reads the affine.
    """
    if len(axcodes) != 3 or any(code not in _RAS_DIRECTIONS for code in axcodes):
        raise ValueError(f"Expected a three-letter orientation code of R/L, A/P, S/I, got {axcodes!r}")
    affine = torch.eye(4, dtype=torch.float64)
    affine[:3, :3] = 0
    for axis, code in enumerate(axcodes):
        world_axis, sign = _RAS_DIRECTIONS[code]
        affine[world_axis, axis] = sign
    if not affine[:3, :3].abs().sum(dim=1).eq(1).all():
        raise ValueError(f"Orientation code {axcodes!r} does not name each of R/L, A/P and S/I exactly once")
    return affine


class DINO3DAdapter(EncoderAdapter):
    """``EncoderAdapter`` for 3DINO's ``DinoVisionTransformer3d`` (the teacher backbone).

    Bridges 3DINO to ``kneeno.evaluation``'s unified feature format, as ``VJepa21Adapter`` (vjepa2) and
    ``DINOv2Adapter`` (lightly-train) do for the other two models. Like DINOv2, 3DINO produces a cls token
    (``x_norm_clstoken``), so ``has_cls_token`` is True and KneeNo's ``linear`` task is available in addition to
    ``linear_pool``.

    ``prepare_input`` mirrors the validation transforms of 3DINO's own classification evaluation
    (``dinov2/data/transforms.py::make_classification_transform_3d`` with ``min_int=-1``): reorient to RAS, scale
    intensities to [-1, 1], crop to the foreground, resize to an ``image_size`` cube.

    :param embed_dim: feature dimension of the backbone, i.e. ``model.embed_dim``.
    :param image_size: edge length of the cube every volume is resized to. Should match the checkpoint's
        ``crops.global_crops_size``; any other size works, but interpolates the positional embedding.
    :param crop_foreground: crop each volume to its foreground (every voxel above the minimum) before resizing,
        as 3DINO's protocol does. False resizes the whole volume.
    """

    has_cls_token = True

    def __init__(self, dataset_type, embed_dim, image_size, crop_foreground=True):
        self.dataset_type = dataset_type
        self._embed_dim = embed_dim
        self.image_size = image_size
        self.crop_foreground = crop_foreground

        self.orientation = Orientation(axcodes="RAS")

        assert dataset_type in ["internal", "external"]
        self.scale_intensity = (
            ScaleIntensity(minv=-1, maxv=1) if self.dataset_type == "internal" else
            ScaleIntensityRangePercentiles(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True, channel_wise=True)
        )

        self.common_transforms = Compose(
            [
                *([CropForeground(select_fn=_foreground)] if crop_foreground else []),
                Resize(spatial_size=(self.image_size, self.image_size, self.image_size), mode="trilinear"),
            ]
        )

    @property
    def embed_dim(self):
        return self._embed_dim

    def prepare_input(self, volume, orientation=None):
        """``(1, D, H, W)`` raw volume in [0, 255] (internal) or unscaled (external) ->
                ``(1, image_size, image_size, image_size)`` in [-1, 1].

        ``orientation`` is the volume's orientation code (one letter per ``D``, ``H``, ``W`` axis), as the labeled
        KneeNo datasets provide it. 3DINO's pipeline loads a NIfTI with ``LoadImaged``, which keeps the
        file's affine, and then reorients with ``Orientationd(axcodes="RAS")``. The KneeNo volumes carry no affine,
        so one is built from the code, and the same ``Orientation`` brings the volume into the same RAS
        layout. That makes the result independent of how the dataset lays out its axes. For an external NIfTI
        this gives the array 3DINO's own pipeline gives for the same file (pinned by
        ``tests/test_kneeno_adapter.py``).

        With our internal evaluation dataset, it is not possible to perfectly recreate the validation transforms
        3DINO expects, since it is already histogram normalized and rounded to uint8. While 3DINO uses
        ScaleIntensityRangePercentiles, we approximate this by assuming the output of the internal dataset is already
        percentile-normalized and just scale it to the right range [-1, 1].

        The external evaluation dataset can be loaded exactly as 3DINO expects
        """

        if orientation is None:
            raise ValueError(
                "DINO3DAdapter needs each volume's orientation to reorient it to RAS, as 3DINO's evaluation does; "
                "the dataset must provide `orientations` (the labeled KneeNo datasets do)"
            )

        # float32, as LoadImaged returns it; the affine stands in for the NIfTI header
        volume = MetaTensor(volume.float(), affine=affine_from_axcodes(orientation))
        volume = self.orientation(volume)  # spatial axes now increase toward R, A, S

        volume = self.scale_intensity(volume)  # differs depending on the evaluation dataset
        volume = self.common_transforms(volume)

        return volume.as_tensor()

    def forward_features(self, model, batch):
        """``batch``: ``(B, C, image_size, image_size, image_size)``

        ``model`` is either a plain ``DinoVisionTransformer3d`` (standalone evaluation of a loaded teacher
        checkpoint), a DDP-wrapped one, or the FSDP-wrapped teacher backbone from training
        (``SSLMetaArch.teacher.backbone``). With FSDP, every rank must run the same batches, since each forward
        all-gathers the parameters.
        """

        # DDP: unwrap, so a frozen forward pass does not go through DDP's bookkeeping. FSDP must *not* be
        # unwrapped: its inner module only holds the parameter shards.
        if isinstance(model, DistributedDataParallel):
            model = model.module
        # Through __call__, as the training loop calls the teacher, never model.forward_features(): on FSDP that
        # attribute resolves to the wrapped module's method and skips the parameter all-gather.
        out = model(batch, is_training=True)
        # A no_grad forward leaves FSDP's gathered parameters allocated (no backward pass frees them), so free
        # them as the training loop does after the teacher forward. No-op without FSDP.
        reshard_fsdp_model(model)
        return {"cls": out["x_norm_clstoken"], "patches": out["x_norm_patchtokens"]}
