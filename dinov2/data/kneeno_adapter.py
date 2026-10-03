from kneeno.evaluation.adapter import EncoderAdapter
from monai.transforms import Compose, CropForeground, Resize, ScaleIntensity
from torch.nn.parallel import DistributedDataParallel

from dinov2.fsdp import reshard_fsdp_model


def _foreground(x):
    return x > -1


class DINO3DAdapter(EncoderAdapter):
    """``EncoderAdapter`` for 3DINO's ``DinoVisionTransformer3d`` (the teacher backbone).

    Bridges 3DINO to ``kneeno.evaluation``'s unified feature format, as ``VJepa21Adapter`` (vjepa2) and
    ``DINOv2Adapter`` (lightly-train) do for the other two models. Like DINOv2, 3DINO produces a cls token
    (``x_norm_clstoken``), so ``has_cls_token`` is True and KneeNo's ``linear`` task is available in addition to
    ``linear_pool``.

    ``prepare_input`` mirrors the validation transforms of 3DINO's own classification evaluation
    (``dinov2/data/transforms.py::make_classification_transform_3d`` with ``min_int=-1``): scale intensities to
    [-1, 1], crop to the foreground, resize to an ``image_size`` cube.

    :param embed_dim: feature dimension of the backbone, i.e. ``model.embed_dim``.
    :param image_size: edge length of the cube every volume is resized to. Should match the checkpoint's
        ``crops.global_crops_size``; any other size works, but interpolates the positional embedding.
    """

    has_cls_token = True

    def __init__(self, embed_dim, image_size):
        self._embed_dim = embed_dim
        self.image_size = image_size

        self.transforms = Compose(
            [
                ScaleIntensity(minv=-1, maxv=1),
                CropForeground(select_fn=_foreground),
                Resize(spatial_size=(self.image_size, self.image_size, self.image_size), mode="trilinear"),
            ]
        )

    @property
    def embed_dim(self):
        return self._embed_dim

    def prepare_input(self, volume):
        """``(1, D, H, W)`` raw volume in [0, 255] -> ``(1, image_size, image_size, image_size)`` in [-1, 1].

        With our internal evaluation dataset, it is not possible to perfectly recreate the validation transforms
        3DINO expects, since it is already histogram normalized and rounded to uint8. While 3DINO uses
        ScaleIntensityRangePercentiles, we approximate this by assuming the output of the internal dataset is already
        percentile-normalized and just scale it to the right range [-1, 1].
        """

        # Monai (and 3DINO) expect channel first, depth last
        volume = volume.permute(0, 2, 3, 1).numpy()  # C H W D
        return self.transforms(volume).as_tensor()

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
