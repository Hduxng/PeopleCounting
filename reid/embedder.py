"""
Re-ID feature extractor using OSNet (torchreid).

Stability improvements vs default MobileNet embedder:
  1. OSNet pre-trained on Market-1501 (person Re-ID), not ImageNet
  2. Minimum crop guard — tiny crops are padded before inference, preventing
     garbage features that cause ID switches
  3. Test-Time Augmentation — averages normal + h-flipped embeddings
  4. L2 normalisation — required for correct cosine distance in DeepSORT
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F


_REID_H, _REID_W = 256, 128   # Standard person Re-ID input resolution
_MIN_H,  _MIN_W  = 32,  16    # Crops smaller than this produce garbage features


def _prepare_crop(bgr: np.ndarray) -> np.ndarray:
    """
    BGR → RGB + edge-pad if crop is smaller than _MIN_H × _MIN_W.
    Padding replicates border pixels so OSNet sees a plausible appearance.
    """
    rgb = bgr[..., ::-1].copy()
    h, w = rgb.shape[:2]
    if h < _MIN_H or w < _MIN_W:
        pad_h = max(0, _MIN_H - h)
        pad_w = max(0, _MIN_W - w)
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return rgb


class OSNetEmbedder:
    """
    OSNet-based Re-ID embedder compatible with deep-sort-realtime.

    Receives a list of BGR numpy crops, returns a list of 1-D feature vectors.

    Args:
        model_name:   osnet_x0_25 / x0_5 / x0_75 / x1_0 / osnet_ain_x1_0
        weights_path: path to .pth weights; None = auto-download Market-1501
        device:       "auto" | "cuda" | "cpu"
        tta:          Test-Time Augmentation (h-flip averaging)
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        weights_path: str | None = None,
        device: str = "auto",
        half: bool = False,
        tta: bool = True,
    ):
        try:
            from torchreid.reid.utils import FeatureExtractor
        except ImportError:
            raise ImportError(
                "torchreid is required for OSNet Re-ID.\n"
                "Install: pip install torchreid gdown tensorboard"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not device.startswith("cuda"):
            half = False   # FP16 not supported on CPU

        self.device = device
        self.half   = half
        self.tta    = tta

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._extractor = FeatureExtractor(
                model_name=model_name,
                model_path=weights_path or "",
                image_size=(_REID_H, _REID_W),
                device=device,
                verbose=False,
            )

        if half:
            self._extractor.model.half()

    def _extract_features(self, rgb_imgs: list[np.ndarray]) -> torch.Tensor:
        images = [
            self._extractor.preprocess(self._extractor.to_pil(img))
            for img in rgb_imgs
        ]
        images = torch.stack(images, dim=0).to(self._extractor.device)
        if self.half:
            images = images.half()
        return self._extractor.model(images)

    @torch.no_grad()
    def __call__(self, crops: list) -> list:
        """
        Args:
            crops: list of BGR numpy arrays (H×W×3, uint8)
        Returns:
            list of L2-normalised 512-D feature vectors
        """
        if not crops:
            return []

        rgb_imgs = [_prepare_crop(c) for c in crops]

        feats = self._extract_features(rgb_imgs)

        if self.tta:
            rgb_flip  = [img[:, ::-1, :].copy() for img in rgb_imgs]
            feats_flip = self._extract_features(rgb_flip)
            feats = (feats + feats_flip) * 0.5

        feats = F.normalize(feats.float(), dim=1)   # normalize in FP32 for precision
        return feats.cpu().numpy().tolist()
