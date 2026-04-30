"""
sentinelai.models.face_model
-----------------------------
Face-analysis encoder module.

Responsibilities
----------------
* Detect and align faces in an input image/video frame.
* Extract a fixed-dimensional embedding per face via a CNN backbone.
* Optionally predict auxiliary attributes (emotion, age, gaze direction).

Architecture sketch
-------------------
  Input (B, C, H, W)
      │
  [Backbone CNN]          ← e.g. ResNet-50 / MobileNetV3 / EfficientNet-B0
      │
  [Pooling + Projection]  ← reduce spatial dims → embedding vector
      │
  face_embedding (B, D)   → fed into FusionModel

TODO
----
* Choose / benchmark backbone (ResNet-50 vs. EfficientNet-B0).
* Integrate face-detection pre-processing (MTCNN / RetinaFace).
* Add auxiliary classification heads (emotion, age).
* Add data-augmentation hooks (random flip, colour jitter).
* Write unit tests in tests/test_face_model.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

SUPPORTED_BACKBONES = {
    "resnet50": tv_models.resnet50,
    "mobilenet_v3_small": tv_models.mobilenet_v3_small,
    "efficientnet_b0": tv_models.efficientnet_b0,
}

DEFAULT_EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Helper: Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Reduces backbone feature dimension to a fixed embedding size."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class FaceModel(nn.Module):
    """
    Face-analysis encoder.

    Parameters
    ----------
    backbone_name : str
        Key from ``SUPPORTED_BACKBONES``.
    embedding_dim : int
        Dimension of the output face embedding vector.
    pretrained : bool
        Load ImageNet-pretrained weights for the backbone.
    num_emotion_classes : int, optional
        If provided, attach an auxiliary emotion classifier head.
    freeze_backbone : bool
        If True, backbone weights are frozen (useful for fine-tuning only the head).
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        pretrained: bool = True,
        num_emotion_classes: Optional[int] = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from: {list(SUPPORTED_BACKBONES)}"
            )

        # ── Backbone ────────────────────────────────────────────────────────
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = SUPPORTED_BACKBONES[backbone_name](weights=weights)
        backbone_out_features = self._get_backbone_out_features(backbone_name)

        # Strip the original classifier / FC layer
        self._strip_classifier(backbone_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("FaceModel: backbone weights frozen.")

        # ── Projection head ─────────────────────────────────────────────────
        self.projection = ProjectionHead(
            in_features=backbone_out_features,
            out_features=embedding_dim,
        )

        # ── Auxiliary head: emotion ──────────────────────────────────────────
        self.emotion_head: Optional[nn.Linear] = None
        if num_emotion_classes is not None:
            self.emotion_head = nn.Linear(embedding_dim, num_emotion_classes)
            logger.info(
                "FaceModel: emotion head added (%d classes).", num_emotion_classes
            )

        self.embedding_dim = embedding_dim
        logger.info(
            "FaceModel initialised | backbone=%s | emb_dim=%d | pretrained=%s",
            backbone_name,
            embedding_dim,
            pretrained,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_backbone_out_features(self, name: str) -> int:
        """Return the feature dimension produced by the backbone (before its classifier)."""
        # TODO: extend mapping when adding new backbones
        mapping = {
            "resnet50": 2048,
            "mobilenet_v3_small": 576,
            "efficientnet_b0": 1280,
        }
        return mapping[name]

    def _strip_classifier(self, name: str) -> None:
        """Remove the classification head from the backbone in-place."""
        if name == "resnet50":
            self.backbone.fc = nn.Identity()
        elif name == "mobilenet_v3_small":
            self.backbone.classifier = nn.Identity()
        elif name == "efficientnet_b0":
            self.backbone.classifier = nn.Identity()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batched face-crop images of shape (B, C, H, W).
            Expected to be normalised (ImageNet mean/std or equivalent).

        Returns
        -------
        dict with keys:
            ``"embedding"`` — (B, embedding_dim) face embedding.
            ``"emotion_logits"`` — (B, num_emotion_classes) if emotion head is attached.
        """
        # TODO: Add face detection / alignment pre-processing step here
        features = self.backbone(x)          # (B, backbone_out)
        embedding = self.projection(features)  # (B, embedding_dim)

        output: dict[str, torch.Tensor] = {"embedding": embedding}

        if self.emotion_head is not None:
            output["emotion_logits"] = self.emotion_head(embedding)

        return output
