"""
sentinelai.models.env_model
----------------------------
Environmental scene-understanding encoder module.

Responsibilities
----------------
* Classify the high-level scene context (indoor / outdoor, lighting conditions,
  crowd density, etc.).
* Produce a dense scene embedding used by the fusion head for contextual
  reasoning.
* Optionally perform lightweight semantic segmentation to provide spatial
  context maps.

Architecture sketch
-------------------
  Input (B, C, H, W)
      │
  [Scene Backbone CNN]    ← e.g. EfficientNet / Swin Transformer
      │
  [Global Average Pool]
      │
  [Projection MLP]
      │
  env_embedding (B, D)   → fed into FusionModel
      │ (optional branch)
  [Segmentation Decoder] → semantic map (B, num_seg_classes, H', W')

TODO
----
* Choose scene-understanding backbone (EfficientNet-B2 vs. Swin-T).
* Add depth-estimation head for 3-D spatial awareness.
* Integrate optional semantic segmentation decoder (DeepLab-v3+ or FPN).
* Write unit tests in tests/test_env_model.py.
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

SCENE_CLASSES = [
    "indoor_office",
    "indoor_home",
    "indoor_retail",
    "outdoor_urban",
    "outdoor_nature",
    "vehicle_interior",
    "unknown",
]

LIGHTING_CONDITIONS = ["bright", "dim", "dark", "backlit"]

DEFAULT_EMBEDDING_DIM = 256


# ---------------------------------------------------------------------------
# Helper: Scene projection MLP
# ---------------------------------------------------------------------------

class SceneProjection(nn.Module):
    """Maps high-dimensional backbone features to a compact scene embedding."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # spatial → (B, C, 1, 1)
            nn.Flatten(),              # → (B, C)
            nn.Linear(in_features, out_features * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_features * 2, out_features),
            nn.LayerNorm(out_features),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.net(feature_map)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class EnvModel(nn.Module):
    """
    Environment / scene-understanding encoder.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the output environment embedding vector.
    pretrained : bool
        Load ImageNet-pretrained weights for the backbone.
    num_scene_classes : int, optional
        If provided, attach a scene-classification head.
    num_seg_classes : int, optional
        If provided, attach a lightweight segmentation decoder.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        pretrained: bool = True,
        num_scene_classes: Optional[int] = None,
        num_seg_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        # ── Backbone ────────────────────────────────────────────────────────
        weights = "IMAGENET1K_V1" if pretrained else None
        _full_net = tv_models.efficientnet_b2(weights=weights)
        # Keep only the feature extractor (strip classifier)
        self.backbone = _full_net.features  # (B, 1408, H', W')
        backbone_channels = 1408           # EfficientNet-B2 final feature channels

        # ── Scene projection ─────────────────────────────────────────────────
        self.projection = SceneProjection(
            in_features=backbone_channels,
            out_features=embedding_dim,
        )

        # ── Optional scene classification head ──────────────────────────────
        self.scene_head: Optional[nn.Linear] = None
        if num_scene_classes is not None:
            self.scene_head = nn.Linear(embedding_dim, num_scene_classes)
            logger.info(
                "EnvModel: scene-classification head added (%d classes).",
                num_scene_classes,
            )

        # ── Optional segmentation decoder (lightweight) ──────────────────────
        self.seg_decoder: Optional[nn.Sequential] = None
        if num_seg_classes is not None:
            # TODO: Replace with a proper FPN / DeepLab-v3+ decoder
            self.seg_decoder = nn.Sequential(
                nn.ConvTranspose2d(backbone_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, num_seg_classes, kernel_size=4, stride=2, padding=1),
            )
            logger.info(
                "EnvModel: segmentation decoder added (%d classes).", num_seg_classes
            )

        self.embedding_dim = embedding_dim
        logger.info(
            "EnvModel initialised | emb_dim=%d | pretrained=%s", embedding_dim, pretrained
        )

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
            Batched scene images of shape (B, C, H, W).

        Returns
        -------
        dict with keys:
            ``"embedding"``      — (B, embedding_dim).
            ``"scene_logits"``   — (B, num_scene_classes) if head is attached.
            ``"seg_map"``        — (B, num_seg_classes, H', W') if decoder is attached.
        """
        feature_map = self.backbone(x)            # (B, 1408, H', W')
        embedding = self.projection(feature_map)  # (B, embedding_dim)

        output: dict[str, torch.Tensor] = {"embedding": embedding}

        if self.scene_head is not None:
            output["scene_logits"] = self.scene_head(embedding)

        if self.seg_decoder is not None:
            output["seg_map"] = self.seg_decoder(feature_map)

        return output
