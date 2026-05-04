"""
sentinelai.models.posture_model
--------------------------------
Human posture / body-pose estimation encoder module.

Responsibilities
----------------
* Detect human body keypoints (skeleton) from an image or video frame.
* Encode the skeleton graph into a fixed-dimensional posture embedding.
* Optionally classify posture into semantic categories (sitting, standing,
  slouching, etc.).

Architecture sketch
-------------------
  Input (B, C, H, W)
      │
  [Keypoint Detector]     ← e.g. HigherHRNet / MoveNet (pre-trained, frozen or fine-tuned)
      │
  keypoints (B, K, 3)     ← K joints × (x, y, confidence)
      │
  [Graph Encoder / MLP]   ← learn relational structure between joints
      │
  posture_embedding (B, D) → fed into FusionModel

TODO
----
* Integrate a keypoint detector (e.g. torchvision's KeypointRCNN as baseline).
* Implement GNN-based joint encoder for richer relational modelling.
* Add posture-class taxonomy and labelling guide.
* Write unit tests in tests/test_posture_model.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# COCO-style 17 keypoints used as default skeleton definition
COCO_NUM_KEYPOINTS = 17
DEFAULT_EMBEDDING_DIM = 256

POSTURE_CLASSES = [
    "standing",
    "sitting",
    "walking",
    "slouching",
    "lying_down",
    "unknown",
]


# ---------------------------------------------------------------------------
# Helper: MLP joint encoder
# ---------------------------------------------------------------------------

class JointMLP(nn.Module):
    """
    Flattens (K, 3) keypoints and passes them through an MLP.

    A Graph Neural Network (GNN) would capture joint relations more expressively;
    this MLP acts as a strong, simple baseline.
    """

    def __init__(self, num_keypoints: int, out_features: int, hidden: int = 256) -> None:
        super().__init__()
        in_features = num_keypoints * 3  # x, y, confidence per joint
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_features),
        )

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        # keypoints: (B, K, 3) → flatten → (B, K*3)
        B = keypoints.size(0)
        x = keypoints.view(B, -1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class PostureModel(nn.Module):
    """
    Posture-analysis encoder.

    Parameters
    ----------
    num_keypoints : int
        Number of skeleton keypoints produced by the upstream detector
        (default: 17 — COCO body keypoints).
    embedding_dim : int
        Dimension of the output posture embedding vector.
    num_posture_classes : int, optional
        If provided, attach a classification head.
    hidden_dim : int
        Width of the internal MLP layers.
    """

    def __init__(
        self,
        num_keypoints: int = COCO_NUM_KEYPOINTS,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_posture_classes: Optional[int] = None,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # ── Keypoint encoder ────────────────────────────────────────────────
        self.encoder = JointMLP(
            num_keypoints=num_keypoints,
            out_features=embedding_dim,
            hidden=hidden_dim,
        )

        # ── Optional classification head ─────────────────────────────────────
        self.posture_head: Optional[nn.Linear] = None
        if num_posture_classes is not None:
            self.posture_head = nn.Linear(embedding_dim, num_posture_classes)
            logger.info(
                "PostureModel: classification head added (%d classes).",
                num_posture_classes,
            )

        self.embedding_dim = embedding_dim
        logger.info(
            "PostureModel initialised | keypoints=%d | emb_dim=%d",
            num_keypoints,
            embedding_dim,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, keypoints: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        keypoints : torch.Tensor
            Skeleton keypoints of shape (B, K, 3) where the last dim is
            (normalised_x, normalised_y, confidence_score).

        Returns
        -------
        dict with keys:
            ``"embedding"`` — (B, embedding_dim).
            ``"posture_logits"`` — (B, num_posture_classes) if head is attached.
        """
        # TODO: Replace with a proper keypoint detector pipeline;
        #       for now we assume the caller supplies keypoints directly.
        embedding = self.encoder(keypoints)  # (B, embedding_dim)

        output: dict[str, torch.Tensor] = {"embedding": embedding}

        if self.posture_head is not None:
            output["posture_logits"] = self.posture_head(embedding)

        return output
