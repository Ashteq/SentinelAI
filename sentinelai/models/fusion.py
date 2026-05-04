"""
sentinelai.models.fusion
-------------------------
Multi-modal late-fusion head.

Responsibilities
----------------
* Accept embeddings from FaceModel, PostureModel, and EnvModel.
* Learn cross-modal interactions via a Transformer-based or MLP fusion block.
* Produce a single unified representation used for downstream tasks:
    - Threat / anomaly detection (binary or multi-class)
    - Activity recognition
    - Attention / alert scoring

Fusion strategies implemented (choose via ``fusion_strategy``)
--------------------------------------------------------------
``"concat_mlp"``   — Concatenate all modal embeddings → MLP (default, fast baseline).
``"attention"``    — Cross-modal Transformer attention (richer, slower).

TODO
----
* Add gating mechanism (dynamic modality weighting) for missing-modal robustness.
* Benchmark concat_mlp vs. attention on a held-out validation set.
* Add contrastive alignment pre-training option between modalities.
* Write unit tests in tests/test_fusion.py.
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

FUSION_STRATEGIES = ("concat_mlp", "attention")
DEFAULT_NUM_CLASSES = 2   # e.g. normal vs. anomalous


# ---------------------------------------------------------------------------
# Helper: concat + MLP fusion
# ---------------------------------------------------------------------------

class ConcatMLP(nn.Module):
    """
    Concatenates modal embeddings along the feature dim, then applies a deep MLP.
    """

    def __init__(self, in_features: int, out_features: int, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_features),
        )

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        x = torch.cat(embeddings, dim=-1)  # (B, sum_of_dims)
        return self.net(x)


# ---------------------------------------------------------------------------
# Helper: cross-modal Transformer fusion
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Treats each modal embedding as a token and applies multi-head self-attention
    to learn cross-modal interactions before pooling.

    Expects all embeddings to share the same dimension (``d_model``).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_features: int = 128,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_features)

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        # Stack modalities as a sequence: (B, num_modalities, d_model)
        tokens = torch.stack(embeddings, dim=1)
        attended = self.transformer(tokens)         # (B, num_modalities, d_model)
        pooled = attended.mean(dim=1)               # (B, d_model) — mean over modalities
        return self.out_proj(pooled)                # (B, out_features)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """
    Multi-modal late-fusion head for SentinelAI.

    Parameters
    ----------
    face_dim : int
        Embedding dimension from FaceModel.
    posture_dim : int
        Embedding dimension from PostureModel.
    env_dim : int
        Embedding dimension from EnvModel.
    num_classes : int
        Number of output classes for the downstream task head.
    fusion_strategy : str
        One of ``"concat_mlp"`` or ``"attention"``.
    fusion_hidden_dim : int
        Internal width of the fusion MLP / Transformer projection.
    """

    def __init__(
        self,
        face_dim: int = 512,
        posture_dim: int = 256,
        env_dim: int = 256,
        num_classes: int = DEFAULT_NUM_CLASSES,
        fusion_strategy: str = "concat_mlp",
        fusion_hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        if fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion_strategy '{fusion_strategy}'. "
                f"Choose from: {FUSION_STRATEGIES}"
            )

        self.fusion_strategy = fusion_strategy
        total_in = face_dim + posture_dim + env_dim

        # ── Fusion block ─────────────────────────────────────────────────────
        if fusion_strategy == "concat_mlp":
            self.fusion_block = ConcatMLP(
                in_features=total_in,
                out_features=fusion_hidden_dim,
                hidden=fusion_hidden_dim,
            )
            task_in = fusion_hidden_dim

        elif fusion_strategy == "attention":
            # Project all modalities to a common dimension first
            common_dim = fusion_hidden_dim
            self.face_proj = nn.Linear(face_dim, common_dim)
            self.posture_proj = nn.Linear(posture_dim, common_dim)
            self.env_proj = nn.Linear(env_dim, common_dim)
            self.fusion_block = CrossModalAttention(
                d_model=common_dim,
                out_features=fusion_hidden_dim,
            )
            task_in = fusion_hidden_dim

        # ── Task head ────────────────────────────────────────────────────────
        # TODO: Swap for task-specific heads (e.g. regression for alert score).
        self.task_head = nn.Linear(task_in, num_classes)

        logger.info(
            "FusionModel initialised | strategy=%s | num_classes=%d",
            fusion_strategy,
            num_classes,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        face_emb: torch.Tensor,
        posture_emb: torch.Tensor,
        env_emb: torch.Tensor,
        return_fused: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        face_emb : torch.Tensor
            Face embedding from FaceModel, shape (B, face_dim).
        posture_emb : torch.Tensor
            Posture embedding from PostureModel, shape (B, posture_dim).
        env_emb : torch.Tensor
            Environment embedding from EnvModel, shape (B, env_dim).
        return_fused : bool
            If True, also return the intermediate fused representation.

        Returns
        -------
        dict with keys:
            ``"logits"``  — (B, num_classes) task logits.
            ``"fused"``   — (B, fusion_hidden_dim) intermediate rep (if requested).
        """
        if self.fusion_strategy == "concat_mlp":
            fused = self.fusion_block(face_emb, posture_emb, env_emb)

        elif self.fusion_strategy == "attention":
            # Project to common dim before attention
            f = self.face_proj(face_emb)
            p = self.posture_proj(posture_emb)
            e = self.env_proj(env_emb)
            fused = self.fusion_block(f, p, e)

        logits = self.task_head(fused)  # (B, num_classes)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if return_fused:
            output["fused"] = fused

        return output
