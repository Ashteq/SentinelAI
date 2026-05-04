"""
tests/test_models.py
--------------------
Smoke tests verifying that all model modules instantiate correctly
and produce outputs of the expected shape.

Run with:  pytest tests/ -v
"""

import pytest
import torch

from sentinelai.models import FaceModel, PostureModel, EnvModel, FusionModel


BATCH = 2
IMG_C, IMG_H, IMG_W = 3, 224, 224
NUM_KEYPOINTS = 17


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def face_model():
    return FaceModel(backbone_name="mobilenet_v3_small", pretrained=False)


@pytest.fixture(scope="module")
def posture_model():
    return PostureModel(num_keypoints=NUM_KEYPOINTS, embedding_dim=256)


@pytest.fixture(scope="module")
def env_model():
    return EnvModel(embedding_dim=256, pretrained=False)


@pytest.fixture(scope="module")
def fusion_model():
    return FusionModel(
        face_dim=576,   # mobilenet_v3_small default after projection (matches fixture above)
        posture_dim=256,
        env_dim=256,
        num_classes=2,
        fusion_strategy="concat_mlp",
    )


# ---------------------------------------------------------------------------
# FaceModel
# ---------------------------------------------------------------------------

class TestFaceModel:
    def test_output_shape(self, face_model):
        x = torch.randn(BATCH, IMG_C, IMG_H, IMG_W)
        out = face_model(x)
        assert "embedding" in out
        assert out["embedding"].shape == (BATCH, face_model.embedding_dim)

    def test_emotion_head(self):
        m = FaceModel(backbone_name="mobilenet_v3_small", pretrained=False, num_emotion_classes=7)
        x = torch.randn(BATCH, IMG_C, IMG_H, IMG_W)
        out = m(x)
        assert "emotion_logits" in out
        assert out["emotion_logits"].shape == (BATCH, 7)

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError):
            FaceModel(backbone_name="unknown_backbone")


# ---------------------------------------------------------------------------
# PostureModel
# ---------------------------------------------------------------------------

class TestPostureModel:
    def test_output_shape(self, posture_model):
        kp = torch.randn(BATCH, NUM_KEYPOINTS, 3)
        out = posture_model(kp)
        assert "embedding" in out
        assert out["embedding"].shape == (BATCH, 256)

    def test_posture_head(self):
        m = PostureModel(num_posture_classes=6)
        kp = torch.randn(BATCH, NUM_KEYPOINTS, 3)
        out = m(kp)
        assert "posture_logits" in out
        assert out["posture_logits"].shape == (BATCH, 6)


# ---------------------------------------------------------------------------
# EnvModel
# ---------------------------------------------------------------------------

class TestEnvModel:
    def test_output_shape(self, env_model):
        x = torch.randn(BATCH, IMG_C, IMG_H, IMG_W)
        out = env_model(x)
        assert "embedding" in out
        assert out["embedding"].shape == (BATCH, 256)

    def test_scene_head(self):
        m = EnvModel(pretrained=False, num_scene_classes=7)
        x = torch.randn(BATCH, IMG_C, IMG_H, IMG_W)
        out = m(x)
        assert "scene_logits" in out
        assert out["scene_logits"].shape == (BATCH, 7)


# ---------------------------------------------------------------------------
# FusionModel
# ---------------------------------------------------------------------------

class TestFusionModel:
    def test_concat_mlp_output_shape(self, fusion_model):
        face_emb = torch.randn(BATCH, 576)
        posture_emb = torch.randn(BATCH, 256)
        env_emb = torch.randn(BATCH, 256)
        out = fusion_model(face_emb, posture_emb, env_emb)
        assert "logits" in out
        assert out["logits"].shape == (BATCH, 2)

    def test_attention_strategy(self):
        dim = 256
        m = FusionModel(
            face_dim=dim, posture_dim=dim, env_dim=dim,
            fusion_strategy="attention", num_classes=3,
        )
        f = torch.randn(BATCH, dim)
        p = torch.randn(BATCH, dim)
        e = torch.randn(BATCH, dim)
        out = m(f, p, e, return_fused=True)
        assert "logits" in out
        assert "fused" in out
        assert out["logits"].shape == (BATCH, 3)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            FusionModel(fusion_strategy="bad_strategy")
