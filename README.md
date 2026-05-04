# 🛡️ SentinelAI

> **Modular multi-modal computer vision framework built on PyTorch.**  
> Combines face analysis, body posture estimation, and environmental scene understanding
> into a unified late-fusion pipeline for intelligent situational awareness.

---

## ✨ Features

| Modality | Module | Description |
|---|---|---|
| 👤 Face | `face_model.py` | CNN backbone (ResNet / MobileNet / EfficientNet) → face embedding + optional emotion head |
| 🧍 Posture | `posture_model.py` | Skeleton keypoint encoder (MLP / GNN-ready) → posture embedding + class head |
| 🏙️ Environment | `env_model.py` | Scene backbone → env embedding + optional segmentation decoder |
| 🔗 Fusion | `fusion.py` | Late fusion: `concat_mlp` (fast) or cross-modal `attention` (rich) |

---

## 📁 Project Structure

```
SentinelAI/
├── sentinelai/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── face_model.py      ← Face encoder
│   │   ├── posture_model.py   ← Posture encoder
│   │   ├── env_model.py       ← Environment encoder
│   │   └── fusion.py          ← Multi-modal fusion head
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py         ← SentinelDataset (stub)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py          ← Logging setup
│   │   └── checkpoint.py      ← Save / load checkpoints (stub)
│   └── configs/
│       └── config.yaml        ← Central YAML config (Hydra-compatible)
├── tests/
│   └── test_models.py         ← Pytest smoke tests
├── scripts/
│   └── train.py               ← Training entry point (stub)
├── docs/
├── notebooks/
├── config.yaml                ← Root-level config alias
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/SentinelAI.git
cd SentinelAI

# Create virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate                            # Windows

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 2. Verify installation (smoke tests)

```bash
pytest tests/ -v
```

### 3. Configure

Edit `config.yaml` to set your backbone choices, embedding dims, fusion strategy, training hyper-params, and data paths.

### 4. Train *(once training loop is implemented)*

```bash
python scripts/train.py
# Override any config key via Hydra CLI:
python scripts/train.py training.batch_size=64 models.fusion.strategy=attention
```

---

## 🔧 Configuration

All knobs live in `config.yaml` (Hydra-compatible).  
Key sections:

```yaml
models:
  face:
    backbone: "resnet50"    # resnet50 | mobilenet_v3_small | efficientnet_b0
    embedding_dim: 512

  fusion:
    strategy: "concat_mlp"  # concat_mlp | attention
    num_classes: 2

training:
  epochs: 50
  batch_size: 32
  optimizer: { name: "adamw", lr: 1.0e-4 }
```

---

## 🧩 Module API Reference

### FaceModel

```python
from sentinelai.models import FaceModel

model = FaceModel(backbone_name="resnet50", embedding_dim=512, pretrained=True)
out = model(image_batch)   # image_batch: (B, 3, 224, 224)
# out["embedding"]       → (B, 512)
# out["emotion_logits"]  → (B, N)  if num_emotion_classes was set
```

### PostureModel

```python
from sentinelai.models import PostureModel

model = PostureModel(num_keypoints=17, embedding_dim=256)
out = model(keypoints)     # keypoints: (B, 17, 3) — (x, y, confidence)
# out["embedding"]        → (B, 256)
# out["posture_logits"]   → (B, N)  if num_posture_classes was set
```

### EnvModel

```python
from sentinelai.models import EnvModel

model = EnvModel(embedding_dim=256, pretrained=True)
out = model(image_batch)   # image_batch: (B, 3, 224, 224)
# out["embedding"]      → (B, 256)
# out["scene_logits"]   → (B, N)  if num_scene_classes was set
# out["seg_map"]        → (B, C, H', W')  if num_seg_classes was set
```

### FusionModel

```python
from sentinelai.models import FusionModel

model = FusionModel(face_dim=512, posture_dim=256, env_dim=256,
                    fusion_strategy="attention", num_classes=2)
out = model(face_emb, posture_emb, env_emb, return_fused=True)
# out["logits"]  → (B, 2)
# out["fused"]   → (B, fusion_hidden_dim)
```

---

## 🗺️ Roadmap

- [ ] Implement `SentinelDataset` with multi-modal loading
- [ ] Integrate face detector (RetinaFace / MTCNN) into `FaceModel`
- [ ] Add body keypoint detector (MediaPipe / MMPose) into `PostureModel`
- [ ] Implement full training loop in `scripts/train.py`
- [ ] Add GNN-based joint encoder in `PostureModel`
- [ ] Add depth-estimation head in `EnvModel`
- [ ] Benchmark `concat_mlp` vs. `attention` fusion
- [ ] Add TorchScript / ONNX export for deployment
- [ ] Docker image for reproducible training

---

## 🤝 Contributing

1. Fork the repo and create a feature branch (`git checkout -b feat/my-feature`).
2. Install pre-commit hooks: `pre-commit install`.
3. Make changes, add tests under `tests/`, ensure `pytest` passes.
4. Submit a pull request.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
