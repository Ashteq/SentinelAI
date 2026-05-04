from __future__ import annotations

import os
import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from sentinelai.models.face_model import FaceModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            logits = outputs["emotion_logits"]

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(loader)

    return avg_loss, acc


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs=1, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("weights", exist_ok=True)

    best_acc = 0.0
    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            logits = outputs["emotion_logits"]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        log.info(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/face_best.pth")
            log.info("Saved best model -> weights/face_best.pth")


# ============================================================
# Main
# ============================================================

@hydra.main(version_base=None, config_path="../sentinelai/configs", config_name="config")
def main(cfg: DictConfig):
    data_dir = Path("sentinelai/datasets/raw/fer2013")

    if not data_dir.exists():
        raise FileNotFoundError(f"FER2013 folder not found: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    full_dataset = ImageFolder(data_dir / "train", transform=transform)

    # Use only 5000 images for quick training
    subset_size = 5000
    dataset, _ = random_split(
        full_dataset,
        [subset_size, len(full_dataset) - subset_size]
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    model = FaceModel(
        backbone_name="mobilenet_v3_small",
        pretrained=False,
        num_emotion_classes=7
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        lr=1e-4
    )


if __name__ == "__main__":
    main()