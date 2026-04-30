from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "datasets"

RAW = DATASETS / "raw"
PROCESSED = DATASETS / "processed"
SPLITS = DATASETS / "splits"


def create_directories():
    folders = [
        RAW,
        PROCESSED,
        SPLITS,
        RAW / "fer2013",
        RAW / "coco_keypoints",
        RAW / "places365",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")


def dataset_instructions():
    print("\n=== SENTINELAI DATASET SETUP ===\n")

    print("1. FER2013")
    print("Download manually from:")
    print("https://www.kaggle.com/datasets/msambare/fer2013")
    print(f"Place files inside: {RAW / 'fer2013'}\n")

    print("2. COCO Keypoints")
    print("Download annotations from:")
    print("https://cocodataset.org/#download")
    print(f"Place files inside: {RAW / 'coco_keypoints'}\n")

    print("3. Places365")
    print("Download subset from:")
    print("http://places2.csail.mit.edu/")
    print(f"Place files inside: {RAW / 'places365'}\n")

    print("Optional (skip for now):")
    print("- AffectNet")
    print("- MPII Human Pose\n")


if __name__ == "__main__":
    create_directories()
    dataset_instructions()