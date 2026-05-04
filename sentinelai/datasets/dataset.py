"""
sentinelai.data.dataset
------------------------
Multi-modal dataset definition for SentinelAI.

TODO
----
* Implement ``__getitem__`` to load image + keypoint + label triplets.
* Support lazy loading and memory-mapped datasets for large-scale use.
* Add caching layer (lmdb / hdf5) for pre-processed features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset


class SentinelDataset(Dataset):
    """
    Loads multi-modal samples for SentinelAI training/evaluation.

    Each sample contains:
    * A scene image      → fed into FaceModel + EnvModel after cropping.
    * Body keypoints     → fed into PostureModel.
    * A label            → supervision signal for FusionModel.

    Parameters
    ----------
    root : str | Path
        Path to the dataset root directory.
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    transform : callable, optional
        Image transform applied to the raw scene image.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # TODO: Parse annotation file / directory structure here
        self._samples: list = []  # list of (image_path, keypoints_path, label)
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Populate self._samples from disk.  TODO: implement."""
        raise NotImplementedError("_load_annotations() not yet implemented.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        dict with keys:
            ``"image"``     — (C, H, W) tensor.
            ``"keypoints"`` — (K, 3) tensor.
            ``"label"``     — int scalar.
        """
        # TODO: Implement loading, decoding, and transform application
        raise NotImplementedError("__getitem__() not yet implemented.")
