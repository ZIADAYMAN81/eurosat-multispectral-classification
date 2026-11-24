"""Data loading for the EuroSAT "all bands" release (13 Sentinel-2 bands, 64 x 64 pixels).

Reading thousands of GeoTIFFs every epoch is slow, so each split is decoded once into a compact
``uint16`` NumPy cache; later epochs read from memory-mapped arrays instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

N_BANDS, SIZE = 13, 64
REFLECTANCE_SCALE = 10_000.0  # Sentinel-2 L1C digital numbers -> approximate reflectance
SPLITS = ("train", "validation", "test")


def load_label_map(root: Path) -> Dict[str, int]:
    with open(Path(root) / "label_map.json", encoding="utf-8") as f:
        return json.load(f)


def build_cache(root: Path, split: str, cache_dir: Path) -> Tuple[Path, Path]:
    """Decode one split's GeoTIFFs into ``<split>_x.npy`` (uint16) and ``<split>_y.npy`` (int64)."""
    import rasterio  # only needed the first time

    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path = cache_dir / f"{split}_x.npy", cache_dir / f"{split}_y.npy"
    if x_path.exists() and y_path.exists():
        return x_path, y_path

    df = pd.read_csv(Path(root) / f"{split}.csv")
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    x = np.lib.format.open_memmap(x_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint16,
                                  shape=(len(df), N_BANDS, SIZE, SIZE))
    for i, rel in enumerate(df["Filename"]):
        with rasterio.open(Path(root) / rel) as src:
            x[i] = src.read()
    x.flush()
    del x
    x_path.with_suffix(".tmp.npy").replace(x_path)
    np.save(y_path, df["Label"].to_numpy(np.int64))
    return x_path, y_path


def random_flip_rotate(x: torch.Tensor) -> torch.Tensor:
    """Satellite tiles have no 'up', so flips and 90-degree rotations are label-preserving."""
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[2])
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[1])
    k = int(torch.randint(0, 4, (1,)))
    return torch.rot90(x, k, dims=[1, 2]) if k else x


class EuroSATAllBands(Dataset):
    def __init__(self, x_path: Path, y_path: Path, transform: Optional[Callable] = None):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx].astype(np.float32) / REFLECTANCE_SCALE)
        if self.transform:
            x = self.transform(x)
        return x, int(self.y[idx])


def get_loaders(root: str, cache_dir: str, batch_size: int = 64, augment: bool = True):
    root_p, cache_p = Path(root), Path(cache_dir)
    datasets = {}
    for split in SPLITS:
        xp, yp = build_cache(root_p, split, cache_p)
        datasets[split] = EuroSATAllBands(xp, yp, random_flip_rotate if (augment and split == "train") else None)
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"), num_workers=0)
               for s, d in datasets.items()}
    return loaders, load_label_map(root_p)
