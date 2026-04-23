from enum import Enum
import os
from typing import Callable, Optional, Tuple, Union

import logging
import numpy as np
import pandas as pd
from PIL import Image

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")


class _Split(Enum):
    TRAIN = "train"
    # VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 772_602,
            _Split.TEST: 232_412,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, eid: int, zoom_level: int) -> str:
        dirname = self.get_dirname()
        basename = f"bev_image_eid-{eid:07d}_zoom-{zoom_level}"
        return os.path.join(dirname, basename + ".png")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[int, int]:
        dirname, filename = os.path.split(image_relpath)
        basename, _ = os.path.splitext(filename)
        parts = basename.split('_')
        eid = int(parts[2].split('-')[1])
        zoom_level = int(parts[3].split('-')[1])
        return eid, zoom_level


class UKBBEV(ExtendedVisionDataset):
    Target = Union[int]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "UKBBEV.Split",
        root: str,
        extra: str,
        label_file: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        if label_file is None:
            raise ValueError("label_file must be provided")
        self._label = pd.read_csv(label_file, delimiter="\t").loc[:, ["eid", "population_density_urban_or_rural"]]
        self._label.fillna(0)
        self._label.columns = ["eid", "label"]
        self._entries = None

    @property
    def split(self) -> "UKBBEV.Split":
        return self._split

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        eid = entries[index]["eid"]
        zoom_level = entries[index]["zoom_level"]

        image_relpath = self.split.get_image_relpath(eid, zoom_level)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()

        return image_data

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    def get_target(self, index: int) -> int:
        entries = self._get_entries()
        label = entries[index]['label']

        return label

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return np.array([0, 1])

    def get_target_for_entry(self, eid: int) -> int:
        label = self._label.loc[self._label.eid == eid]['label'].fillna(0).values[0]

        return int(label)

    def dump_extra(self) -> None:
        self._dump_entries()

    def _dump_entries(self) -> None:
        from tqdm.auto import tqdm
        split = self.split

        dataset_root = os.path.join(self.root, split.get_dirname())
        dataset = os.listdir(dataset_root)
        sample_count = len(dataset)

        dtype = np.dtype(
            [
                ("eid", "<u4"),
                ("zoom_level", "<u4"),
                ("label", "<u4"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        old_percent = -1
        for index in tqdm(range(sample_count)):
            percent = 100 * (index + 1) // sample_count
            if percent > old_percent:
                logger.info(f"creating entries: {percent}%")
                old_percent = percent

            image_full_path = dataset[index]
            image_relpath = os.path.relpath(image_full_path, self.root)
            eid, zoom_level = split.parse_image_relpath(image_relpath)
            label = self.get_target_for_entry(eid)
            entries_array[index] = (eid, zoom_level, label)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate entry arrays for UKBBEV dataset.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing train/test image splits.")
    parser.add_argument("--extra", type=str, required=True, help="Directory to save entry numpy files.")
    parser.add_argument("--label_file", type=str, required=True, help="Path to label TSV file.")
    args = parser.parse_args()

    for split in [UKBBEV.Split.TRAIN, UKBBEV.Split.TEST]:
        dataset = UKBBEV(split=split, root=args.root, extra=args.extra, label_file=args.label_file)
        dataset.dump_extra()
