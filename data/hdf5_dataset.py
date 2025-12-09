import h5py
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Dict
import random


class HDF5ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        hdf5_dir,
        caption_dir=None,
        datasets=["s2looking", "whu_cd", "xbd"],
        split="train",
        transform=None,
    ):
        self.hdf5_dir = str(hdf5_dir)
        self.caption_dir = str(caption_dir) if caption_dir else None
        self.transform = transform

        self.file_paths = []
        self.file_sizes = []
        self.file_names = []
        self.samples = []
        self.caption_data = {}

        hdf5_dir_path = Path(hdf5_dir)

        for dataset_name in datasets:
            hdf5_path = hdf5_dir_path / f"{dataset_name}_{split}.h5"

            if not hdf5_path.exists():
                continue

            with h5py.File(str(hdf5_path), "r") as fh:
                num_samples = fh.attrs.get("num_samples", len(fh["images_a"]))

            # Loaing captions 
            captions = self._load_captions(dataset_name) if self.caption_dir else []
            if captions:
                self.caption_data[dataset_name] = captions

            file_idx = len(self.file_paths)
            self.file_paths.append(str(hdf5_path))
            self.file_sizes.append(num_samples)
            self.file_names.append(dataset_name)

            for local_idx in range(num_samples):
                self.samples.append((file_idx, local_idx, dataset_name))

    def _load_captions(self, dataset_name: str):
        if not self.caption_dir:
            return []

        caption_dir_path = Path(self.caption_dir)
        captions = []

        if dataset_name == "s2looking":
            caption_files = [
                caption_dir_path / "s2looking_train_captions.json",
                caption_dir_path / "s2looking_val_captions.json",
                caption_dir_path / "s2looking_test_captions.json",
            ]
        else:
            caption_files = [caption_dir_path / f"{dataset_name}_all_captions.json"]

        for caption_file in caption_files:
            if caption_file.exists():
                with open(caption_file, "r") as f:
                    data = json.load(f)
                captions.extend(data)

        return captions

    def _get_caption(self, dataset_name, local_idx):
        # Random selection if multiple captions available
        caption_list = self.caption_data.get(dataset_name, [])
        if local_idx < len(caption_list):
            entry = caption_list[local_idx]
            sentences = entry.get("sentences", entry.get("captions", []))

            if isinstance(sentences, list) and sentences:
                return random.choice(sentences)
            return str(sentences)

        return "Changes detected in temporal images"

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_file_handles" in state:
            del state["_file_handles"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_file_handle(self, file_idx):
        if not hasattr(self, "_file_handles"):
            self._file_handles = {}

        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = h5py.File(self.file_paths[file_idx], "r")

        return self._file_handles[file_idx]

    def __getitem__(self, idx):
        file_idx, local_idx, dataset_name = self.samples[idx]
        fh = self._get_file_handle(file_idx)

        img_a_np = fh["images_a"][local_idx]
        img_b_np = fh["images_b"][local_idx]

        img_a = Image.fromarray(img_a_np, mode="RGB")
        img_b = Image.fromarray(img_b_np, mode="RGB")

        # Resize to 224x224 for CLIP
        target_size = (224, 224)
        if img_a.size != target_size:
            img_a = img_a.resize(target_size, Image.LANCZOS)
        if img_b.size != target_size:
            img_b = img_b.resize(target_size, Image.LANCZOS)

        if self.transform is not None:
            img_a, img_b = self.transform(img_a, img_b)

        caption = self._get_caption(dataset_name, local_idx)

        return {
            "image_a": img_a,
            "image_b": img_b,
            "caption": caption,
            "dataset": dataset_name,
        }

    def __del__(self):
        if hasattr(self, "_file_handles"):
            for fh in self._file_handles.values():
                fh.close()
