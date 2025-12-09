import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import Counter


class ImagePair:
    def __init__(
        self,
        image_a_path,
        image_b_path,
        label_path,
        image_id,
        split,
        dataset,
        has_change=None,
    ):
        self.image_a_path = image_a_path
        self.image_b_path = image_b_path
        self.label_path = label_path
        self.image_id = image_id
        self.split = split
        self.dataset = dataset
        self.has_change = has_change


# Data adapters
class S2LookingAdapter:
    def __init__(self, root_dir):
        self.root = Path(root_dir)

    def get_image_pairs(self, split):
        image1_dir = self.root / split / "Image1"
        image2_dir = self.root / split / "Image2"
        label_dir = self.root / split / "label"

        if not image1_dir.exists():
            return []

        pairs = []
        for img1 in sorted(image1_dir.glob("*.png")):
            img_id = img1.stem
            img2_path = image2_dir / f"{img_id}.png"
            label_path = label_dir / f"{img_id}.png"

            if img2_path.exists():
                has_change = self._check_change(label_path)
                pairs.append(
                    ImagePair(
                        str(img1),
                        str(img2_path),
                        str(label_path) if label_path.exists() else None,
                        f"s2looking_{split}_{img_id}",
                        split,
                        "s2looking",
                        has_change,
                    )
                )
        return pairs

    def load_image_pair(self, pair):
        img_a = Image.open(pair.image_a_path).convert("RGB")
        img_b = Image.open(pair.image_b_path).convert("RGB")
        return img_a, img_b

    def _check_change(self, label_path):
        if not label_path or not Path(label_path).exists():
            return None
        try:
            label = Image.open(label_path)
            return np.any(np.array(label) > 0)
        except:
            return None


class XBDAdapter:
    def __init__(self, root_dir):
        self.root = Path(root_dir) / "geotiffs"

    def get_image_pairs(self, split="all"):
        if not self.root.exists():
            return []

        pairs = []
        for split_dir in sorted(self.root.iterdir()):
            if not split_dir.is_dir():
                continue

            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            if not images_dir.exists():
                continue

            pre_images = {
                p.stem.replace("_pre_disaster", ""): p
                for p in images_dir.glob("*_pre_disaster.tif")
            }
            post_images = {
                p.stem.replace("_post_disaster", ""): p
                for p in images_dir.glob("*_post_disaster.tif")
            }

            for img_id in sorted(pre_images.keys() & post_images.keys()):
                label_path = (
                    labels_dir / f"{img_id}_pre_disaster.json"
                    if labels_dir.exists()
                    else None
                )
                pairs.append(
                    ImagePair(
                        str(pre_images[img_id]),
                        str(post_images[img_id]),
                        str(label_path) if label_path and label_path.exists() else None,
                        f"xbd_{split_dir.name}_{img_id}",
                        split,
                        "xbd",
                        True,
                    )
                )

        return pairs

    def load_image_pair(self, pair):
        img_a = self._load_geotiff(pair.image_a_path)
        img_b = self._load_geotiff(pair.image_b_path)
        return img_a, img_b

    def _load_geotiff(self, tif_path):
        try:
            import rasterio

            with rasterio.open(tif_path) as src:
                if src.count >= 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                    rgb = np.dstack([r, g, b])
                    rgb = (
                        (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255
                    ).astype(np.uint8)
                    return Image.fromarray(rgb)
        except:
            pass
        return Image.open(tif_path).convert("RGB")


# Format conversion utilities
def convert_to_instruction_format(
    captions_file, output_file, image_output_dir, target_size=(252, 252)
):
    print(f"Converting {captions_file} to instruction format...")

    with open(captions_file) as f:
        data = json.load(f)

    instruction_data = []
    image_dir = Path(image_output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(data, desc="Converting"):
        dataset = item.get("dataset", "unknown")
        split = item.get("split", "train")
        img_id = item.get("image_id", "unknown")
        original_paths = item.get("original_paths", {})

        # Organize images
        try:
            dataset_dir = image_dir / dataset / split
            a_dir = dataset_dir / "A"
            b_dir = dataset_dir / "B"
            a_dir.mkdir(parents=True, exist_ok=True)
            b_dir.mkdir(parents=True, exist_ok=True)

            # Load and resize images
            img_a = Image.open(original_paths.get("image_a")).convert("RGB")
            img_b = Image.open(original_paths.get("image_b")).convert("RGB")
            img_a = img_a.resize(target_size, Image.LANCZOS)
            img_b = img_b.resize(target_size, Image.LANCZOS)

            filename = f"{img_id}.png"
            img_a.save(a_dir / filename)
            img_b.save(b_dir / filename)

            rel_path_a = f"{dataset}/{split}/A/{filename}"
            rel_path_b = f"{dataset}/{split}/B/{filename}"

            # Create instruction samples
            for caption in item.get("captions", []):
                instruction_data.append(
                    {
                        "id": len(instruction_data),
                        "image": [rel_path_a, rel_path_b],
                        "changeflag": 1 if item.get("has_change") else 0,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image> <image> Please briefly describe the changes in these two images.",
                            },
                            {"from": "gpt", "value": caption},
                        ],
                    }
                )
        except Exception as e:
            print(f"\nError processing {img_id}: {e}")
            continue

    with open(output_file, "w") as f:
        json.dump(instruction_data, f, indent=2)

    print(f"Saved {len(instruction_data)} instruction samples to {output_file}")
    return output_file


def merge_datasets(instruction_files, output_file, shuffle=True):
    print("\nMerging instruction files...")

    all_data = []
    for file in instruction_files:
        if not Path(file).exists():
            continue
        with open(file) as f:
            data = json.load(f)
            print(f"  {Path(file).name}: {len(data)} samples")
            all_data.extend(data)

    print(f"\nTotal samples: {len(all_data)}")

    if shuffle:
        random.shuffle(all_data)

    for idx, item in enumerate(all_data):
        item["id"] = idx

    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Saved merged dataset to: {output_file}")

    changed = sum(1 for x in all_data if x.get("changeflag", 1) == 1)
    unchanged = len(all_data) - changed
    print(f"\nStats: {changed} changed, {unchanged} unchanged")

    return output_file
