import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from data.hdf5_dataset import HDF5ChangeDetectionDataset
from models.clip_encoder import BitemporalCLIPEncoder


class HDF5Transform:
    def __init__(self, clip_transform):
        self.clip_transform = clip_transform

    def __call__(self, img_a, img_b):
        return self.clip_transform(img_a), self.clip_transform(img_b)


def compute_recall_at_k(similarities, k_values=[1, 5, 10]):
    # Compute recall@K metrics for retrieval evaluation
    results = {}
    num_queries = similarities.shape[0]

    for k in k_values:
        # Get top-k predictions for each query
        top_k = np.argsort(-similarities, axis=1)[:, :k]
        # Check if correct match is in top-k
        correct = np.any(top_k == np.arange(num_queries)[:, None], axis=1)
        recall = correct.mean()
        results[f"Recall@{k}"] = recall * 100

    return results


def evaluate_model(checkpoint_path, hdf5_dir, caption_dir, strategy, device="cuda"):

    clip_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    test_dataset = HDF5ChangeDetectionDataset(
        hdf5_dir=hdf5_dir,
        caption_dir=caption_dir,
        datasets=["s2looking", "whu_cd", "xbd"],
        split="val",
        transform=HDF5Transform(clip_transform),
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = BitemporalCLIPEncoder(strategy=strategy, freeze_clip=False, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            img_a = batch["image_a"].to(device)
            img_b = batch["image_b"].to(device)
            captions = batch["caption"]

            change_emb, text_emb = model(img_a, img_b, captions)

            all_image_features.append(change_emb.cpu().numpy())
            all_text_features.append(text_emb.cpu().numpy())

    image_features = np.concatenate(all_image_features, axis=0)
    text_features = np.concatenate(all_text_features, axis=0)

    similarities = image_features @ text_features.T

    results = compute_recall_at_k(similarities, k_values=[1, 5, 10])

    print("\nRetrieval Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}%")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate temporal change retrieval model"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hdf5-dir", type=str, required=True)
    parser.add_argument("--caption-dir", type=str, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["difference", "concat", "learned", "cross_attn", "fst"],
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        hdf5_dir=args.hdf5_dir,
        caption_dir=args.caption_dir,
        strategy=args.strategy,
        device=args.device,
    )


if __name__ == "__main__":
    main()
