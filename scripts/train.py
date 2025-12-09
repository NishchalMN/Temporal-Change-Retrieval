import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from data.hdf5_dataset import HDF5ChangeDetectionDataset
from models.clip_encoder import BitemporalCLIPEncoder


class InfoNCELoss(nn.Module):
    # Contrastive loss to match change embeddings and text descriptions

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # Compute similarity matrix
        logits = (image_features @ text_features.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        # Bidirectional loss of image to text and text to image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


class HDF5Transform:
    def __init__(self, clip_transform):
        self.clip_transform = clip_transform

    def __call__(self, img_a, img_b):
        return self.clip_transform(img_a), self.clip_transform(img_b)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        img_a = batch["image_a"].to(device)
        img_b = batch["image_b"].to(device)
        captions = batch["caption"]

        optimizer.zero_grad()

        change_emb, text_emb = model(img_a, img_b, captions)
        loss = criterion(change_emb, text_emb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            img_a = batch["image_a"].to(device)
            img_b = batch["image_b"].to(device)
            captions = batch["caption"]

            change_emb, text_emb = model(img_a, img_b, captions)
            loss = criterion(change_emb, text_emb)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Temporal change retrieval model")
    parser.add_argument("--hdf5-dir", type=str)
    parser.add_argument("--caption-dir", type=str)
    parser.add_argument(
        "--strategy",
        type=str,
        default="concat",
        choices=["difference", "concat", "learned", "cross_attn", "fst"],
    )
    parser.add_argument("--use-remote-clip", action="store_true")
    parser.add_argument("--freeze-early-layers", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="checkpoints")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clip_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    train_dataset = HDF5ChangeDetectionDataset(
        hdf5_dir=args.hdf5_dir,
        caption_dir=args.caption_dir,
        datasets=["s2looking", "whu_cd", "xbd"],
        split="train",
        transform=HDF5Transform(clip_transform),
    )

    val_dataset = HDF5ChangeDetectionDataset(
        hdf5_dir=args.hdf5_dir,
        caption_dir=args.caption_dir,
        datasets=["s2looking", "whu_cd", "xbd"],
        split="val",
        transform=HDF5Transform(clip_transform),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = BitemporalCLIPEncoder(
        strategy=args.strategy,
        freeze_clip=False,
        use_remote_clip=args.use_remote_clip,
        device=args.device,
    )

    # Keeps last 4 layers trainable for fine-tuning if freeze_early_layers is true
    if args.freeze_early_layers:
        for name, param in model.clip_model.named_parameters():
            if "visual.transformer.resblocks" in name:
                layer_num = int(name.split(".")[3])
                if layer_num < 8:
                    param.requires_grad = False

    criterion = InfoNCELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = validate(model, val_loader, criterion, args.device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                args.output_dir, f"clip_{args.strategy}_best.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "strategy": args.strategy,
                },
                checkpoint_path,
            )
            print(f"Saved best model: {checkpoint_path}")


if __name__ == "__main__":
    main()
