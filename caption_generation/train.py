import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from PIL import Image
import clip
import os


class CaptionDataset(Dataset):
    def __init__(self, json_file, image_dir, max_samples=None):
        with open(json_file) as f:
            self.data = json.load(f)

        if max_samples:
            self.data = self.data[:max_samples]

        self.image_dir = Path(image_dir)

        _, self.preprocess = clip.load("ViT-B/32", device="cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_a_path = self.image_dir / item["image"][0]
        img_b_path = self.image_dir / item["image"][1]

        img_a = self.preprocess(Image.open(img_a_path).convert("RGB"))
        img_b = self.preprocess(Image.open(img_b_path).convert("RGB"))

        caption = ""
        for conv in item.get("conversations", []):
            if conv["from"] == "gpt":
                caption = conv["value"]
                break

        return img_a, img_b, caption


class SimpleChangeModel(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # Load CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device=device)

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        clip_dim = 512
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim), nn.ReLU(), nn.Linear(clip_dim, clip_dim)
        )

        self.device = device

    def forward(self, img_a, img_b, texts):
        with torch.no_grad():
            feat_a = self.clip_model.encode_image(img_a)
            feat_b = self.clip_model.encode_image(img_b)
            text_feats = self.clip_model.encode_text(
                clip.tokenize(texts, truncate=True).to(self.device)
            )

        change_feat = torch.cat([feat_a, feat_b], dim=1)
        change_emb = self.fusion(change_feat.float())

        return change_emb, text_feats.float()


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_emb, text_emb):
        # Normalize
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        logits = (image_emb @ text_emb.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


def train(args):
    # Load dataset
    train_dataset = CaptionDataset(
        args.data, args.image_dir, max_samples=args.max_samples
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    model = SimpleChangeModel(device=args.device).to(args.device)

    criterion = ContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.05,
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = 0
        train_acc = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img_a, img_b, captions in pbar:
            img_a = img_a.to(args.device)
            img_b = img_b.to(args.device)

            change_emb, text_emb = model(img_a, img_b, captions)
            loss = criterion(change_emb, text_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                logits = (
                    F.normalize(change_emb, dim=-1) @ F.normalize(text_emb, dim=-1).T
                ) / args.temperature
                preds = logits.argmax(dim=1)
                labels = torch.arange(len(logits), device=args.device)
                acc = (preds == labels).float().mean()

            train_loss += loss.item()
            train_acc += acc.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.3f}"})

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for img_a, img_b, captions in tqdm(val_loader, desc="Validating"):
                img_a = img_a.to(args.device)
                img_b = img_b.to(args.device)

                change_emb, text_emb = model(img_a, img_b, captions)
                loss = criterion(change_emb, text_emb)

                logits = (
                    F.normalize(change_emb, dim=-1) @ F.normalize(text_emb, dim=-1).T
                ) / args.temperature
                preds = logits.argmax(dim=1)
                labels = torch.arange(len(logits), device=args.device)
                acc = (preds == labels).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.output_dir) / f"model_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                save_path,
            )
            print(f"  ✓ Saved best model: {save_path}")

    print(f"\nTraining finished. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to instruction JSON file")
    parser.add_argument("--image-dir", required=True, help="Path to image directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit samples for testing"
    )

    args = parser.parse_args()
    train(args)
