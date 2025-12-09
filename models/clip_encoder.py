import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from pathlib import Path
import numpy as np
from models.fst_module import FrequencySpatialTemporalModule

def load_remoteclip(model_name, device):
    # Load RemoteCLIP weights from local checkpoint.
    manual_path = Path("models/checkpoints") / f"RemoteCLIP-{model_name}.pt"
    if manual_path.exists():
        ckpt = torch.load(manual_path, map_location=device, weights_only=False)
        return ckpt
    return None


class BitemporalCLIPEncoder(nn.Module):
    """
    CLIP-based encoder 

    strategies implemented:
        - difference: Simple feature difference
        - concat: Concatenation of before, after, and difference
        - learned: MLP-based change encoder
        - cross_attn: Cross-attention between before and after 
        - fst: Frequency-Spatial-Temporal fusion
    """

    def __init__(
        self,
        model_name='ViT-B-32',
        pretrained='laion2b_s34b_b79k',
        strategy='difference',
        freeze_clip=False,
        use_remote_clip=False,
        device='cuda'
    ):
        super().__init__()

        self.strategy = strategy
        self.device = device

        # Load CLIP model
        if use_remote_clip:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
            self.tokenizer = open_clip.get_tokenizer(model_name)

            ckpt = load_remoteclip(model_name, device)
            if ckpt is not None:
                self.clip_model.load_state_dict(ckpt)

            self.clip_model = self.clip_model.to(device)
        else:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

        # Freeze CLIP backbone based on flag
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.embed_dim = self.clip_model.visual.output_dim

        if strategy == 'concat':
            # Projecting concatenated features to embedding
            self.projection = nn.Sequential(
                nn.Linear(self.embed_dim * 3, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            ).to(device)

        elif strategy == 'learned':
            # Learnable function
            self.change_mlp = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            ).to(device)

        elif strategy == 'cross_attn':
            # Cross-attention
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=8,
                batch_first=True
            ).to(device)

        elif strategy == 'fst':
            # Frequency-Spatial-Temporal fusion
            transformer_width = self.clip_model.visual.conv1.out_channels
            self.fst_module = FrequencySpatialTemporalModule(
                channels=transformer_width,
                output_dim=transformer_width
            ).to(device)

    def encode_image(self, image):
        # Encode image to CLIP embedding
        features = self.clip_model.encode_image(image)
        return F.normalize(features, dim=-1)

    def encode_text(self, text):
        # Encode text to CLIP embedding
        if isinstance(text, str):
            text = [text]

        tokens = self.tokenizer(text).to(self.device)
        features = self.clip_model.encode_text(tokens)
        return F.normalize(features, dim=-1)

    def extract_patch_features(self, image):
        # Getting patch embeddings
        x = self.clip_model.visual.conv1(image)

        # Adding positional embeddings
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        # Adding class token
        class_token = self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, C, dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)

        # Adding positional embeddings
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)

        # Reshape to 2D
        patches_no_cls = x[:, 1:, :]
        num_spatial = patches_no_cls.shape[1]
        H_out = W_out = int(np.sqrt(num_spatial))
        patches_2d = patches_no_cls.reshape(B, H_out, W_out, C).permute(0, 3, 1, 2)

        return patches_2d

    def encode_change(self, img_before, img_after):
        # Encodes change between two images
        feat_before = self.encode_image(img_before)
        feat_after = self.encode_image(img_after)

        if self.strategy == 'difference':
            change = feat_after - feat_before
            change = F.normalize(change, dim=-1)

        elif self.strategy == 'concat':
            diff = feat_after - feat_before
            combined = torch.cat([feat_before, feat_after, diff], dim=-1)
            change = self.projection(combined)
            change = F.normalize(change, dim=-1)

        elif self.strategy == 'learned':
            combined = torch.cat([feat_before, feat_after], dim=-1)
            change = self.change_mlp(combined)
            change = F.normalize(change, dim=-1)

        elif self.strategy == 'cross_attn':
            feat_before_seq = feat_before.unsqueeze(1)
            feat_after_seq = feat_after.unsqueeze(1)

            attn_out, _ = self.cross_attention(
                feat_after_seq,
                feat_before_seq,
                feat_before_seq
            )

            change = attn_out.squeeze(1)
            change = F.normalize(change, dim=-1)

        elif self.strategy == 'fst':
            # Frequency-Spatial-Temporal fusion
            patches_2d_before = self.extract_patch_features(img_before)
            patches_2d_after = self.extract_patch_features(img_after)

            change, _ = self.fst_module(patches_2d_before, patches_2d_after)

            # Projecting back to CLIP embedding space
            change = self.clip_model.visual.ln_post(change)
            change = change @ self.clip_model.visual.proj
            change = F.normalize(change, dim=-1)

        return change

    def forward(self, img_before, img_after, text=None):
        """
        Input is before and after images, and text.
        Returns change embedding and text embeddings
        """
        change_emb = self.encode_change(img_before, img_after)

        if text is not None:
            text_emb = self.encode_text(text)
            return change_emb, text_emb

        return change_emb
