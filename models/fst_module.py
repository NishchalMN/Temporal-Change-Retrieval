import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySpatialTemporalModule(nn.Module):
    """
    Frequency-Spatial-Temporal fusion.

    Uses FFT to separate high-frequency structural changes from
    low-frequency lighting or seasonal variations.
    """
    
    def __init__(self, channels=768, output_dim=None):
        super().__init__()

        self.channels = channels
        self.output_dim = output_dim if output_dim is not None else channels

        # Frequency band processing
        self.high_freq_conv = nn.Conv2d(channels, channels, 3, padding=1)

        # Spatial attention to focus on change regions
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Temporal fusion combines before, after, and difference
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # Project to embedding
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

    def forward(self, feat_before, feat_after):
        # FFT to frequency domain
        dtype_orig = feat_before.dtype
        before_fft = torch.fft.rfft2(feat_before.float(), norm='ortho')
        after_fft = torch.fft.rfft2(feat_after.float(), norm='ortho')

        # Extract high-frequency components
        h, w = before_fft.shape[-2:]
        high_freq_mask = torch.zeros_like(before_fft)
        high_freq_mask[..., h//2:, :] = 1

        # Convert back to spatial domain
        high_before = torch.fft.irfft2(
            before_fft * high_freq_mask,
            s=feat_before.shape[-2:],
            norm='ortho'
        ).to(dtype_orig)

        high_after = torch.fft.irfft2(
            after_fft * high_freq_mask,
            s=feat_after.shape[-2:],
            norm='ortho'
        ).to(dtype_orig)

        # Compute high-frequency change
        high_change = self.high_freq_conv(high_after - high_before)

        # Spatial attention on change regions
        avg_pool = torch.mean(high_change, dim=1, keepdim=True)
        max_pool = torch.max(high_change, dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        change_mask = self.spatial_attn(spatial_input)

        # Apply attention and fuse temporally
        before_weighted = feat_before * change_mask
        after_weighted = feat_after * change_mask
        diff_weighted = torch.abs(feat_after - feat_before) * change_mask

        fused = self.temporal_fusion(
            torch.cat([before_weighted, after_weighted, diff_weighted], dim=1)
        )

        # Global pooling to embedding
        change_emb = self.projection(fused)

        return change_emb, change_mask
