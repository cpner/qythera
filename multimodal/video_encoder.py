import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    def __init__(self, projection_dim: int = 4096, num_frames: int = 8):
        super().__init__()
        self.num_frames = num_frames
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(2),
            nn.Linear(64 * 7 * 7, 2048),
        )
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.proj = nn.Linear(2048, projection_dim)

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, h, w = video_frames.shape
        x = video_frames.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, h, w)
        features = self.frame_encoder(x)
        features = features.view(batch, frames, -1)
        features = self.temporal(features)
        features = features.mean(dim=1)
        return self.proj(features)
