import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768, projection_dim: int = 4096):
        super().__init__()
        self.conv1 = nn.Conv1d(80, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.proj = nn.Linear(hidden_size, projection_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        x = audio_features.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=1)
        return self.proj(x)
