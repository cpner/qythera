import torch
import torch.nn as nn

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 projection_dim: int = 4096):
        super().__init__()
        self.projection_dim = projection_dim
        if HAS_CLIP:
            self.clip = CLIPVisionModel.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            self.proj = nn.Linear(self.clip.config.hidden_size, projection_dim)
        else:
            self.clip = None
            self.proj = nn.Linear(768, projection_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.clip:
            features = self.clip(pixel_values=pixel_values).last_hidden_state[:, 0]
        else:
            features = torch.randn(pixel_values.size(0), 768, device=pixel_values.device)
        return self.proj(features)
