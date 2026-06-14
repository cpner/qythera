import torch
import torch.nn as nn
from typing import Optional

from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel
from multimodal.vision_encoder import VisionEncoder
from multimodal.audio_encoder import AudioEncoder
from multimodal.projection import MultimodalProjection


class MultimodalVaelon(nn.Module):
    def __init__(self, config: VaelonConfig = None, use_vision: bool = True,
                 use_audio: bool = False):
        super().__init__()
        if config is None:
            config = VaelonConfig.vaelon_7b()
        self.language_model = VaelonModel(config)
        self.use_vision = use_vision
        self.use_audio = use_audio

        if use_vision:
            self.vision_encoder = VisionEncoder(projection_dim=config.hidden_size)
            self.vision_proj = MultimodalProjection(config.hidden_size, config.hidden_size)

        if use_audio:
            self.audio_encoder = AudioEncoder(projection_dim=config.hidden_size)
            self.audio_proj = MultimodalProjection(config.hidden_size, config.hidden_size)

    def forward(self, input_ids: torch.LongTensor, attention_mask=None,
                labels=None, pixel_values=None, audio_features=None):
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None and self.use_vision:
            vision_features = self.vision_encoder(pixel_values)
            vision_tokens = self.vision_proj(vision_features).unsqueeze(1)
            inputs_embeds = torch.cat([vision_tokens, inputs_embeds], dim=1)
            if attention_mask is not None:
                vision_mask = torch.ones(attention_mask.size(0), 1, device=attention_mask.device)
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        if audio_features is not None and self.use_audio:
            audio_t = self.audio_encoder(audio_features)
            audio_tokens = self.audio_proj(audio_t).unsqueeze(1)
            inputs_embeds = torch.cat([audio_tokens, inputs_embeds], dim=1)
            if attention_mask is not None:
                audio_mask = torch.ones(attention_mask.size(0), 1, device=attention_mask.device)
                attention_mask = torch.cat([audio_mask, attention_mask], dim=1)

        outputs = self.language_model(
            input_ids=None, attention_mask=attention_mask,
            labels=labels,
        )
        outputs.logits = self.language_model.lm_head(
            self.language_model.norm(
                self.language_model.layers[0].input_norm.weight.new_empty(inputs_embeds.shape)
            )
        )
        return outputs
