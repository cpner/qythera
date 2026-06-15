import numpy as np
from typing import Tuple, Optional


class PatchEmbedding:
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        scale = np.sqrt(2.0 / (patch_size * patch_size * in_channels + embed_dim))
        self.projection = np.random.randn(patch_size * patch_size * in_channels, embed_dim) * scale
        self.bias = np.zeros(embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[np.newaxis]
        batch_size, h, w, c = x.shape
        patches = []
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = x[:, i:i + self.patch_size, j:j + self.patch_size, :]
                patches.append(patch.reshape(batch_size, -1))
        patches = np.stack(patches, axis=1)
        return patches @ self.projection + self.bias


class LayerNorm:
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class MultiHeadAttention:
    def __init__(self, embed_dim: int, num_heads: int):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        scale = np.sqrt(2.0 / (embed_dim + self.head_dim))
        self.W_q = np.random.randn(embed_dim, embed_dim) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        out = np.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        return out @ self.W_o


class TransformerBlock:
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int):
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = LayerNorm(embed_dim)
        scale1 = np.sqrt(2.0 / (embed_dim + mlp_dim))
        self.mlp_fc1 = np.random.randn(embed_dim, mlp_dim) * scale1
        self.mlp_fc2 = np.random.randn(mlp_dim, embed_dim) * scale1 * np.sqrt(mlp_dim / embed_dim)
        self.mlp_bias1 = np.zeros(mlp_dim)
        self.mlp_bias2 = np.zeros(embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.norm1.forward(x)
        h = x + self.attn.forward(h)
        h2 = self.norm2.forward(h)
        h2 = np.maximum(0, h2 @ self.mlp_fc1 + self.mlp_bias1)
        h2 = h2 @ self.mlp_fc2 + self.mlp_bias2
        return h + h2


class ViTEncoder:
    def __init__(self, image_size: int, patch_size: int, in_channels: int,
                 embed_dim: int, num_heads: int, num_layers: int, num_classes: int):
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02
        self.pos_embed = np.random.randn(1, num_patches + 1, embed_dim) * 0.02
        self.blocks = [TransformerBlock(embed_dim, num_heads, embed_dim * 4) for _ in range(num_layers)]
        self.norm = LayerNorm(embed_dim)
        scale = np.sqrt(2.0 / (embed_dim + num_classes))
        self.classifier = np.random.randn(embed_dim, num_classes) * scale
        self.classifier_bias = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[np.newaxis]
        batch_size = x.shape[0]
        patches = self.patch_embed.forward(x)
        cls = np.tile(self.cls_token, (batch_size, 1, 1))
        x = np.concatenate([cls, patches], axis=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        cls_out = x[:, 0]
        return cls_out @ self.classifier + self.classifier_bias


class CLIPLoss:
    def __init__(self, embed_dim: int, vocab_size: int, max_seq_len: int):
        self.embed_dim = embed_dim
        scale = np.sqrt(2.0 / (embed_dim + 128))
        self.text_embed = np.random.randn(vocab_size, embed_dim) * scale
        self.image_proj = np.random.randn(embed_dim, embed_dim) * scale
        self.text_proj = np.random.randn(embed_dim, embed_dim) * scale
        self.logit_scale = np.log(1 / 0.07)

    def forward(self, image_features: np.ndarray, text_features: np.ndarray,
                labels: np.ndarray = None) -> float:
        image_features = image_features @ self.image_proj
        text_features = text_features @ self.text_proj
        image_features = image_features / (np.linalg.norm(image_features, axis=-1, keepdims=True) + 1e-8)
        text_features = text_features / (np.linalg.norm(text_features, axis=-1, keepdims=True) + 1e-8)
        logits = image_features @ text_features.T * np.exp(self.logit_scale)
        if labels is None:
            labels = np.arange(len(logits))
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        log_sum_exp = np.log(exp_logits.sum(axis=1) + 1e-8) + logits_max.squeeze()
        log_probs = logits - log_sum_exp[:, np.newaxis]
        loss = -np.mean(np.sum(log_probs * np.eye(len(logits))[labels], axis=1))
        return float(loss)


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160,
                 n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_filterbank = self._create_mel_filterbank()

    def _create_mel_filterbank(self) -> np.ndarray:
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.round(hz_points / (self.sample_rate / self.n_fft)).astype(int)
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)
        return filterbank

    def stft(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.squeeze()
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        stft_result = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.complex128)
        window = np.hanning(self.n_fft)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft] * window
            fft = np.fft.rfft(frame)
            stft_result[:, i] = fft
        return stft_result

    def mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        stft = self.stft(audio)
        power_spectrum = np.abs(stft) ** 2
        mel_spec = self.mel_filterbank @ power_spectrum
        return mel_spec

    def log_compress(self, mel_spec: np.ndarray) -> np.ndarray:
        return np.log(mel_spec + 1e-8)

    def forward(self, audio: np.ndarray) -> np.ndarray:
        mel_spec = self.mel_spectrogram(audio)
        log_mel = self.log_compress(mel_spec)
        return log_mel


class CLIPModel:
    def __init__(self, embed_dim: int = 512, image_size: int = 224, patch_size: int = 32,
                 in_channels: int = 3, num_heads: int = 8, num_layers: int = 6,
                 vocab_size: int = 49408, max_text_len: int = 77, temperature_init: float = 0.07):
        self.embed_dim = embed_dim
        self.temperature = temperature_init
        self.image_encoder = ViTEncoder(image_size, patch_size, in_channels, embed_dim, num_heads, num_layers, embed_dim)
        self.image_proj = np.random.randn(embed_dim, embed_dim) * np.sqrt(2.0 / (embed_dim * 2))
        self.text_embedding = np.random.randn(vocab_size, embed_dim) * np.sqrt(2.0 / vocab_size)
        self.text_proj = np.random.randn(embed_dim, embed_dim) * np.sqrt(2.0 / (embed_dim * 2))
        self.text_pos = np.random.randn(max_text_len, embed_dim) * 0.02
        self.text_attn = MultiHeadAttention(embed_dim, num_heads)
        self.text_norm = LayerNorm(embed_dim)

    def encode_image(self, images: np.ndarray) -> np.ndarray:
        features = self.image_encoder.forward(images)
        features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
        return features

    def encode_text(self, token_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = token_ids.shape
        x = self.text_embedding[token_ids]
        x = x + self.text_pos[:seq_len]
        x = self.text_attn.forward(x)
        x = self.text_norm.forward(x)
        mask = (token_ids != 0).astype(np.float64)
        x = x * mask[:, :, np.newaxis]
        pooled = x.sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-8)
        pooled = pooled / (np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-8)
        return pooled

    def forward(self, images: np.ndarray, token_ids: np.ndarray) -> float:
        image_features = self.encode_image(images) @ self.image_proj
        text_features = self.encode_text(token_ids) @ self.text_proj
        image_features = image_features / (np.linalg.norm(image_features, axis=-1, keepdims=True) + 1e-8)
        text_features = text_features / (np.linalg.norm(text_features, axis=-1, keepdims=True) + 1e-8)
        logits = image_features @ text_features.T / self.temperature
        batch_size = len(logits)
        labels = np.arange(batch_size)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        log_sum_exp = np.log(exp_logits.sum(axis=1) + 1e-8)
        log_probs = logits - log_sum_exp[:, np.newaxis]
        loss_i2t = -np.mean(np.sum(log_probs * np.eye(batch_size)[labels], axis=1))
        log_probs_t = logits.T - np.log(exp_logits.sum(axis=0, keepdims=True).T + 1e-8)
        loss_t2i = -np.mean(np.sum(log_probs_t * np.eye(batch_size)[labels], axis=1))
        return (loss_i2t + loss_t2i) / 2


class VideoProcessor:
    def __init__(self, frame_size: int = 224, patch_size: int = 16, num_channels: int = 3,
                 temporal_patch_size: int = 4, num_frames: int = 8, embed_dim: int = 256):
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.temporal_patch_size = temporal_patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.spatial_patches_per_frame = (frame_size // patch_size) ** 2
        self.patch_dim = num_channels * patch_size * patch_size
        self.flat_dim = self.temporal_patch_size * self.spatial_patches_per_frame * self.patch_dim
        scale = np.sqrt(2.0 / (self.flat_dim + embed_dim))
        self.video_embed = np.random.randn(self.flat_dim, embed_dim) * scale
        self.video_bias = np.zeros(embed_dim)
        self.temporal_pos = np.random.randn(num_frames // temporal_patch_size, embed_dim) * 0.02

    def sample_frames(self, video: np.ndarray, num_frames: int = None) -> np.ndarray:
        if num_frames is None:
            num_frames = self.num_frames
        total_frames = video.shape[0]
        if total_frames <= num_frames:
            indices = np.arange(total_frames)
            padded = np.zeros((num_frames,) + video.shape[1:], dtype=video.dtype)
            padded[:total_frames] = video[indices]
            return padded
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return video[indices]

    def extract_temporal_patches(self, video: np.ndarray) -> np.ndarray:
        num_frames = video.shape[0]
        t_groups = num_frames // self.temporal_patch_size
        h, w = video.shape[1], video.shape[2]
        patches = []
        for t in range(t_groups):
            frame_patches = []
            for f in range(t * self.temporal_patch_size, (t + 1) * self.temporal_patch_size):
                for i in range(0, h, self.patch_size):
                    for j in range(0, w, self.patch_size):
                        patch = video[f, i:i+self.patch_size, j:j+self.patch_size].flatten()
                        frame_patches.append(patch)
            patches.append(np.concatenate(frame_patches))
        return np.array(patches, dtype=np.float64)

    def forward(self, video: np.ndarray) -> np.ndarray:
        video = self.sample_frames(video)
        temporal_patches = self.extract_temporal_patches(video)
        embeddings = temporal_patches @ self.video_embed + self.video_bias
        t_groups = len(embeddings)
        embeddings = embeddings + self.temporal_pos[:t_groups]
        return embeddings


class SimpleDiffusion:
    def __init__(self, input_dim: int, num_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1 - self.alpha_cumprod)

        hidden_dim = input_dim * 4
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.fc1 = np.random.randn(input_dim + 1, hidden_dim) * scale
        self.fc2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.fc_out = np.random.randn(hidden_dim, input_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.b_out = np.zeros(input_dim)

    def forward_noise(self, x: np.ndarray, t: int, noise: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        if noise is None:
            noise = np.random.randn(*x.shape)
        sqrt_alpha = self.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t]
        noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return noisy_x, noise

    def denoise(self, x: np.ndarray, t: int) -> np.ndarray:
        t_normalized = np.full((len(x), 1), t / self.num_timesteps)
        h = np.concatenate([x, t_normalized], axis=-1)
        h = np.maximum(0, h @ self.fc1 + self.b1)
        h = np.maximum(0, h @ self.fc2 + self.b2)
        return h @ self.fc_out + self.b_out

    def forward(self, x: np.ndarray, t: int) -> np.ndarray:
        noisy_x, noise = self.forward_noise(x, t)
        predicted_noise = self.denoise(noisy_x, t)
        loss = np.mean((predicted_noise - noise) ** 2)
        return loss

    def sample(self, num_samples: int, num_steps: int = None) -> np.ndarray:
        if num_steps is None:
            num_steps = self.num_timesteps
        x = np.random.randn(num_samples, self.input_dim)
        for t in reversed(range(num_steps)):
            predicted_noise = self.denoise(x, t)
            alpha = self.alphas[t]
            alpha_cumprod = self.alpha_cumprod[t]
            beta = self.betas[t]
            if t > 0:
                noise = np.random.randn(*x.shape)
            else:
                noise = np.zeros_like(x)
            x = (1 / np.sqrt(alpha)) * (x - (beta / np.sqrt(1 - alpha_cumprod)) * predicted_noise) + np.sqrt(beta) * noise
        return x
