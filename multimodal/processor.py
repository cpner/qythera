from typing import List, Optional


class MultimodalProcessor:
    def __init__(self):
        pass

    def process_images(self, images, max_size: int = 384):
        return images

    def process_audio(self, audio, sample_rate: int = 16000):
        return audio

    def process_video(self, video, num_frames: int = 8):
        return video

    def format_multimodal_prompt(self, text: str, has_image: bool = False,
                                  has_audio: bool = False) -> str:
        parts = []
        if has_image:
            parts.append("[Image]")
        if has_audio:
            parts.append("[Audio]")
        parts.append(text)
        return " ".join(parts)
