SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "ja", "zh", "ko", "ru", "ar", "pt", "it", "nl"]

LANGUAGE_PROMPTS = {
    "en": "You are a helpful assistant.",
    "es": "Eres un asistente util.",
    "fr": "Vous etes un assistant utile.",
    "de": "Sie sind ein hilfreicher Assistent.",
    "ja": "あなたは有用なアシスタントです。",
    "zh": "你是一个有用的助手。",
    "ko": "당신은 유용한 어시스턴트입니다.",
    "ru": "Вы полезный помощник.",
    "ar": "أنت مساعد مفيد.",
    "pt": "Voce e um assistente util.",
    "it": "Sei un assistente utile.",
    "nl": "Je bent een behulpzame assistent.",
}

def get_system_prompt(lang_code="en"):
    return LANGUAGE_PROMPTS.get(lang_code, LANGUAGE_PROMPTS["en"])

def detect_language(text):
    for char in text:
        if ord(char) > 0x4E00 and ord(char) < 0x9FFF: return "zh"
        if ord(char) > 0x3040 and ord(char) < 0x309F: return "ja"
        if ord(char) > 0xAC00 and ord(char) < 0xD7AF: return "ko"
    return "en"
