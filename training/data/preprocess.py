import re
from typing import List, Dict, Optional


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r' +', ' ', text).strip()
    return text


def format_chatml(messages, system_prompt=None):
    parts = []
    if system_prompt:
        parts.append('<|system|>
' + system_prompt + '<|end|>')
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        parts.append('<|' + role + '|>
' + content + '<|end|>')
    parts.append('<|assistant|>
')
    return '
'.join(parts)


def filter_by_length(samples, min_len=10, max_len=8192):
    filtered = []
    for s in samples:
        total = sum(len(m.get('content', '')) for m in s.get('messages', []))
        if min_len <= total <= max_len:
            filtered.append(s)
    return filtered


def deduplicate(samples):
    seen = set()
    unique = []
    for s in samples:
        key = str([(m.get('role',''), m.get('content','')[:50]) for m in s.get('messages',[])])
        h = hash(key)
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique


def preprocess_dataset(samples, min_len=10, max_len=8192):
    cleaned = []
    for s in samples:
        msgs = s.get('messages', s.get('conversation', []))
        if isinstance(msgs, list):
            cleaned_msgs = []
            for m in msgs:
                if isinstance(m, dict) and 'content' in m:
                    cleaned_msgs.append({
                        'role': m.get('role', 'user'),
                        'content': clean_text(m['content']),
                    })
            if cleaned_msgs:
                cleaned.append({'messages': cleaned_msgs})
    cleaned = deduplicate(cleaned)
    cleaned = filter_by_length(cleaned, min_len, max_len)
    return cleaned
