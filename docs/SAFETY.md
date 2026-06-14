# Safety Features

## Content Moderation

Qythera includes built-in safety filters:

1. **Toxicity Detection**: Pattern-based toxic content detection
2. **Jailbreak Filter**: Detects adversarial prompt attacks
3. **PII Redaction**: Automatically redacts personally identifiable information

## Usage

```python
from safety.moderation_api import ModerationAPI

mod = ModerationAPI()
result = mod.moderate("User input text")
if not result["safe"]:
    print("Content blocked")
```
