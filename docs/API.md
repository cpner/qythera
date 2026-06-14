# API Reference

## POST /v1/chat/completions
```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 512,
  "temperature": 0.7
}
```

## GET /health
Returns server status.

## GET /v1/models
Returns available models.
