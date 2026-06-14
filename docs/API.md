# API Documentation

## Endpoints

### POST /v1/chat/completions

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 2048,
  "temperature": 0.7
}
```

### GET /v1/models

Returns available models.

### GET /health

Health check endpoint.
