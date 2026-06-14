# Deployment

## Docker
```bash
docker build -t qythera -f inference/Dockerfile .
docker run --gpus all -p 8000:8000 qythera
```

## Cloud
- AWS: Use g5.xlarge instances
- GCP: Use a2-highgpu-1g
- Azure: Use NC6s_v3

## Mobile Access
The web UI works on any device via browser. Heavy computation runs on GPU server.
