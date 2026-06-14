# Deployment Guide

## Docker Compose (Recommended)

```bash
docker compose -f infra/docker-compose.yml up -d
```

## Kubernetes

```bash
kubectl apply -f infra/kubernetes/
```

## Cloud (Terraform)

```bash
cd infra/terraform
terraform init
terraform apply
```

## Environment Variables

- `VAELEN_API_URL` - Inference server URL (default: http://localhost:8000)
- `VAELEN_MODEL_PATH` - Path to model weights
- `NEXT_PUBLIC_API_URL` - Frontend API URL
