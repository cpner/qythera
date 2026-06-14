import subprocess, sys

def deploy(provider="aws", region="us-east-1"):
    configs = {
        "aws": {"script": "infra/terraform/main.tf", "init": "cd infra/terraform && terraform init && terraform apply -auto-approve"},
        "gcp": {"script": "cloud/gcp/deploy.py", "init": "python3 cloud/gcp/deploy.py"},
        "azure": {"script": "cloud/azure/deploy.py", "init": "python3 cloud/azure/deploy.py"},
        "docker": {"init": "docker compose -f infra/docker-compose.yml up -d"},
        "kubernetes": {"init": "kubectl apply -f infra/kubernetes/"},
    }
    if provider not in configs:
        print(f"Unknown provider: {provider}. Available: {list(configs.keys())}")
        return
    print(f"Deploying to {provider}...")
    subprocess.run(configs[provider]["init"], shell=True)

if __name__ == "__main__":
    deploy(sys.argv[1] if len(sys.argv) > 1 else "docker")
