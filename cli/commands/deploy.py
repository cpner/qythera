import subprocess
import os


def deploy_cloud(provider="docker"):
    print(f"\n  Deploying Qythera via {provider}...\n")
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")

    if provider == "docker":
        cmd = ["docker", "compose", "-f", os.path.join(project_root, "infra", "docker-compose.yml"), "up", "-d"]
    elif provider == "kubernetes":
        cmd = ["kubectl", "apply", "-f", os.path.join(project_root, "infra", "kubernetes")]
    else:
        print(f"Unknown provider: {provider}")
        return

    try:
        subprocess.run(cmd, check=True)
        print("Deployment complete!")
    except FileNotFoundError:
        print(f"{provider} not found. Please install it first.")
    except subprocess.CalledProcessError as e:
        print(f"Deployment failed: {e}")
