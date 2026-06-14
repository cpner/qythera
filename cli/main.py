import sys
import os

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


console = Console() if HAS_RICH else None


def print_rich(text, style=None):
    if HAS_RICH and console:
        console.print(text, style=style)
    else:
        print(text)


def print_banner():
    banner = """
    ╔══════════════════════════════════════╗
    ║         Qythera AI v0.1.0            ║
    ║    Powered by Vaelon Architecture    ║
    ╚══════════════════════════════════════╝
    """
    print_rich(banner, style="bold cyan")


if HAS_CLICK:
    @click.group()
    @click.version_option(version="0.1.0")
    def cli():
        """Qythera: Production Superintelligence CLI."""
        pass

    @cli.command()
    @click.option("--model", default=None, help="Model name or path")
    @click.option("--server", default="http://localhost:8000", help="Inference server URL")
    def chat(model, server):
        """Start interactive chat with Qythera."""
        from cli.commands.chat import run_chat
        run_chat(model=model, server=server)

    @cli.command()
    @click.option("--port", default=3000, help="Web UI port")
    def web(port):
        """Launch Qythera web interface."""
        from cli.commands.web import launch_web
        launch_web(port=port)

    @cli.command()
    @click.option("--config", default=None, help="Training config path")
    @click.option("--gpus", default=1, help="Number of GPUs")
    def train(config, gpus):
        """Start model training."""
        from cli.commands.train import run_train
        run_train(config=config, gpus=gpus)

    @cli.command()
    @click.option("--model", default=None, help="Model path")
    @click.option("--port", default=8000, help="Server port")
    @click.option("--host", default="0.0.0.0", help="Server host")
    def serve(model, port, host):
        """Start inference server."""
        from cli.commands.serve import run_serve
        run_serve(model=model, port=port, host=host)

    @cli.command()
    def info():
        """Show Qythera system information."""
        import torch
        print_banner()
        print_rich(f"PyTorch: {torch.__version__}", style="green")
        print_rich(f"CUDA available: {torch.cuda.is_available()}", style="green")
        if torch.cuda.is_available():
            print_rich(f"GPU: {torch.cuda.get_device_name(0)}", style="green")
            print_rich(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", style="green")

    def main():
        cli()
else:
    def main():
        print_banner()
        print("Usage: qythera [chat|web|train|serve|info]")
        print("Install click: pip install click")

if __name__ == "__main__":
    main()
