import sys

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def run_chat(model=None, server="http://localhost:8000"):
    console = Console() if HAS_RICH else None

    print("\n  Qythera Chat - Type 'quit' to exit\n")

    messages = []

    while True:
        try:
            user_input = input("\033[36mYou:\033[0m ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            if HAS_REQUESTS:
                try:
                    resp = requests.post(
                        f"{server}/v1/chat/completions",
                        json={"messages": messages, "max_tokens": 2048, "temperature": 0.7},
                        timeout=60,
                    )
                    data = resp.json()
                    assistant_msg = data["choices"][0]["message"]["content"]
                except Exception as e:
                    assistant_msg = f"Error connecting to server: {e}\nMake sure the server is running: qythera serve"
            else:
                assistant_msg = f"Echo: {user_input}\nInstall requests: pip install requests"

            messages.append({"role": "assistant", "content": assistant_msg})

            if HAS_RICH and console:
                console.print(Panel(Markdown(assistant_msg), title="Vaelon", border_style="purple"))
            else:
                print(f"\n\033[35mVaelon:\033[0m {assistant_msg}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break
