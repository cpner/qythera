"""Qythera CLI - beautiful terminal interface."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


BANNER = """
\033[38;5;129m    ██████╗ ██╗   ██╗██╗██████╗ ██╗   ██╗
\033[38;5;141m   ██╔═══██╗██║   ██║██║██╔══██╗╚██╗ ██╔╝
\033[38;5;135m   ██║   ██║██║   ██║██║██████╔╝ ╚████╔╝
\033[38;5;129m   ██║▄▄ ██║██║   ██║██║██╔═══╝  ╚██╔╝
\033[38;5;141m   ╚██████╔╝╚██████╔╝██║██║        ██║
\033[38;5;135m    ╚═════╝  ╚═════╝ ╚═╝╚═╝        ╚═╝
\033[0m    \033[38;5;245mProduction Superintelligence v1.0\033[0m
"""

def cmd_chat():
    from core.knowledge.base import get_answer
    from core.safety import SafetyModerator
    sf = SafetyModerator()

    print(BANNER)
    print("  \033[38;5;245mType your message and press Enter\033[0m")
    print("  \033[38;5;245mType 'help' for commands, 'quit' to exit\033[0m")
    print("  \033[38;5;245m" + "─" * 40 + "\033[0m\n")

    while True:
        try:
            user = input(f"\033[38;5;129m  You\033[0m \033[38;5;245m│\033[0m ").strip()

            if not user:
                continue
            if user.lower() in ("quit", "exit", "q", "выход"):
                print(f"\n  \033[38;5;245m{'─' * 40}\033[0m")
                print(f"  \033[38;5;129mGoodbye! 👋\033[0m\n")
                break
            if user.lower() == "help":
                print(f"\n  \033[38;5;245mCommands:\033[0m")
                print(f"  \033[38;5;245m  help    - Show this help\033[0m")
                print(f"  \033[38;5;245m  quit    - Exit chat\033[0m")
                print(f"  \033[38;5;245m  clear   - Clear screen\033[0m")
                print(f"  \033[38;5;245m  info    - Show system info\033[0m\n")
                continue
            if user.lower() == "clear":
                os.system("clear" if os.name != "nt" else "cls")
                print(BANNER)
                continue
            if user.lower() == "info":
                import platform
                ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
                print(f"\n  \033[38;5;245mSystem Info:\033[0m")
                print(f"  \033[38;5;245m  Platform: {platform.machine()}\033[0m")
                print(f"  \033[38;5;245m  Python: {platform.python_version()}\033[0m")
                print(f"  \033[38;5;245m  RAM: {ram:.1f} GB\033[0m")
                print(f"  \033[38;5;245m  CPU: {os.cpu_count()} cores\033[0m\n")
                continue

            safe, result = sf.filter_input(user)
            if not safe:
                print(f"\n  \033[38;5;196m⚠ {result}\033[0m\n")
                continue

            start = time.time()
            response = get_answer(user)
            elapsed = time.time() - start

            print(f"\n  \033[38;5;129m  Qythera\033[0m \033[38;5;245m│\033[0m")
            for line in response.split("\n"):
                print(f"  \033[38;5;245m  │\033[0m {line}")
            print(f"  \033[38;5;245m  │\033[0m")
            print(f"  \033[38;5;245m  └─ {elapsed:.2f}s\033[0m\n")

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  \033[38;5;129mGoodbye! 👋\033[0m\n")
            break


def cmd_serve():
    from core.inference.server import run_server
    run_server()


def cmd_info():
    import platform
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    print(BANNER)
    print(f"  \033[38;5;245mSystem:\033[0m")
    print(f"  \033[38;5;245m  Platform:  {platform.machine()}\033[0m")
    print(f"  \033[38;5;245m  Python:    {platform.python_version()}\033[0m")
    print(f"  \033[38;5;245m  CPU:       {os.cpu_count()} cores\033[0m")
    print(f"  \033[38;5;245m  RAM:       {ram:.1f} GB\033[0m")
    print(f"  \033[38;5;245m  OS:        {platform.system()}\033[0m")
    print(f"  \033[38;5;245m  Arch:      {platform.machine()}\033[0m\n")


def main():
    if len(sys.argv) < 2:
        cmd_chat()
        return
    cmd = sys.argv[1]
    cmds = {"chat": cmd_chat, "serve": cmd_serve, "info": cmd_info}
    if cmd in cmds:
        cmds[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: chat, serve, info")

if __name__ == "__main__":
    main()
