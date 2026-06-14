"""Qythera CLI - command line interface."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def cmd_chat():
    print("\n  Qythera Chat\n  Type 'quit' to exit\n")
    from core.knowledge.base import get_answer
    from core.safety import SafetyModerator
    sf = SafetyModerator()
    while True:
        try:
            user = input("\033[36mYou:\033[0m ").strip()
            if user.lower() in ("quit", "exit", "q"): print("\nGoodbye!"); break
            if not user: continue
            safe, result = sf.filter_input(user)
            if not safe:
                print(f"\n\033[31m{result}\033[0m")
                continue
            response = get_answer(user)
            print(f"\n\033[35mQythera:\033[0m {response}\n")
        except (KeyboardInterrupt, EOFError): print("\nGoodbye!"); break


def cmd_serve():
    from core.inference.server import run_server
    run_server()


def cmd_info():
    import platform, os
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    print(f"\n  Qythera v1.0.0")
    print(f"  Platform: {platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  RAM: {ram:.1f} GB")
    print(f"  OS: {platform.system()}\n")


def main():
    if len(sys.argv) < 2: cmd_chat(); return
    cmd = sys.argv[1]
    cmds = {"chat": cmd_chat, "serve": cmd_serve, "info": cmd_info}
    if cmd in cmds: cmds[cmd]()
    else: print(f"Unknown: {cmd}. Commands: chat, serve, info")

if __name__ == "__main__": main()
