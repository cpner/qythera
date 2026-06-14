
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BANNER = """\033[35m
  ╔═══════════════════════════════════╗
  ║       Q y t h e r a   A I         ║
  ║    Powered by Vaelon Model        ║
  ╚═══════════════════════════════════╝
\033[0m"""

def cmd_chat():
    import requests
    print(BANNER + "\n  Type 'quit' to exit\n")
    msgs = []
    while True:
        try:
            user = input("\033[36mYou:\033[0m ").strip()
            if user.lower() in ("quit", "exit", "q"): print("\nGoodbye!"); break
            if not user: continue
            msgs.append({"role": "user", "content": user})
            try:
                r = requests.post("http://localhost:8000/v1/chat/completions",
                                  json={"messages": msgs}, timeout=120)
                reply = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                reply = f"Error: {e}\nStart server: python -m inference.server"
            msgs.append({"role": "assistant", "content": reply})
            print(f"\n\033[35mVaelon:\033[0m {reply}\n")
        except (KeyboardInterrupt, EOFError): print("\nGoodbye!"); break

def cmd_serve():
    from inference.server import run_server
    run_server()

def cmd_web():
    import webbrowser, subprocess
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    if os.path.exists(os.path.join(web_dir, "node_modules")):
        proc = subprocess.Popen(["npm", "run", "dev"], cwd=web_dir)
    else:
        subprocess.run(["npm", "install"], cwd=web_dir)
        proc = subprocess.Popen(["npm", "run", "dev"], cwd=web_dir)
    print("Web UI: http://localhost:3000")
    webbrowser.open("http://localhost:3000")
    proc.wait()

def cmd_info():
    import torch
    print(BANNER)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

def main():
    if len(sys.argv) < 2: cmd_chat(); return
    cmds = {"chat": cmd_chat, "serve": cmd_serve, "web": cmd_web, "info": cmd_info}
    cmd = sys.argv[1]
    if cmd in cmds: cmds[cmd]()
    else: print(f"Unknown: {cmd}. Use: chat, serve, web, info")

if __name__ == "__main__": main()
