import subprocess
import sys
import os
import webbrowser


def launch_web(port=3000):
    web_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web")
    web_dir = os.path.abspath(web_dir)

    print(f"\n  Starting Qythera Web UI on port {port}...\n")

    if os.path.exists(os.path.join(web_dir, "node_modules")):
        cmd = ["npm", "run", "dev"]
    else:
        print("Installing web dependencies...")
        subprocess.run(["npm", "install"], cwd=web_dir, check=False)
        cmd = ["npm", "run", "dev"]

    try:
        proc = subprocess.Popen(cmd, cwd=web_dir)
        print(f"Web UI: http://localhost:{port}")
        webbrowser.open(f"http://localhost:{port}")
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping web UI...")
        proc.terminate()
