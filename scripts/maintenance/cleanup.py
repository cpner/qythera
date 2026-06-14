import os, shutil

def cleanup_build_artifacts(project_root="."):
    dirs_to_clean = ["__pycache__", ".pytest_cache", "node_modules/.cache", ".ruff_cache"]
    files_to_clean = ["*.pyc", "*.pyo", "*.egg-info"]
    for dirpath, dirnames, filenames in os.walk(project_root):
        for d in dirnames:
            if d in dirs_to_clean or d in [".git", ".venv", "venv"]:
                continue
        for f in filenames:
            for pattern in files_to_clean:
                if f.endswith(pattern[1:]):
                    os.remove(os.path.join(dirpath, f))
    print("Cleanup complete")

if __name__ == "__main__":
    cleanup_build_artifacts()
