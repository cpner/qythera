import os
from agent.tools.tool_registry import Tool


class FileSystemTool(Tool):
    name = "filesystem"
    description = "Read, write, and list files on the filesystem"

    def __init__(self, base_dir: str = "."):
        self.base_dir = os.path.abspath(base_dir)

    def _safe_path(self, path: str) -> str:
        full = os.path.abspath(os.path.join(self.base_dir, path))
        if not full.startswith(self.base_dir):
            raise ValueError("Path traversal not allowed")
        return full

    def execute(self, action: str = "read", path: str = "", content: str = "", **kwargs) -> str:
        try:
            safe = self._safe_path(path)
            if action == "read":
                if not os.path.exists(safe):
                    return f"File not found: {path}"
                with open(safe) as f:
                    return f.read()[:10000]
            elif action == "write":
                os.makedirs(os.path.dirname(safe), exist_ok=True)
                with open(safe, "w") as f:
                    f.write(content)
                return f"Written to {path}"
            elif action == "list":
                if os.path.isdir(safe):
                    entries = os.listdir(safe)
                    return "\n".join(entries[:100])
                return f"Not a directory: {path}"
            elif action == "exists":
                return str(os.path.exists(safe))
            return f"Unknown action: {action}"
        except Exception as e:
            return f"Filesystem error: {str(e)}"
