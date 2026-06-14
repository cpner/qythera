import os, shutil, json

class FileOpsTool:
    name = "file_ops"
    description = "Read, write, list, copy, move, and delete files"

    def execute(self, action="read", path="", content="", dest="", **kwargs):
        try:
            if action == "read":
                with open(path) as f: return f.read()[:50000]
            elif action == "write":
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                with open(path, 'w') as f: f.write(content)
                return f"Written {len(content)} bytes to {path}"
            elif action == "list":
                if os.path.isdir(path):
                    entries = []
                    for e in os.listdir(path)[:100]:
                        full = os.path.join(path, e)
                        entries.append(f"{'[DIR]' if os.path.isdir(full) else '[FILE]'} {e}")
                    return "\n".join(entries)
                return f"Not a directory: {path}"
            elif action == "exists":
                return str(os.path.exists(path))
            elif action == "copy":
                shutil.copy2(path, dest)
                return f"Copied to {dest}"
            elif action == "move":
                shutil.move(path, dest)
                return f"Moved to {dest}"
            elif action == "delete":
                if os.path.isfile(path): os.remove(path)
                elif os.path.isdir(path): shutil.rmtree(path)
                return f"Deleted {path}"
            elif action == "size":
                return str(os.path.getsize(path))
            return f"Unknown action: {action}"
        except Exception as e:
            return f"FileOps Error: {e}"
