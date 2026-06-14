from plugins.registry import PluginRegistry, Plugin

class PluginManager:
    def __init__(self):
        self.registry = PluginRegistry()

    def load_plugins(self, directory="plugins/"):
        import importlib, os
        for fname in os.listdir(directory):
            if fname.endswith(".py") and not fname.startswith("_"):
                module = importlib.import_module(f"plugins.{fname[:-3]}")
                if hasattr(module, "register"):
                    module.register(self.registry)
