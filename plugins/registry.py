from typing import Dict, Any, Callable

class Plugin:
    def __init__(self, name, version, description, hooks=None):
        self.name = name
        self.version = version
        self.description = description
        self.hooks = hooks or {}

class PluginRegistry:
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, list] = {}

    def register(self, plugin: Plugin):
        self.plugins[plugin.name] = plugin
        for hook_name, hook_fn in plugin.hooks.items():
            if hook_name not in self.hooks: self.hooks[hook_name] = []
            self.hooks[hook_name].append(hook_fn)

    def trigger(self, hook_name, *args, **kwargs):
        results = []
        for fn in self.hooks.get(hook_name, []):
            results.append(fn(*args, **kwargs))
        return results

    def list_plugins(self):
        return [{"name": p.name, "version": p.version, "description": p.description}
                for p in self.plugins.values()]
