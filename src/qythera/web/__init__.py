import os

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(WEB_DIR, "ui.html")
SW_PATH = os.path.join(WEB_DIR, "sw.js")
MANIFEST_PATH = os.path.join(WEB_DIR, "manifest.json")

__all__ = ["WEB_DIR", "UI_PATH", "SW_PATH", "MANIFEST_PATH"]