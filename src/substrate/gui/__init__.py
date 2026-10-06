"""A local web GUI for substrate: run an experiment, watch it propagate through the scales, edit its inputs and see every output move, and
explore the transfer study. `python -m substrate gui` starts it. Standard library only; the page is plain HTML, CSS and JavaScript."""
from .app import App, GuiError
from .server import make_server, serve

__all__ = ["App", "GuiError", "make_server", "serve"]
