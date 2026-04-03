# HuggingFace Spaces entry point.
# HF Spaces with FastAPI SDK looks for app.py in the root.
# We simply re-export the FastAPI app from main.py.
from main import app  # noqa: F401
