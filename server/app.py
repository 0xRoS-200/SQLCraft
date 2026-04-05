import sys
import os
from pathlib import Path

# Ensure root is in sys.path so we can import main.py from any context.
# This helps the platform's evaluator find the FastAPI app.
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app

def main():
    """Main entry point for the OpenEnv server command."""
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
