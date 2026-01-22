# src/ui/bootstrap.py
from __future__ import annotations

import sys
from pathlib import Path

def bootstrap() -> None:
    """
    Ensure project root is importable so `import src.*` works
    for Streamlit pages under /pages.
    """
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
