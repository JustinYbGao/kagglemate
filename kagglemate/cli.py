"""CLI entrypoint shim for the `km` and `kagglemate` console scripts.

The Typer app is defined in `main.py` at the project root.  This module makes
it importable as ``kagglemate.cli:app`` so that ``[project.scripts]`` works
reliably in editable installs without duplicating command definitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# main.py lives at the project root; ensure it is importable from an installed
# console script entry point.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from main import app  # noqa: E402

__all__ = ["app"]
