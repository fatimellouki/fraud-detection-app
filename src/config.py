"""Re-export config from project root for convenience."""

import sys
import os

# Ensure project root is on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import *  # noqa: F401, F403
