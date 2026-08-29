"""Shared bootstrap for the diagnostic scripts: puts lstm_common on the path."""
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "lstm_models", REPO / "backtesting"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
warnings.filterwarnings("ignore")
