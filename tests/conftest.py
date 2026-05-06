import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))