import os
from pathlib import Path

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[1]))
ZIP_PATH = Path(os.environ.get("MLCHALL_ZIP", ROOT / "6ab10eb3b23ba_student_resource.zip"))
DATA_DIR = ROOT / "data"
TRAIN_PREFIX = "student_resource/dataset/train"
