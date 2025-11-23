from pathlib import Path

# Repo root is 2 levels up from utils/config.py
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

def get_root_dir() -> Path:
    return ROOT_DIR

def get_data_dir() -> Path:
    return DATA_DIR

def get_models_dir() -> Path:
    return MODELS_DIR
