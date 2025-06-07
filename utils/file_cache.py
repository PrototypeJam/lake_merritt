"""Simple file cache."""
import json
from pathlib import Path
from hashlib import sha256


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def get(key: str):
    path = CACHE_DIR / sha256(key.encode()).hexdigest()
    if path.exists():
        return json.loads(path.read_text())
    return None


def set(key: str, value):
    path = CACHE_DIR / sha256(key.encode()).hexdigest()
    path.write_text(json.dumps(value))
