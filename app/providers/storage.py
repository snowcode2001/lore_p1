import json
from pathlib import Path
from datetime import datetime, UTC


class JSONFileStorage:
    """Stores beliefs in a human-readable JSON file."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def _load(self) -> dict:
        if not self.filepath.exists():
            return {}
        return json.loads(self.filepath.read_text())

    def _save(self, data: dict) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.write_text(json.dumps(data, indent=2))

    def save_beliefs(self, user_id: int, beliefs: list[dict]) -> None:
        data = self._load()
        key = str(user_id)
        if key not in data:
            data[key] = []
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "beliefs": beliefs,
        }
        data[key].append(entry)
        self._save(data)

    def save_generic(self, user_id, records: list[dict]) -> None:
        data = self._load()
        key = str(user_id)
        if key not in data:
            data[key] = []
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "records": records,
        }
        data[key].append(entry)
        self._save(data)

    def get_history(self, user_id: int) -> list[dict]:
        data = self._load()
        return data.get(str(user_id), [])
