import json
from pathlib import Path
from typing import Any


def save_json(file_path: str, Path, data: dict[str, Any]) -> None:
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    except json.JSONDecodeError as e:
        print(f"Unable to save json file : {e}")
