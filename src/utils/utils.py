import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], file_path: str | Path) -> None:
    """Save dictionary as json file."""

    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    except json.JSONDecodeError as e:
        print(f"Unable to save json file : {e}")


def load_json(file_path: str | Path) -> dict[str, Any]:
    """Load json file as dictionary"""

    try:
        with open(file_path, "r") as file:
            return json.load(file)

    except json.JSONDecodeError as e:
        print(f"Can't load file : {e}")
