import json
from pathlib import Path
from typing import Any


def create_folder(dir_path: str | Path) -> None:
    """Create folder if not exist."""

    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)


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


def get_token_usage(response: dict[str, Any]) -> dict[str, int]:
    """Get token usage from 'usage' keys in json output.

    Args:
        response (dict[str, Any]):
            Output from Perplexity API after sending payload.

    Returns:
        (dict[str, int]):
            Dictionary containing 'id', 'prompt_tokens', 'completion_tokens'
            and 'total_tokens'.
    """

    # Get 'id' info in 'response["choices"][0]["message"]["content"]
    content = json.loads(response["choices"][0]["message"]["content"])
    token_dict = {"id": content["id"]}

    # Remove 'search_context_size' from 'usage'
    usage_dict = {
        k: v for k, v in response["usage"].items() if k != "search_context_size"
    }

    return dict(**token_dict, **usage_dict)
