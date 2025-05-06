import importlib
import json
import re
from pathlib import Path
from typing import Any, Type, TypeVar

from omegaconf import DictConfig, OmegaConf, ValidationError

# Create generic type variable 'T'
T = TypeVar("T")


def load_config(cfg_path: str = "./config/config.yaml") -> DictConfig | None:
    """Load configuration parameters in 'config.yaml'."""

    if not Path(cfg_path).is_file():
        raise FileNotFoundError(f"No 'config.yaml' file found at '{cfg_path}'.")

    try:
        return OmegaConf.load(cfg_path)

    except ValidationError as e:
        print(f"Unable to load configuration : {e}")


def get_class_instance(
    class_name: str, script_path: str, **params: dict[str, Any]
) -> T:
    """Return instance of a class that is initialized with 'params'.

    Args:
        class_name (str):
            Name of class in python script.
        script_path (str):
            Relative file path to python script that contains the required class.
        **params (dict[str, Any]):
            Arbitrary Keyword input arguments to initialize class instance.

    Returns:
        (T): Initialized instance of class.
    """

    # Convert script path to package path
    module_path = convert_path_to_pkg(script_path)

    try:
        # Import python script at class path as python module
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Module not found in '{script_path}' : {e}")

    try:
        # Get class from module
        req_class: Type[T] = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"'{class_name}' class is not found in module.")

    # Intialize instance of class
    return req_class(**params)


def convert_path_to_pkg(script_path: str) -> str:
    """Convert file path to package path that can be used as input to importlib."""

    script_path = Path(script_path)

    if not script_path.is_file():
        raise FileNotFoundError(
            f"{script_path.name} is not found in '{script_path.parent}' folder."
        )

    # Remove suffix ".py"
    script_path = Path(script_path).with_suffix("").as_posix()

    # Convert to package format for use in 'importlib.import_module'
    return script_path.replace("/", ".")


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


def remove_think(text: str) -> str:
    """Remove the think block i.e. '<think> ... </think>' from 'text' string."""

    # Get the start and end index for think block
    start_idx = text.find("<think>")
    end_idx = text.find("</think>") + len("</think>")

    # Exclude think block
    return f"{text[0:start_idx]}\n{text[end_idx:]}"


def extract_dict_response(text: str) -> str:
    """Extract dictionary response from local LLM i.e. exclude thinking quotes."""

    # Get the starting message fence
    msg_fence = re.findall(r"```\w*", text)[0]

    # Get start and end index
    start_idx = text.find(msg_fence) + len(msg_fence)
    end_idx = -3  # Exclude ``` at the end

    # Extract dictionary response
    text = text[start_idx:end_idx]

    # Remove new lines and extra whitespaces
    return re.sub(r"\n+|\s\s+", "", text)
