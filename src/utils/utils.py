import importlib
import json
import re
from pathlib import Path
from typing import Any, Type, TypeVar

from omegaconf import DictConfig, OmegaConf, ValidationError

from src.prompt_template import (
    batch_sys_prompt,
    batch_user_prompt,
    sys_prompt,
    user_prompt,
)

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


def extract_json_response(text: str) -> str:
    """Extract dictionary response from local LLM i.e. exclude thinking quotes."""

    # List message fence if any
    msg_fence_list = re.findall(r"```\w*", text)

    # If message fence exist
    if len(msg_fence_list) > 0:
        # Get the starting message fence
        msg_fence = msg_fence_list[0]

        # Get start and end index
        start_idx = text.find(msg_fence) + len(msg_fence)
        end_idx = -3  # Exclude ``` at the end

        # Extract dictionary response
        text = text[start_idx:end_idx]

    # Remove new lines and extra whitespaces
    text = re.sub(r"\n+|\s\s+", "", text)

    # Ensure double quotes
    return ensure_double_quotes(text)


def ensure_double_quotes(text: str) -> str:
    """Ensure double quotes for keys and each reason in 'reasons' list."""

    # Ensure double quotes for keys i.e. 'id', 'rating', 'reasons'
    for key in ["id", "rating", "reasons"]:
        text = re.sub(f"'{key}'", f'"{key}"', text)

    # Convert ', ' to ", "
    text = re.sub(r"',\s*'", '", "', text)

    # Convert '[ to "[ and '] to "]
    text = re.sub(r"(?<=\[)'|'(?=\])", '"', text)

    return text


def init_sys_usr_prompt(
    news: dict[str, int | str] | list[dict[str, int | str]],
) -> tuple[str, str]:
    """Initialize system and user prompts given news item or list of news items.

    - Use batch system and user prompt if 'news' is list of news items.
    - Use normal system and user prompt if 'news' is dictionary type.

    Args:
        news (dict[str, int  |  str] | list[dict[str, int | str]]):
            List of dictionary or dictionary containing 'id', 'ticker'
            and 'news' info.

    Returns:
        sys_p (str): System prompt updated with 'news'.
        usr_p (str): User prompt updated with 'news'.
    """

    if isinstance(news, list):
        sys_p = batch_sys_prompt
        usr_p = batch_user_prompt.format(news_list=news)

    else:
        sys_p = sys_prompt
        usr_p = user_prompt.format(news_item=news)

    return sys_p, usr_p
