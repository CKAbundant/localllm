"""Classes for various local llm download and inference."""

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pformat
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import ChatCompletionRequestResponseFormat, Llama

repo_dir = Path(__file__).parents[1].as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.prompt_template import sys_prompt, user_prompt
from src.senti_rater import SentiRating
from src.utils import utils

MODELS = {
    "mistral": (
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    ),
    "llama3": (
        "TheBloke/Meta-Llama-3-8B-Instruct-GGUF",
        "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    ),
    "qwen": ("Qwen/Qwen1.5-7B-Chat-GGUF", "qwen1_5-7b-chat-q4_k_m.gguf"),
    "deepseek": (
        "TheBloke/deepseek-llm-7B-chat-GGUF",
        "deepseek-llm-7b-chat.Q4_K_M.gguf",
    ),
    "gemma": (
        "google/gemma-3-4b-it-qat-q4_0-gguf",
        "gemma-3-4b-it-q4_0.gguf",
    ),  # Requires auth
}


def download_hf(model_name: str = "Gemma-4B") -> None:
    """Download local LLM."""
    load_dotenv()

    repo_id = MODELS[model_name][0]
    file_name = MODELS[model_name][1]

    utils.create_folder("./models")

    hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        local_dir=f"./models/{model_name}",
        token=os.getenv("HF_KEY") if "gemma" in repo_id else None,
    )


class InferLLM(ABC):
    """Abstract class for local llm inference.

    Args:
        model_name (str):
            Name of model family e.g. "gemma".
        model_dir (str):
            Relative path to folder containing GGUF files.
        n_ctx (int):
            Context window i.e. Maximum number of input and output tokens allowed.
        n_threads (int):
            Number of CPU threads. Ryzen 5700 has 8 CPU cores.
        n_gpu_layers (int):
            Number of GPUs available. 0 for no GPU.

    Attributes:
        model_name (str):
            Name of model family e.g. "gemma".
        model_dir (str):
            Relative path to folder containing GGUF files.
        n_ctx (int):
            Context window i.e. Maximum number of input and output tokens allowed.
        n_threads (int):
            Number of CPU threads. Ryzen 5700 has 8 CPU cores.
        n_gpu_layers (int):
            Number of GPUs available. 0 for no GPU.
        model_path (str):
            Relative path to desired GGUF file.
    """

    def __init__(
        self,
        model_name: str,
        model_dir: str,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
    ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.model_path = f"{model_dir}/{model_name}/{MODELS[model_name][1]}"

    @abstractmethod
    def senti_rate(self, news_item: dict[str, int | str]) -> dict[str, Any]:
        """Perform sentiment rating on 'news_items' by generating structured
        JSON output based on 'SentiRating' Pydantic class.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            payload (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        pass

    @abstractmethod
    def gen_payload(self, news_item: dict[str, int | str]) -> dict[str, Any]:
        """Generate payload for local llm chat completion.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        pass


class GemmaLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class."""

    def __init__(
        self,
        model_name: str = "gemma",
        model_dir: str = "./models",
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
    ) -> None:
        super().__init__(model_name, model_dir, n_ctx, n_threads, n_gpu_layers)

    def senti_rate(self, news_item: dict[str, int | str]) -> dict[str, Any]:
        """Perform sentiment rating on 'news_item'.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            payload (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        # Load local llm
        llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
        )

        # Generate payload for chat completion
        payload = self.gen_payload(news_item)

        # Get sentiment rating and reasons
        response = llm.create_chat_completion(**payload)
        content = response["choices"][0]["message"]["content"]

        # Remove message fences and white spaces
        content = re.sub(r"```(json)*|\n\s*", "", content)

        utils.save_json(response, "response.json")

        # Ensure dictionary is returned
        return json.loads(content)

    def gen_payload(self, news_item: dict[str, int | str]) -> dict[str, Any]:
        """Generate payload for local llm chat completion.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            payload (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt.format(news_item=news_item)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": SentiRating.model_json_schema()},
            },
            "temperature": 0.2,
        }

        return payload
