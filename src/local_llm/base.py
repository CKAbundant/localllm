"""Abstract Classes and function for local LLM download and inference."""

import ast
import time
from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pformat
from typing import Any

import httpx
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

from src.local_llm.timed_method import TimedMethod
from src.utils import utils


def download_hf(
    repo_id: str, filename: str, model_dir: str, token: str | None = None
) -> None:
    """Download local LLM."""

    utils.create_folder(model_dir)

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
        token=token,
    )


class InferLLM(ABC):
    """Abstract class for local llm inference.

    Args:
        model_path (str):
            Relative path to desired GGUF file.
        n_ctx (int):
            Context window i.e. Maximum number of input and output tokens allowed.
        n_threads (int):
            Number of CPU threads. Ryzen 5700 has 8 CPU cores.
        n_gpu_layers (int):
            Number of GPUs available. 0 for no GPU.
        verbose (bool):
            Whether to show all info.
        temperature (float):
            Temperature to use for sampling.

    Attributes:
        model_path (str):
            Relative path to desired GGUF file.
        n_ctx (int):
            Context window i.e. Maximum number of input and output tokens allowed.
        n_threads (int):
            Number of CPU threads. Ryzen 5700 has 8 CPU cores.
        n_gpu_layers (int):
            Number of GPUs available. 0 for no GPU.
        verbose (bool):
            Whether to show all info.
        temperature (float):
            Temperature to use for sampling.
        timings (list[float]):
            List to store time taken to 'senti_rate' method.
        json_grammar (LlamaGrammar):
            GBNF grammar for JSON output.
        llm (Llama):
            Initialized instance of Llama class for specific local LLM.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
        verbose: bool,
        temperature: float,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.temperature = temperature
        self.timings = []
        self.json_grammar = self.get_json_grammar()
        self.llm = None

    @abstractmethod
    def gen_llm(self) -> Llama:
        """Generate initialized instance of Llama for specific local LLM."""

        pass

    @abstractmethod
    def gen_payload(
        self, news: dict[str, int | str] | list[dict[str, int | str]]
    ) -> dict[str, Any]:
        """Generate payload for local llm chat completion.

        Args:
            news (dict[str, int  |  str] | list[dict[str, int | str]]):
                List of dictionary or dictionary containing 'id', 'ticker'
                and 'news' info.

        Returns:
            payload (dict[str, int | str]):
                Dictionary containing input parameters for 'create_chat_completion'
                method.
        """

        pass

    def get_json_grammar(self) -> LlamaGrammar:
        """Load GBNF Grammar for JSON from 'json.gbnf' file."""

        # Hard code relative path to 'json.gnbf'
        gbnf_path = f"./models/json.gbnf"
        url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars/json_arr.gbnf"

        if not Path(gbnf_path).is_file():
            # Download 'json.gbnf' via httpx
            response = httpx.get(url)
            response.raise_for_status()
            grammar_text = response.text

            # Save as 'json.gbnf' file in 'models' folder
            with open(gbnf_path, "w", encoding="utf-8") as file:
                file.write(grammar_text)

        # Load 'json.gbnf' file
        with open(gbnf_path, "r", encoding="utf-8") as file:
            grammar_text = file.read()

        return LlamaGrammar.from_string(grammar_text)

    @TimedMethod
    def senti_rate(self, news_item: dict[str, int | str]) -> dict[str, Any]:
        """Perform sentiment rating on 'news_item'.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        # Generate payload for chat completion
        payload = self.gen_payload(news_item)

        counter = 0
        while counter < 3:
            try:
                # Get sentiment rating and reasons
                response = self.llm.create_chat_completion(**payload)
                content = response["choices"][0]["message"]["content"]

                print(f"\nresponse : \n\n{pformat(response)}\n")

                # Extract json response which is a list of dictionary
                content = utils.extract_json_response(content)

                # Convert string into list of dictionaries
                payload = self.gen_response(content)

                # 'payload' contains only 1 item
                return payload[0]

            except Exception as e:
                counter += 1
                print(f"Attempts to rate sentiment : {counter}")
                print(e)

                # Wait 3 seconds to attempt again
                time.sleep(10)

        return {}

    @TimedMethod
    def batch_senti_rate(
        self, news_batch: list[dict[str, int | str]]
    ) -> list[dict[str, Any]]:
        """Perform sentiment on a batch of news items.

        Args:
            news_batch (list[dict[str, int | str]]):
                Batch of news item which is a dictionary containing 'id', 'ticker',
                and 'news'.
        Returns:
            (list[dict[str, Any]]):
                List of ratings i.e. dictionary containing 'id', 'rating'
                and 'reasons'.
        """

        # Generate payload for chat completion
        payload = self.gen_payload(news_batch)

        counter = 0
        while counter < 3:
            try:
                # Get sentiment rating and reasons
                response = self.llm.create_chat_completion(**payload)
                content = response["choices"][0]["message"]["content"]

                print(f"\nresponse : \n\n{pformat(response)}\n")

                # Extract json response which is a list of dictionary
                content = utils.extract_json_response(content)

                # Convert string into list of dictionaries
                return ast.literal_eval(content)

            except Exception as e:
                counter += 1
                print(f"Attempts to rate sentiment : {counter}")
                print(e)

                # Wait 3 seconds to attempt again
                time.sleep(3)

        return []

    def gen_response(self, content: str) -> dict[str, int | str]:
        """Ensure llm output is a dictionary and not list of dictionary.

        Args:
            content (str): String output by local LLM.

        Returns:
            (dict[str, int | str]): Dictionary containing 'id', 'rating' and 'reasons'.
        """

        # Evaluate text string
        payload = ast.literal_eval(content)

        if not isinstance(payload, (list, dict)):
            raise ValueError("payload is neither list or dictionary.")

        if isinstance(payload, dict):
            return [payload]

        return payload
