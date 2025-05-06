"""Classes for various local llm download and inference."""

import ast
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pformat
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import ChatCompletionRequestResponseFormat, Llama

repo_dir = Path(__file__).parents[1].as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.prompt_template import sys_prompt, user_prompt
from src.senti_rater import SentiRating
from src.utils import utils
from src.utils.timed_method import TimedMethod


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
        _timings (list[float]):
            List to store time taken to 'senti_rate' method.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
        verbose: bool,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.timings = []

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


class LlamaLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class for 'Gemma' models."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(model_path, n_ctx, n_threads, n_gpu_layers, verbose)

    @TimedMethod
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
            verbose=self.verbose,
        )

        # Generate payload for chat completion
        payload = self.gen_payload(news_item)

        # Get sentiment rating and reasons
        response = llm.create_chat_completion(**payload)
        content = response["choices"][0]["message"]["content"]

        # Remove message fences and white spaces
        content = re.sub(r"```(json)*|\n\s*", "", content)

        print(f"\nresponse : \n\n{response}\n")

        # Ensure dictionary is returned
        return ast.literal_eval(content)

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


class MistralLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class for 'Mistral' models.

    Args:
        chat_format (str):
            Chat format template required if 'chat_template' is unavailable in
            HuggingFace (Default: "mistral-instruct").
        stop (list[str]):
            List of stop characters to indicate end of prompt (Default:
            ["</s>", "[INST]", "[/INST]"]).

    Attributes:
        chat_format (str):
            Chat format template required if 'chat_template' is unavailable in
            HuggingFace (Default: "mistral-instruct").
        stop (list[str]):
            List of stop characters to indicate end of prompt (Default:
            ["</s>", "[INST]", "[/INST]"]).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        chat_format: str = "mistral-instruct",
        stop: list[str] = ["</s>", "[INST]", "[/INST]"],
    ) -> None:
        super().__init__(model_path, n_ctx, n_threads, n_gpu_layers, verbose)
        self.chat_format = chat_format
        self.stop = stop

    @TimedMethod
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
            verbose=self.verbose,
            chat_format=self.chat_format,
        )

        # Generate payload for chat completion
        payload = self.gen_payload(news_item)

        counter = 0
        while counter < 3:
            try:
                # Get sentiment rating and reasons
                response = llm.create_chat_completion(**payload)
                content = response["choices"][0]["message"]["content"]

                # Remove message fences and white spaces
                content = re.sub(r"```(json)*|\n\s*", "", content)

                # Ensure keys are wrapped in double quotes and not single quotes

                print(f"\nresponse : \n\n{pformat(response)}\n")

                # Ensure dictionary is returned
                return ast.literal_eval(content)

            except Exception as e:
                counter += 1
                print(f"Attempts to rate sentiment : {counter}")

                # Wait 3 seconds to attempt again
                time.sleep(3)

        return {}

    def gen_payload(self, news_item: dict[str, int | str]) -> dict[str, Any]:
        """Generate payload for local llm chat completion.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            payload (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        usr_prompt = user_prompt.format(news_item=news_item)

        payload = {
            "messages": [{"role": "user", "content": f"{sys_prompt}\n\n{usr_prompt}"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": SentiRating.model_json_schema()},
            },
            "temperature": 0.1,
            "stop": self.stop,
        }

        return payload


class QwenLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class for 'QwenLLM' models.

    - Used No thinking mode as structured output is required.

    Args:
        chat_format (str):
            Chat format template required if 'chat_template' is unavailable in
            HuggingFace (Default: "mistral-instruct")..
        temperature (float):
            Temperature to use for sampling.
        top_p (float):
            Top-p value to use for nucleus sampling.
        top_k (float):
            Top-k value to use for sampling.
        min_p (float):
            Min-p value to use for minimum p sampling.
        max_tokens (int):
            Maximum number of tokens to generate.

    Attributes:
        chat_format (str):
            Chat format template required if 'chat_template' is unavailable in
            HuggingFace (Default: "mistral-instruct")..
        temperature (float):
            Temperature to use for sampling.
        top_p (float):
            Top-p value to use for nucleus sampling.
        top_k (float):
            Top-k value to use for sampling.
        min_p (float):
            Min-p value to use for minimum p sampling.
        max_tokens (int):
            Maximum number of tokens to generate.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        chat_format: str = "chatml",
        rope_freq_base: float = 1000000.0,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(model_path, n_ctx, n_threads, n_gpu_layers, verbose)
        self.chat_format = chat_format
        self.rope_freq_base = rope_freq_base
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens

    @TimedMethod
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
            verbose=self.verbose,
            chat_format=self.chat_format,
            rope_freq_base=self.rope_freq_base,
        )

        # Generate payload for chat completion
        payload = self.gen_payload(news_item)

        counter = 0
        while counter < 3:
            try:
                # Get sentiment rating and reasons
                response = llm.create_chat_completion(**payload)
                content = response["choices"][0]["message"]["content"]

                # Remove thinking quotes i.e. <think> ... </think>
                content = utils.remove_think(content)

                # Remove message fences and white spaces
                content = re.sub(r"```(json)*|\n\s*", "", content)

                print(f"\nresponse : \n\n{pformat(response)}\n")

                # Ensure dictionary is returned
                return ast.literal_eval(content)

            except Exception as e:
                counter += 1
                print(f"Attempts to rate sentiment : {counter}")

                # Wait 3 seconds to attempt again
                time.sleep(3)

        return {}

    def gen_payload(self, news_item: dict[str, int | str]) -> dict[str, Any]:
        """Generate payload for local llm chat completion.

        - Ensure '/no_think' is appended at end of prompt to disable thinking mode.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            payload (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        usr_prompt = user_prompt.format(news_item=news_item)

        payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"{usr_prompt}\n/no_think"},
            ],
            "response_format": {
                "type": "json_schema",
                "schema": SentiRating.model_json_schema(),
            },
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
        }

        return payload


class DeepSeekLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class for 'DeepSeek' models.

    Args:
        stop (list[str]):
            List of stop characters to indicate end of prompt (Default:
            ["< | User | > "]).
        temperature (float):
            Temperature to use for sampling.
        top_p (float):
            The top-p value to use for nucleus sampling.

    Attributes:
        stop (list[str]):
            List of stop characters to indicate end of prompt (Default:
            ["< | User | > "]).
        temperature (float):
            Temperature to use for sampling.
        top_p (float):
            The top-p value to use for nucleus sampling.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        stop: list[str] = ["< | User | >"],
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> None:
        super().__init__(model_path, n_ctx, n_threads, n_gpu_layers, verbose)
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p

    @TimedMethod
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
            verbose=self.verbose,
        )

        # Generate payload for chat completion
        payload = self.gen_payload(news_item)

        counter = 0
        while counter < 3:
            try:
                # Get sentiment rating and reasons
                response = llm.create_chat_completion(**payload)
                content = response["choices"][0]["message"]["content"]

                # Extract dictionary LLM response
                content = utils.extract_dict_response(content)

                print(f"\nresponse : \n\n{pformat(response)}\n")

                # Ensure dictionary is returned
                return ast.literal_eval(content)

            except Exception as e:
                counter += 1
                print(f"Attempts to rate sentiment : {counter}")

                # Wait 3 seconds to attempt again
                time.sleep(3)

        return {}

    def gen_payload(self, news_item: dict[str, int | str]) -> dict[str, Any]:
        """Generate payload for local llm chat completion.

        - System prompt is combined with user prompt as per usage recommendations
        in https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF
        - 'top_k = 0.95' is based on GGUF-specific recommendations.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            payload (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        usr_prompt = user_prompt.format(news_item=news_item)
        prompt = f"< | User | >{sys_prompt}\n\n{usr_prompt}< | Assistant | >"

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": SentiRating.model_json_schema()},
            },
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
        }

        return payload
