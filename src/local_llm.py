"""Classes for various local llm download and inference."""

import ast
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pformat
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

repo_dir = Path(__file__).parents[1].as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.prompt_template import (
    batch_sys_prompt,
    batch_user_prompt,
    sys_prompt,
    user_prompt,
)
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
        self.llm = self.gen_llm()

    @abstractmethod
    def gen_llm(self) -> Llama:
        """Generate initialized instance of Llama for specific local LLM."""

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
                payload = ast.literal_eval(content)

                # 'payload' contains only 1 item
                return payload[0]

            except Exception as e:
                counter += 1
                print(f"Attempts to rate sentiment : {counter}")

                # Wait 3 seconds to attempt again
                time.sleep(3)

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

                # Wait 3 seconds to attempt again
                time.sleep(3)

        return []


class LlamaLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class for 'Gemma' models."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(
            model_path, n_ctx, n_threads, n_gpu_layers, verbose, temperature
        )

    def gen_llm(self) -> Llama:
        """Generate initialized instance of Llama for specific local LLM."""

        return Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )

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
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        if isinstance(news, list):
            sys_p = batch_sys_prompt
            usr_p = batch_user_prompt.format(news_list=news)

        else:
            sys_p = sys_prompt
            usr_p = user_prompt.format(news_item=news)

        payload = {
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            "response_format": {
                "type": "json_schema",
                "schema": SentiRating.model_json_schema(),
            },
            "temperature": self.temperature,
            "grammar": self.json_grammar,
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
        temperature: float = 0.1,
    ) -> None:
        super().__init__(
            model_path, n_ctx, n_threads, n_gpu_layers, verbose, temperature
        )
        self.chat_format = chat_format
        self.stop = stop

    def gen_llm(self) -> Llama:
        """Generate initialized instance of Llama for specific local LLM."""

        return Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            chat_format=self.chat_format,
        )

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
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        if isinstance(news, list):
            sys_p = batch_sys_prompt
            usr_p = batch_user_prompt.format(news_list=news)

        else:
            sys_p = sys_prompt
            usr_p = user_prompt.format(news_item=news)

        payload = {
            "messages": [{"role": "user", "content": f"{sys_p}\n\n{usr_p}"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": SentiRating.model_json_schema()},
            },
            "temperature": self.temperature,
            "stop": self.stop,
            "grammar": self.json_grammar,
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

                print(f"\nresponse : \n\n{pformat(response)}\n")

                # Extract json response which is a list of dictionary
                content = utils.extract_json_response(content)

                # Convert string into list of dictionaries
                payload = ast.literal_eval(content)

                # 'payload' contains only 1 item
                return payload[0]

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
