"""Classes for various local llm download and inference."""

import time
from pprint import pformat
from typing import Any

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.local_llm.base import InferLLM
from src.local_llm.timed_method import TimedMethod
from src.senti_rater import SentiRating
from src.utils import utils


class LlamaLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class for 'Gemma', 'Llama'
    or 'Phi' models."""

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
        self.llm = self.gen_llm()

    def gen_llm(self) -> Llama:
        """Generate initialized instance of Llama for specific local llama or
        gemma LLM."""

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
        """Generate payload for local llama or gemma llm chat completion.

        Args:
            news (dict[str, int  |  str] | list[dict[str, int | str]]):
                List of dictionary or dictionary containing 'id', 'ticker'
                and 'news' info.

        Returns:
            (dict[str, int | str]):
                Dictionary containing input parameters for 'create_chat_completion'
                method.
        """

        # Get initialized system and user prompts
        sys_p, usr_p = utils.init_sys_usr_prompt(news)

        return {
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            "response_format": {
                "type": "json_object",
                "schema": SentiRating.model_json_schema(),
            },
            "temperature": self.temperature,
            "grammar": self.json_grammar,
        }


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
        temperature: float = 0.1,
        chat_format: str = "mistral-instruct",
        stop: list[str] = ["</s>", "[INST]", "[/INST]"],
    ) -> None:
        super().__init__(
            model_path, n_ctx, n_threads, n_gpu_layers, verbose, temperature
        )
        self.chat_format = chat_format
        self.stop = stop
        self.llm = self.gen_llm()

    def gen_llm(self) -> Llama:
        """Generate initialized instance of Llama for specific local mistral LLM."""

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
        """Generate payload for local mistral llm chat completion.

        Args:
            news (dict[str, int  |  str] | list[dict[str, int | str]]):
                List of dictionary or dictionary containing 'id', 'ticker'
                and 'news' info.

        Returns:
            (dict[str, int | str]):
                Dictionary containing input parameters for 'create_chat_completion'
                method.
        """

        # Get initialized system and user prompts
        sys_p, usr_p = utils.init_sys_usr_prompt(news)

        return {
            "messages": [{"role": "user", "content": f"{sys_p}\n\n{usr_p}"}],
            "response_format": {
                "type": "json_object",
                "schema": SentiRating.model_json_schema(),
            },
            "temperature": self.temperature,
            "stop": self.stop,
            "grammar": self.json_grammar,
        }


class QwenLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class for qwen models.

    - Used No thinking mode as structured output is required.

    Args:
        chat_format (str):
            Chat format template required if 'chat_template' is unavailable in
            HuggingFace (Default: "mistral-instruct").
        rope_freq_base (float):
            RoPE frequency scaling factor (Default: 1000000.0).
        top_p (float):
            Top-p value to use for nucleus sampling (Default: 0.8).
        top_k (float):
            Top-k value to use for sampling (Default: 20).
        min_p (float):
            Min-p value to use for minimum p sampling (Default: 0).
        max_tokens (int):
            Maximum number of tokens to generate (Default: 512).
        repeat_penalty (float):
            Penalty to apply for repeated tokens. 1.0 = No penalty. Not more than 1.2
            (Default: 1.0).
        is_qwq (bool):
            Whether using QwQ model (Default: False).

    Attributes:
        chat_format (str):
            Chat format template required if 'chat_template' is unavailable in
            HuggingFace (Default: "mistral-instruct").
        rope_freq_base (float):
            RoPE frequency scaling factor (Default: 1000000.0).
        top_p (float):
            Top-p value to use for nucleus sampling (Default: 0.8).
        top_k (float):
            Top-k value to use for sampling (Default: 20).
        min_p (float):
            Min-p value to use for minimum p sampling (Default: 0).
        max_tokens (int):
            Maximum number of tokens to generate (Default: 512).
        repeat_penalty (float):
            Penalty to apply for repeated tokens. 1.0 = No penalty. Not more than 1.2
            (Default: 1.0).
        is_qwq (bool):
            Whether using QwQ model (Default: False).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        temperature: float = 0.7,
        chat_format: str = "chatml",
        rope_freq_base: float = 1000000.0,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0,
        max_tokens: int = 512,
        repeat_penalty: float = 1.0,
        is_qwq: bool = False,
    ) -> None:
        super().__init__(
            model_path, n_ctx, n_threads, n_gpu_layers, verbose, temperature
        )
        self.chat_format = chat_format
        self.rope_freq_base = rope_freq_base
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.is_qwq = is_qwq
        self.llm = self.gen_llm()

    def gen_llm(self) -> Llama:
        """Generate initialized instance of Llama for specific local qwen LLM."""

        return Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            chat_format=self.chat_format,
            rope_freq_base=self.rope_freq_base,
        )

    def gen_payload(
        self, news: dict[str, int | str] | list[dict[str, int | str]]
    ) -> dict[str, Any]:
        """Generate payload for local qwen llm chat completion.

        Args:
            news (dict[str, int  |  str] | list[dict[str, int | str]]):
                List of dictionary or dictionary containing 'id', 'ticker'
                and 'news' info.

        Returns:
            (dict[str, int | str]):
                Dictionary containing input parameters for 'create_chat_completion'
                method.
        """

        # Get initialized system and user prompts
        sys_p, usr_p = utils.init_sys_usr_prompt(news)

        # Append '/no_think' to end of user prompt to disable thinking
        prompt = usr_p if self.is_qwq else f"{usr_p}\n/no_think"

        return {
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_object",
                "schema": SentiRating.model_json_schema(),
            },
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "grammar": self.json_grammar,
        }


class DeepSeekLLM(InferLLM):
    """Concrete implementation of 'InferLLM' abstract class for 'DeepSeek' models.

    Args:
        stop (list[str]):
            List of stop characters to indicate end of prompt (Default:
            ["< | User | > "]).
        top_p (float):
            The top-p value to use for nucleus sampling.

    Attributes:
        stop (list[str]):
            List of stop characters to indicate end of prompt (Default:
            ["< | User | > "]).
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
        temperature: float = 0.6,
        stop: list[str] = ["< | User | >"],
        top_p: float = 0.95,
    ) -> None:
        super().__init__(
            model_path, n_ctx, n_threads, n_gpu_layers, verbose, temperature
        )
        self.stop = stop
        self.top_p = top_p
        self.llm = self.gen_llm()

    def gen_llm(self) -> Llama:
        """Generate initialized instance of Llama for specific local deepseek LLM."""

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
        """Generate payload for local llm deepseek chat completion.

        - System prompt is combined with user prompt as per usage recommendations
        in https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF
        - 'top_k = 0.95' is based on GGUF-specific recommendations.

        Args:
            news (dict[str, int  |  str] | list[dict[str, int | str]]):
                List of dictionary or dictionary containing 'id', 'ticker'
                and 'news' info.

        Returns:
            (dict[str, int | str]):
                Dictionary containing input parameters for 'create_chat_completion'
                method.
        """

        # Get initialized system and user prompts
        sys_p, usr_p = utils.init_sys_usr_prompt(news)

        # Get customized prompt specifically for Deepseek model
        prompt = f"< | User | >{sys_p}\n\n{usr_p}< | Assistant | >"

        return {
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_object",
                "schema": SentiRating.model_json_schema(),
            },
            "temperature": self.temperature,
            "stop": self.stop,
            "top_p": self.top_p,
            "grammar": self.json_grammar,
        }
