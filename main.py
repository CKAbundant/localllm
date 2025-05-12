"""Perform sentiment rating on news article (title and content) from 1 to 5:

- Use template provided in https://github.com/ppl-ai/api-cookbook/blob/main/sonar-use-cases/fact_checker_cli/fact_checker.py
- Allow batch processing to limit number of API calls
- Use 'sonar' model based on pricing guide: https://docs.perplexity.ai/guides/pricing
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.gen_analysis import GenAnalysis
from src.local_llm import InferLLM, download_hf
from src.senti_rater import SentiRater
from src.utils import utils


def main() -> None:
    # Load environment variables from '.env' file
    load_dotenv()

    # Load configuration
    cfg = utils.load_config()

    # Get parameters specific to selected model
    cfg_model = cfg[cfg.model]

    if cfg.gen_test_data:
        senti_rater = SentiRater()
        df_combined, df_tokens = senti_rater.run()

        print(f"df_combined : \n\n{df_combined}\n")
        print(f"df_combined.columns : {df_combined.columns}")

        print(f"df_tokens : \n\n{df_tokens}\n")
        print(f"df_tokens.columns : {df_tokens.columns}")
        print(f"\ndf_tokens.describe() : \n\n{df_tokens.describe()}\n")

    # Relative path to desired model in GGUF format
    gguf_path = cfg_model.infer.model_path

    if cfg.download and not Path(gguf_path).is_file():
        download_hf(**cfg_model.download)

    if cfg.infer:
        # Get instance of desired Inference class
        infer_llm = utils.get_class_instance(**cfg_model.infer)
        print(f"cfg_model.infer.class_name : {cfg_model.infer.class_name}")

        # Generate DataFrame after sentiment analysis on 'news_list'
        gen_analysis = GenAnalysis(local_llm=infer_llm, **cfg.gen_analysis)
        df_senti, df_metrics = gen_analysis.run()

        print(f"df_senti : \n\n{df_senti}\n")
        print(f"df_metrics : \n\n{df_metrics}\n")


if __name__ == "__main__":
    main()
