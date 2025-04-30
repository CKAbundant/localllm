"""Perform sentiment rating on news article (title and content) from 1 to 5:

- Use template provided in https://github.com/ppl-ai/api-cookbook/blob/main/sonar-use-cases/fact_checker_cli/fact_checker.py
- Allow batch processing to limit number of API calls
- Use 'sonar' model based on pricing guide: https://docs.perplexity.ai/guides/pricing
"""

import sys
from pathlib import Path

import pandas as pd

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.local_llm import GemmaLLM, InferLLM
from src.senti_rater import SentiRater


def main() -> None:
    # senti_rater = SentiRater()
    # df_combined, df_tokens = senti_rater.run()

    # print(f"df_combined : \n\n{df_combined}\n")
    # print(f"df_combined.columns : {df_combined.columns}")

    # print(f"df_tokens : \n\n{df_tokens}\n")
    # print(f"df_tokens.columns : {df_tokens.columns}")
    # print(f"\ndf_tokens.describe() : \n\n{df_tokens.describe()}\n")

    test = pd.read_csv("./data/test_new.csv")
    info = test.loc[0, ["id", "ticker", "title", "content"]]
    news_item = {
        "id": info["id"],
        "ticker": info["ticker"],
        "news": f"{info["title"]}\n\n{info["content"]}",
    }

    gemma_llm = GemmaLLM()
    output = senti_rate(gemma_llm, news_item)

    print(f"output : \n\n{output}\n")


def senti_rate(
    local_llm: InferLLM, news_item: dict[str, int | str]
) -> dict[str, int | str]:
    """Generate sentiment rating based on selected local llm."""

    return local_llm.senti_rate(news_item)


if __name__ == "__main__":
    main()
