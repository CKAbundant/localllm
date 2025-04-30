"""Perform sentiment rating on news article (title and content) from 1 to 5:

- Use template provided in https://github.com/ppl-ai/api-cookbook/blob/main/sonar-use-cases/fact_checker_cli/fact_checker.py
- Allow batch processing to limit number of API calls
- Use 'sonar' model based on pricing guide: https://docs.perplexity.ai/guides/pricing
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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

    # test = pd.read_csv("./data/test_new.csv")
    # info = test.loc[0, ["id", "ticker", "title", "content"]]
    # news_item = {
    #     "id": info["id"],
    #     "ticker": info["ticker"],
    #     "news": f"{info["title"]}\n\n{info["content"]}",
    # }

    # Initalize concrete implementation of 'InferLLM'
    gemma_llm = GemmaLLM()

    # Generate DataFrame after sentiment analysis on 'news_list'
    output = senti_rate(gemma_llm, "gemma")

    print(f"output : \n\n{output}\n")


def senti_rate(
    local_llm: InferLLM, model_name: str, test_path: str = "./data/test.csv"
) -> pd.DataFrame:
    """Perform sentiment analysis on list of news items.

    Args:
        local_llm (InferLLM):
            Initialized concrete implementatino of 'InferLLM' abstract class.
        model_name (str):
            Name of local llm.
        test_path (str):
            Relative path to test dataset (Default: "./data/test.csv).

    Returns:
        (pd.DataFrame):
            DataFrame containing sentiment ratings and reasons for all
            news items in 'news_list'.
    """

    if not Path(test_path).is_file():
        raise FileNotFoundError(f"'test.csv' not found at {test_path}")

    # Load 'test.csv' as DataFrame
    df_test = pd.read_csv(test_path)

    # Generate news_list
    news_list = gen_news_list(df_test)

    # Generate list of sentiment ratings and reasons based on 'news_list'
    rating_list = [local_llm.senti_rate(news_item) for news_item in tqdm(news_list)]

    # Convert to 'rating_list' to DataFrame
    df_ratings = pd.DataFrame(rating_list)

    # Merge df_test with rating_list
    df_filter = df_test.loc[
        :, ["id", "pub_date", "ticker", "title", "content", "rating", "reasons"]
    ]
    df_senti = df_filter.merge(
        df_ratings, how="left", on="id", suffixes=(None, f"_{model_name}")
    )
    df_senti.to_csv(Path(test_path).with_name("results.csv"))

    return df_senti


def gen_news_list(df_news: pd.DataFrame) -> list[dict[str, int | str]]:
    """Generate list of news items i.e. dictionary containing 'id', 'ticker'
    # and 'content' keys."""

    df = df_news.copy()

    # Combine 'title' and 'content' to 'news' column
    df["news"] = df["title"] + "\n\n" + df["content"]

    # Filter 'id', 'ticker' and 'news' columns
    df = df.loc[:, ["id", "ticker", "news"]]

    # Convert to list of dictionary
    return df.to_dict(orient="records")


if __name__ == "__main__":
    main()
