"""Perform sentiment rating on news article (title and content) from 1 to 5:

- Use template provided in https://github.com/ppl-ai/api-cookbook/blob/main/sonar-use-cases/fact_checker_cli/fact_checker.py
- Allow batch processing to limit number of API calls
- Use 'sonar' model based on pricing guide: https://docs.perplexity.ai/guides/pricing
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, StringConstraints, field_validator

from src.prompt_template import sys_prompt, user_prompt

# Create literal for Perplexity model that allows structured output
StructModel: TypeAlias = Literal[
    "sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"
]


class SentiRating(BaseModel):
    """Structured response for sentiment rating of stock-related news."""

    id: int = Field(description="ID for each news article")
    ticker: str = Field(description="Stock ticker whose news are sentiment-rated.")
    rating: int = Field(
        description="Sentiment from 1 (negative) to 5 (positive).", gte=1, le=5
    )
    reasons: list[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, max_length=200),
        ]
    ] = Field(
        default_factory=list,
        description="list of reasons for proposed sentiment rating. Maximum 3 reasons. Each reason not more than 1 sentence.",
    )

    @field_validator("reasons")
    def max_three_reasons(cls, reasons: list[str]) -> list[str]:
        if len(reasons) == 0:
            raise ValueError("At least 1 reason required.")
        if len(reasons) > 3:
            raise ValueError("Maximum of 3 reasons allowed.")

        return reasons


class SentiRater:
    """Generate ground truth sentiment rating via Perplexity API for news
    articles that have divergent rating using FinBERT models.

    Args:
        api_url (str):
            URL to Perplexity API
            (Default: "https://api.perplexity.ai/chat/completions").
        model (str):
            Perplexity model to be used (Default: "sonar").
        struct_models (list[StructModel]):
            List of StructModel objects i.e. Perplexity model that supports
            structured outputs(Default: ["sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"]).
        news_path (str):
            Relative path to csv file containing news articles with divergent
            sentiment rating.

    Attributes:
        api_url (str):
            URL to Perplexity API
            (Default: "https://api.perplexity.ai/chat/completions").
        model (str):
            Perplexity model to be used (Default: "sonar").
        struct_models (list[StructModel]):
            List of StructModel objects i.e. Perplexity model that supports
            structured outputs(Default: ["sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"]).
        news_path (str):
            Relative path to csv file containing news articles with divergent
            sentiment rating.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "sonar",
        struct_models: list[StructModel] = [
            "sonar",
            "sonar-pro",
            "sonar-reasoning",
            "sonar-reasoning-pro",
        ],
        news_path: str = "./data/news.csv",
    ):
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.struct_models = struct_models
        self.news_path = news_path

    def _get_api_key(self) -> str:
        """Get API key from environment variables."""

        # Load environment variables from '.env' file
        load_dotenv()

        if (api_key := os.getenv("PPLX_API_KEY")) is None:
            raise ValueError("No api key provided!")

        return api_key

    def run(self) -> pd.DataFrame:
        """Perform sentiment rating for list of news items and return results as
        DataFrame.

        Args:
            None.

        Returns:
            (pd.DataFrame):
                DataFrame with appended sentiment rating and reasons.
        """

        # Generate list of news items
        news_list = self.gen_news_list()

        return news_list

    def gen_news_list(self) -> list[dict[str, int | str]]:
        """Generate list of news items i.e. dictionary containing 'id', 'ticker'
        # and 'content' keys."""

        # Load 'news.csv' as DataFrame
        df_news = pd.read_csv(self.news_path)

        # Extract 'id' and 'ticker', 'title' and 'content'


def main() -> None:
    senti_rater = SentiRater()
    df_senti = senti_rater.run()

    print(f"\n\n{df_senti}\n")
    print(df_senti.columns)


if __name__ == "__main__":
    main()
