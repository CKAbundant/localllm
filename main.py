"""Perform sentiment rating on news article (title and content) from 1 to 5:

- Use template provided in https://github.com/ppl-ai/api-cookbook/blob/main/sonar-use-cases/fact_checker_cli/fact_checker.py
- Allow batch processing to limit number of API calls
- Use 'sonar' model based on pricing guide: https://docs.perplexity.ai/guides/pricing
"""

import json
import os
import sys
import time
from pathlib import Path
from pprint import pformat
from typing import Annotated, Literal, TypeAlias

from tqdm import tqdm

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, StringConstraints, field_validator

from src.prompt_template import sys_prompt, user_prompt
from src.utils import utils

# Create literal for Perplexity model that allows structured output
StructModel: TypeAlias = Literal[
    "sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"
]


class SentiRating(BaseModel):
    """Structured response for sentiment rating of stock-related news."""

    id: int = Field(description="ID for each news article")
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
        news_path (str):
            Relative path to csv file containing news articles with divergent
            sentiment rating (Default: "./data/divergent.csv").

    Attributes:
        api_url (str):
            URL to Perplexity API
            (Default: "https://api.perplexity.ai/chat/completions").
        model (str):
            Perplexity model to be used (Default: "sonar").
        news_path (str):
            Relative path to csv file containing news articles with divergent
            sentiment rating.
        api_key (str):
            Perplexity api key.
    """

    def __init__(
        self,
        api_url: str = "https://api.perplexity.ai/chat/completions",
        model: str = "sonar",
        news_path: str = "./data/divergent.csv",
    ):
        self.api_url = api_url
        self.model = model
        self.news_path = news_path
        self.api_key = self._get_api_key()

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

        # load 'divergent.csv' as DataFrame
        df_news = pd.read_csv(self.news_path)

        # Generate 'id' column from index
        df_news.insert(0, "id", df_news.index)

        # Generate list of news items
        news_list = self.gen_news_list(df_news)

        # # Get response json after posting payload to Perplexity API
        # rating_list = []

        # for idx, news_item in tqdm(enumerate(news_list)):
        #     rating_list.append(self.get_response(news_item))

        #     if idx != 0 and idx % 50 == 0:
        #         # Pause for 45 seconds for every 50 calls to prevent rate limit
        #         time.sleep(45)

        rating_list = utils.load_json("./rating_list.json")

        # Convert 'rating_list' to DataFrame
        df_rating = pd.DataFrame(rating_list)

        # Merge 'df_news' and 'df_rating'
        df_combined = df_news.merge(right=df_rating, how="left", on="id")
        df_combined.to_csv("./data/test.csv", index=False)

        return df_combined

    def get_response(self, news_item: dict[str, int | str]) -> dict[str, int | str]:
        """Get response after posting payload to Perplexity API.

        Args:
            news_item (dict[str, int  |  str]):
                Dictionary containing 'id', 'ticker' and 'news' info.

        Returns:
            (dict[str, int | str]):
                Dictionary containing 'id', 'rating' and 'reasons' info.
        """

        # Set headers for API request
        headers = {
            "accept": "application/json",
            "content_type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Payload to be send to Perplexity API
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt.format(news_item=news_item)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": SentiRating.model_json_schema()},
            },
            "web_search_options": {"search_context_size": "low"},
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # content is a json string
            content = result["choices"][0]["message"]["content"]

            return json.loads(content)

        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {e}"}

        except json.JSONDecodeError:
            return {"error": "Failed to parse API response as JSON."}

        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def gen_news_list(self, df_news: pd.DataFrame) -> list[dict[str, int | str]]:
        """Generate list of news items i.e. dictionary containing 'id', 'ticker'
        # and 'content' keys."""

        df = df_news.copy()

        # Combine 'title' and 'content' to 'news' column
        df["news"] = df["title"] + "\n\n" + df["content"]

        # Filter 'id', 'ticker' and 'news' columns
        df = df.loc[:, ["id", "ticker", "news"]]

        # Convert to list of dictionary
        return df.to_dict(orient="records")


def main() -> None:
    senti_rater = SentiRater()
    df_combined = senti_rater.run()

    print(f"df_combined : \n\n{df_combined}\n")
    print(f"df_combined.columns : {df_combined.columns}")


if __name__ == "__main__":
    main()
