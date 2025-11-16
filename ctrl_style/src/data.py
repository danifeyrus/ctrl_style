"""
Data loading and corpus management for CTRL-Style.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .config import CorpusConfig


@dataclass
class CorpusDataset:
    """
    Dataset wrapper for literary passages.

    Parameters
    ----------
    config : CorpusConfig
        Configuration with file path and column names.
    """
    config: CorpusConfig

    def load_all(self) -> pd.DataFrame:
        """Load the full corpus as a pandas DataFrame."""
        df = pd.read_csv(self.config.passages_csv)
        df = df[[self.config.text_col, self.config.author_col]].dropna()
        df = df.rename(
            columns={
                self.config.text_col: "text",
                self.config.author_col: "author",
            }
        )
        return df.reset_index(drop=True)

    def filter_authors(self, df: pd.DataFrame, authors: Sequence[str]) -> pd.DataFrame:
        """Keep only passages written by the given authors/styles."""
        return df[df["author"].isin(authors)].reset_index(drop=True)

    def sample_per_author(
        self,
        df: pd.DataFrame,
        n_per_author: int = 100,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Sample up to `n_per_author` passages for each author.

        Returns
        -------
        pd.DataFrame
            Balanced subset of the corpus.
        """
        parts = []
        for author in df["author"].unique():
            sub = df[df["author"] == author]
            k = min(n_per_author, len(sub))
            parts.append(sub.sample(n=k, random_state=random_state))
        out = pd.concat(parts, ignore_index=True)
        return out
