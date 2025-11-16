"""
Configuration objects and repository paths for the CTRL-Style project.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    """Base paths for the project repository."""
    base: Path

    @property
    def data_processed(self) -> Path:
        return self.base / "data" / "processed"

    @property
    def figures(self) -> Path:
        return self.base / "results" / "figures"

    @property
    def tables(self) -> Path:
        return self.base / "results" / "tables"


@dataclass(frozen=True)
class CorpusConfig:
    """
    Configuration for the literary corpus.

    Attributes
    ----------
    passages_csv : Path
        Path to the passages CSV file.
    text_col : str
        Name of the column containing text.
    author_col : str
        Name of the column containing author / style label.
    """
    passages_csv: Path
    text_col: str = "text"
    author_col: str = "author"
