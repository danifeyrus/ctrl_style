"""
Sentence embedding model interface for CTRL-Style.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingModelConfig:
    """
    Configuration for the embedding model.

    Attributes
    ----------
    model_name : str
        Sentence-Transformers model identifier.
    batch_size : int
        Batch size used during encoding.
    """
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


class EmbeddingModel:
    """
    High-level wrapper around SentenceTransformer for style analysis.
    """

    def __init__(self, config: EmbeddingModelConfig | None = None) -> None:
        self.config = config or EmbeddingModelConfig()
        self._model = SentenceTransformer(self.config.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into sentence embeddings.
        """
        emb = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return emb

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        emb_col: str = "emb",
    ) -> pd.DataFrame:
        """
        Encode the `text_col` of a DataFrame and attach embeddings as `emb_col`.
        """
        emb = self.encode(df[text_col].tolist())
        out = df.copy()
        out[emb_col] = list(emb)
        return out
