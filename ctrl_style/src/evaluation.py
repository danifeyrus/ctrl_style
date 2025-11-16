"""
Evaluation utilities for style transfer in CTRL-Style.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from sentence_transformers import SentenceTransformer


@dataclass
class StyleEvalConfig:
    """
    Configuration for style-transfer evaluation.
    """
    model_name: str = "all-MiniLM-L6-v2"


class StyleTransferEvaluator:
    """
    Compute similarity and readability metrics for style-transfer pairs.
    """

    def __init__(self, config: StyleEvalConfig | None = None) -> None:
        self.config = config or StyleEvalConfig()
        self._model = SentenceTransformer(self.config.model_name)

    def _cosine_pairs(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        sims = []
        for v1, v2 in zip(emb1, emb2):
            sims.append(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])
        return np.asarray(sims)

    def evaluate_pairs(self, df_pairs: pd.DataFrame) -> pd.DataFrame:
        """
        Expect a DataFrame with columns 'original' and 'rewritten'.
        Returns the same DataFrame with cosine and FK metrics added.
        """
        orig = df_pairs["original"].tolist()
        rew = df_pairs["rewritten"].tolist()

        emb_orig = self._model.encode(orig, convert_to_numpy=True, show_progress_bar=True)
        emb_rew = self._model.encode(rew, convert_to_numpy=True, show_progress_bar=True)

        cosine = self._cosine_pairs(emb_orig, emb_rew)
        fk_orig = np.array([textstat.flesch_kincaid_grade(t) for t in orig])
        fk_rew = np.array([textstat.flesch_kincaid_grade(t) for t in rew])

        out = df_pairs.copy()
        out["cosine"] = cosine
        out["fk_orig"] = fk_orig
        out["fk_rew"] = fk_rew
        out["delta_fk"] = fk_rew - fk_orig
        return out
