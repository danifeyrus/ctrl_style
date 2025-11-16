"""
Dimensionality reduction and visualization utilities for CTRL-Style.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap


@dataclass
class UmapConfig:
    """
    Configuration for UMAP projection.
    """
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


class StyleEmbeddingVisualizer:
    """
    Helper class to project embeddings and visualize style clusters.
    """

    def __init__(self, config: UmapConfig | None = None) -> None:
        self.config = config or UmapConfig()

    def project_umap(self, X: np.ndarray) -> np.ndarray:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
        )
        return reducer.fit_transform(X)

    def scatter_by_author(
        self,
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        author_col: str = "author",
        title: str | None = None,
    ) -> None:
        """
        Plot a 2D scatter of embeddings, colored by author/style label.
        """
        for author in df[author_col].unique():
            sub = df[df[author_col] == author]
            plt.scatter(sub[x_col], sub[y_col], label=author, alpha=0.8)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if title:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
