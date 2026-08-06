#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class PromptEmbedder:
    """
    Reproducible prompt embedding module.

    Training:
      embedder.fit(train_prompt_texts)
      embedder.save("prompt_embedder.pkl")

    Inference:
      embedder = PromptEmbedder.load("prompt_embedder.pkl")
      z = embedder.transform([new_prompt_text])

    Important:
      - HashingVectorizer is deterministic.
      - SVD is fit once on training prompts only.
      - The saved SVD projection is reused during evaluation/inference.
    """

    def __init__(
        self,
        dim: int = 64,
        n_features: int = 2**18,
        ngram_range=(1, 2),
        lowercase: bool = True,
        seed: int = 42,
    ):
        self.dim = int(dim)
        self.n_features = int(n_features)
        self.ngram_range = tuple(ngram_range)
        self.lowercase = bool(lowercase)
        self.seed = int(seed)

        self.vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
        )

        self.svd = None
        self.real_dim = None
        self.fit_n_texts = None

    @staticmethod
    def canonicalize(text: str) -> str:
        if text is None:
            return ""
        text = str(text)
        text = text.strip()
        text = " ".join(text.split())
        return text

    def fit(self, texts: Iterable[str]):
        texts = [self.canonicalize(t) for t in texts]
        self.fit_n_texts = len(texts)

        if self.fit_n_texts < 2:
            raise ValueError("Need at least two training prompts to fit PromptEmbedder.")

        X = self.vectorizer.transform(texts)
        self.real_dim = min(self.dim, max(1, self.fit_n_texts - 1), X.shape[1] - 1)

        self.svd = TruncatedSVD(
            n_components=self.real_dim,
            random_state=self.seed,
        )
        self.svd.fit(X)
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("PromptEmbedder must be fit before transform().")

        texts = [self.canonicalize(t) for t in texts]
        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X)
        Z = normalize(Z).astype(np.float32)

        if self.real_dim < self.dim:
            pad = np.zeros((Z.shape[0], self.dim - self.real_dim), dtype=np.float32)
            Z = np.concatenate([Z, pad], axis=1)

        return Z.astype(np.float32)

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def metadata(self):
        return {
            "dim": self.dim,
            "n_features": self.n_features,
            "ngram_range": list(self.ngram_range),
            "lowercase": self.lowercase,
            "seed": self.seed,
            "real_dim": self.real_dim,
            "fit_n_texts": self.fit_n_texts,
            "type": "HashingVectorizer+TruncatedSVD+L2Normalize",
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f)

        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(self.metadata(), indent=2))

    @staticmethod
    def load(path: str | Path):
        with Path(path).open("rb") as f:
            return pickle.load(f)
