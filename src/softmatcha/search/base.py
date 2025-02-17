from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generator

import numpy as np
import numpy.typing as npt

from softmatcha import stopwatch
import softmatcha.functional as F
from softmatcha.embeddings import Embedding
from softmatcha.struct import Pattern, TokenEmbeddings
from softmatcha.struct.index import Index
from softmatcha.tokenizers import Tokenizer
from softmatcha.typing import NDArrayF32, NDArrayI32


class Search(abc.ABC):
    """Search base class to find the given pattern from a text."""

    tokenizer: Tokenizer
    embedding: Embedding

    @dataclass
    class Match:
        """Match object.

        begin (int): Begin position of a matched span.
        end (int): End position of a matched span.
        scores (NDArrayF32): Match scores of shape `(pattern_len,)`.
        tokens (list[int]): Matched tokens.
        """

        begin: int
        end: int
        scores: NDArrayF32
        tokens: list[int]

    def compute_exact_match(self, pattern: Pattern, embedding: Embedding) -> NDArrayF32:
        """Compute the similarity between pattern and vocabulary.

        Args:
            pattern (Pattern): Pattern tokens and their embeddings of shape `(P, D)`.
            embedding (Embedding): Embedding.

        Returns:
            NDArrayF32: Match matrix of shape `(P, V)`,
              where the matched element is set to 1, otherwise 0.
        """
        scores = np.zeros((len(pattern), len(embedding)), dtype=np.float32)
        with stopwatch.timers["similarity"]:
            for i, p in enumerate(pattern.tokens):
                scores[i, p] = 1.0
        return scores

    def compute_similarity(
        self,
        pattern: Pattern,
        embedding: Embedding,
        vocabulary: NDArrayI32 | None = None,
    ) -> NDArrayF32:
        """Compute the similarity between pattern and vocabulary.

        Args:
            pattern (Pattern): Pattern tokens and their embeddings of shape `(P, D)`.
            embedding (Embedding): Embedding.
            vocabulary (NDArrayI32, optional): Subset of vocabulary for which similarities are calculated.

        Returns:
            NDArrayF32: Similarity matrix of shape `(P, V)`.
        """
        scores = np.zeros((len(pattern), len(embedding)), dtype=np.float32)
        with stopwatch.timers["similarity"]:
            if vocabulary is None:
                scores = F.matmul(
                    pattern.embeddings, embedding.embeddings
                )
            else:
                scores[:, vocabulary] = F.matmul(
                    pattern.embeddings, embedding.embeddings[:, vocabulary]
                )

            scores[:, self.tokenizer.unk_idx] = 0.0
            for i, p in enumerate(pattern.tokens):
                if p == self.tokenizer.unk_idx:
                    scores[i] = 0.0
        return scores


class SearchScan(Search, metaclass=abc.ABCMeta):
    """SearchScan class to find the given pattern from a text.

    Args:
        pattern (Pattern): Pattern token embeddings.
        tokenizer (Tokenizer): Tokenizer.
        embedding (Embedding): Embeddings.
    """

    def __init__(
        self, pattern: Pattern, tokenizer: Tokenizer, embedding: Embedding
    ) -> None:
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.scores = self.compute_similarity(pattern, embedding)
        self.is_match = self.scores >= pattern.thresholds[:, None]

    @abc.abstractmethod
    def search(self, text: TokenEmbeddings, start: int = 0) -> Generator[Search.Match]:
        """Search for the pattern from the given text.

        Args:
            text (TokenEmbeddings): Text token embeddings to be searched.
            start (int): Start position to be searched.

        Yields:
            Match: Yield the Match object when a subsequence of text matches the pattern.
        """


class SearchIndex(Search, metaclass=abc.ABCMeta):
    """SearchIndex class to find the given pattern from a text using the index.

    Args:
        index (Index): An index to be used for searching the text quickly.
        tokenizer (Tokenizer): Tokenizer.
        embedding (Embedding): Embeddings.
    """

    def __init__(
        self, index: Index, tokenizer: Tokenizer, embedding: Embedding
    ) -> None:
        self.index = index
        self.tokenizer = tokenizer
        self.embedding = embedding

    @abc.abstractmethod
    def search(self, pattern: Pattern, start: int = 0) -> Generator[Search.Match]:
        """Search for the pattern from the given text.

        Args:
            pattern (Pattern): Pattern token embeddings.
            start (int): Start position to be searched.

        Yields:
            Match: Yield the Match object when a subsequence of text matches the pattern.
        """
