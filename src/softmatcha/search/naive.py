from __future__ import annotations

from typing import Generator

import numpy as np

from softmatcha import stopwatch
from softmatcha.struct import TokenEmbeddings

from .base import Search, SearchScan


class SearchNaive(SearchScan):
    """SearchNaive naively finds the given pattern from a text.

    Args:
        pattern (Pattern): Pattern token embeddings.
        tokenizer (Tokenizer): Tokenizer.
        embedding (Embedding): Embeddings.
    """

    @stopwatch.timers("search", generator=True)
    def search(self, text: TokenEmbeddings, start: int = 0) -> Generator[Search.Match]:
        """Search for the pattern from the given text.

        Args:
            text (TokenEmbeddings): Text token embeddings to be searched.
            start (int): Start position to be searched.

        Yields:
            Match: Yield the `Match` object when a subsequence of text matches the pattern.
        """
        pattern_length = len(self.pattern)

        i = start
        while i <= len(text) - pattern_length:
            for j in range(pattern_length):
                if not self.is_match[j, text.tokens[i + j]]:
                    break
            else:
                match_scores = self.scores[
                    np.arange(pattern_length), text.tokens[i : i + pattern_length]
                ]
                yield self.Match(
                    i,
                    i + pattern_length,
                    match_scores,
                    text.tokens[i : i + pattern_length],
                )
            i += 1
