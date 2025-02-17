from __future__ import annotations

from typing import Generator

import numpy as np

from softmatcha import stopwatch
from softmatcha.embeddings import Embedding
from softmatcha.struct import Pattern, TokenEmbeddings
from softmatcha.tokenizers import Tokenizer

from .base import Search, SearchScan


class SearchQuick(SearchScan):
    """SearchQuick quickly finds the given pattern from a text.

    Args:
        pattern (Pattern): Pattern token embeddings.
        tokenizer (Tokenizer): Tokenizer.
        embedding (Embedding): Embeddings.
    """

    def __init__(
        self, pattern: Pattern, tokenizer: Tokenizer, embedding: Embedding
    ) -> None:
        super().__init__(pattern, tokenizer, embedding)
        self.pattern_length = len(self.pattern)
        self.shift_table: dict[int, int] = {}
        for j in reversed(range(self.pattern_length)):
            for i in self.is_match[j].nonzero()[0]:
                if i not in self.shift_table:
                    self.shift_table[i] = self.pattern_length - j

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
        text_length = len(text)

        i = start
        while i <= text_length - pattern_length:
            # Match the pattern.
            for j in reversed(range(pattern_length)):
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

            next_t = i + pattern_length
            if next_t >= text_length:
                break

            # Shift pattern tokens.
            i += self.shift_table.get(text.tokens[next_t], self.pattern_length + 1)
