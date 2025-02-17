from __future__ import annotations

import warnings
from typing import Generator

import numba as nb
import numba.typed.typedlist as tl_mod
import numpy as np
import numpy.typing as npt
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaTypeSafetyWarning
from numba.typed.typedlist import List

import softmatcha.functional as F
from softmatcha import stopwatch
from softmatcha.embeddings import Embedding
from softmatcha.struct import IndexInvertedFile, Pattern
from softmatcha.tokenizers import Tokenizer
from softmatcha.typing import NDArrayF32, NDArrayI32, NDArrayI64

from .base import Search, SearchIndex

warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)

numba_array_type = nb.types.Array(nb.int64, 1, "C")


######## Maximize the performance of numba ########
# Speed up TypedList.append()
@nb.njit(nb.types.void(nb.types.ListType(numba_array_type), numba_array_type))
def append_jit(lst: nb.types.ListType, elm: nb.types.Array) -> None:
    lst.append(elm)


append = append_jit.overloads[
    (nb.types.ListType(numba_array_type), numba_array_type)
].entry_point


# Monkey patch numba so that the builtin functions for List() cache between runs.
def monkey_patch_caching(mod, exclude=[]):
    for name, val in mod.__dict__.items():
        if isinstance(val, Dispatcher) and name not in exclude:
            val.enable_caching()


monkey_patch_caching(tl_mod, ["_sort"])

List.empty_list(numba_array_type)
###################################################


@nb.njit(
    nb.boolean[:](nb.int64[:], nb.types.ListType(numba_array_type)),
    locals={
        "mask": nb.boolean[:],
        "m": nb.int64,
        "n": nb.int64,
        "a_i": nb.int64,
        "b": nb.int64[:],
        "blen": nb.int64,
    },
    parallel=True,
    cache=True,
)
def _isin_impl_binsearch_foreach(
    a: NDArrayI64, bs: List[NDArrayI64]
) -> npt.NDArray[np.bool_]:
    n = a.shape[0]
    mask = np.zeros(n, dtype=nb.boolean)
    for j in nb.prange(len(bs)):
        b = bs[j]
        blen = len(b)
        for i in nb.prange(n):
            a_i = a[i]
            m = np.searchsorted(b, a_i)
            if m < blen and b[m] == a_i:
                mask[i] = True
    return mask


@nb.njit(
    nb.boolean[:](nb.int64[:], nb.int64[:]),
    locals={"mask": nb.boolean[:]},
    parallel=True,
    cache=True,
)
def _isin_impl_hash(a: NDArrayI64, b: NDArrayI64) -> npt.NDArray[np.bool_]:
    mask = np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        mask[i] = a[i] in b
    return mask


def find_matches_binsearch(
    matched_invlists: list[list[NDArrayI64]],
) -> NDArrayI64:
    """Find start positions of matched patterns with binary search.

    Args:
        matched_invlists (list[list[NDArrayI64]]): Inverted lists of matched tokens.

    Returns:
        NDArrayI64: The start positions of matched patterns.
    """
    matches: NDArrayI64

    rare_ordered_pattern_indices: list[int] = (
        np.array([sum(map(len, m)) for m in matched_invlists]).argsort().tolist()
    )

    p = rare_ordered_pattern_indices[0]
    with stopwatch.timers["union"]:
        matches = np.concatenate(matched_invlists[p])
    with stopwatch.timers["shift+and"]:
        matches -= p

    for p in rare_ordered_pattern_indices[1:]:
        with stopwatch.timers["union"]:
            matched_invlists_p = List.empty_list(numba_array_type)
            for m in matched_invlists[p]:
                append(matched_invlists_p, m)

        with stopwatch.timers["shift+and"]:
            matches += p
            matches = matches[_isin_impl_binsearch_foreach(matches, matched_invlists_p)]
            matches -= p

    with stopwatch.timers["sort"]:
        matches.sort()
    return matches


def find_matches_hash(
    matched_invlists: list[list[NDArrayI64]],
) -> NDArrayI64:
    """Find start positions of matched patterns with hash.

    Args:
        matched_invlists (list[list[NDArrayI64]]): Inverted lists of matched tokens.

    Returns:
        NDArrayI64: The start positions of matched patterns.
    """
    matches: NDArrayI64
    rare_ordered_pattern_indices: list[int] = (
        np.array([sum(map(len, m)) for m in matched_invlists]).argsort().tolist()
    )

    p = rare_ordered_pattern_indices[0]
    with stopwatch.timers["union"]:
        matches = np.concatenate(matched_invlists[p])
    with stopwatch.timers["shift+and"]:
        matches -= p

    for p in rare_ordered_pattern_indices[1:]:
        with stopwatch.timers["shift+and"]:
            matches += p
            intersections = [
                matches[_isin_impl_hash(matches, m)]
                if len(matches) > len(m)
                else m[_isin_impl_hash(m, matches)]
                for m in matched_invlists[p]
            ]
        with stopwatch.timers["union"]:
            matches = np.concatenate(intersections)
            matches -= p

    with stopwatch.timers["sort"]:
        matches.sort()
    return matches


class SearchIndexInvertedFile(SearchIndex):
    """SearchIndex class to find the given pattern from a text using the index.

    Args:
        index (Index): An index to be used for searching the text quickly.
        tokenizer (Tokenizer): Tokenizer.
        embedding (Embedding): Embeddings.
        use_hash (bool): Use another implementation based on hash.
    """

    def __init__(
        self,
        index: IndexInvertedFile,
        tokenizer: Tokenizer,
        embedding: Embedding,
        use_hash: bool = False,
    ) -> None:
        super().__init__(index, tokenizer, embedding)
        self.vocabulary_embeddings = embedding.embeddings[self.index.vocabulary]
        self.use_hash = use_hash

    index: IndexInvertedFile

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
                scores[:, self.index.vocabulary] = F.matmul(
                    pattern.embeddings, self.vocabulary_embeddings
                )
            else:
                scores[:, vocabulary] = F.matmul(
                    pattern.embeddings, self.embedding.embeddings[vocabulary]
                )

            scores[:, self.tokenizer.unk_idx] = 0.0
            for i, p in enumerate(pattern.tokens):
                if p == self.tokenizer.unk_idx:
                    scores[i] = 0.0

        return scores

    @stopwatch.timers("search")
    def _find(self, pattern: Pattern) -> tuple[NDArrayI64, NDArrayF32]:
        """Find the start positions of the pattern in the text.

        Args:
            pattern (Pattern): Pattern token embeddings.

        Returns:
            tuple[NDArrayI64, NDArrayF32]:
              - NDArrayI64: Start positions of the pattern.
              - NDArrayF32: Similarity score matrix of shape `(P, V)`.
        """
        # Compute pattern--vocabualry pairwise similarity.
        if (pattern.thresholds >= 1.0).all():
            scores = self.compute_exact_match(pattern, self.embedding)
        else:
            scores = self.compute_similarity(pattern, self.embedding)

        # Concatenate the matched token index vectors for each pattern.
        matched_invlists = [[] for _ in range(len(pattern))]
        for pattern_idx, vocabulary_idx in zip(
            *(scores >= pattern.thresholds[:, None]).nonzero()
        ):
            matched_invlists[pattern_idx].append(
                self.index.inverted_lists.getrow(vocabulary_idx)
            )
        if any([len(m) == 0 for m in matched_invlists]):
            return np.array([], dtype=np.int64), scores

        if self.use_hash:
            matches = find_matches_hash(matched_invlists)
        else:
            matches = find_matches_binsearch(matched_invlists)

        return matches, scores

    def _get_span(self, begin: int, end: int, scores: NDArrayF32) -> SearchIndex.Match:
        """Get a matched span.

        Args:
            begin (int): The start position of the span.
            end (int): The end position of the span.
            scores (NDArrayF32): Similarity score matrix of shape `(P, V)`.

        Returns:
            NDArrayI64: Start positions of the pattern.
        """
        matched_tokens = self.index.tokens[begin:end].tolist()
        match_scores = scores[range(end - begin), matched_tokens]
        return self.Match(begin, end, match_scores, matched_tokens)

    def search(self, pattern: Pattern, start: int = 0) -> Generator[Search.Match]:
        """Search for the pattern from the given text.

        Args:
            pattern (Pattern): Pattern token embeddings.
            start (int): Start position to be searched.

        Yields:
            Match: Yield the Match object when a subsequence of text matches the pattern.
        """
        matches, scores = self._find(pattern)
        pattern_length = len(pattern)
        for begin in matches.tolist():
            if begin < start:
                continue
            end = begin + pattern_length
            yield self._get_span(begin, end, scores)
