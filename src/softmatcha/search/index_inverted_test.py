from pathlib import Path
from typing import Generator

import h5py
import numba as nb
import numpy as np
import pytest
from numba.typed.typedlist import List

from softmatcha.embeddings import EmbeddingGensim
from softmatcha.struct import IndexInvertedFile, Pattern
from softmatcha.tokenizers.moses import TokenizerMoses

from .index_inverted import (
    SearchIndexInvertedFile,
    _isin_impl_binsearch_foreach,
    _isin_impl_hash,
)

indices = np.array([0, 1, 3, 5, 9, 10])
numba_array_type = nb.types.Array(nb.int64, 1, "C")


def test_isin_impl_binsearch_foreach():
    invlists = List.empty_list(numba_array_type)
    invlists.append(indices)

    assert np.array_equal(
        _isin_impl_binsearch_foreach(indices, invlists),
        np.ones_like(indices, dtype=np.bool_),
    )
    assert np.array_equal(
        _isin_impl_binsearch_foreach(indices, List.empty_list(numba_array_type)),
        np.zeros_like(indices, dtype=np.bool_),
    )
    b = np.array([8, 2, 3, 9])

    invlists = List.empty_list(numba_array_type)
    invlists.append(b)
    assert np.array_equal(
        _isin_impl_binsearch_foreach(indices, invlists), (indices == 3) | (indices == 9)
    )


def test_isin_impl_hash():
    assert np.array_equal(
        _isin_impl_hash(indices, indices),
        np.ones_like(indices, dtype=np.bool_),
    )
    assert np.array_equal(
        _isin_impl_hash(indices, np.array([], dtype=np.int64)),
        np.zeros_like(indices, dtype=np.bool_),
    )

    b = np.array([8, 2, 3, 9])
    assert np.array_equal(_isin_impl_hash(indices, b), (indices == 3) | (indices == 9))


@pytest.fixture
def index(tmp_path: Path, tokenizer_glove: TokenizerMoses) -> IndexInvertedFile:
    text_path = str(tmp_path / "text.txt")
    with open(text_path, mode="w") as f:
        print("I like the jazz music.", file=f)
        print("I have a pen.", file=f)

    index_path = str(tmp_path / "index.bin")
    index_root = h5py.File(index_path, mode="w")
    index_group = index_root.create_group("index")

    return IndexInvertedFile.build(index_group, text_path, tokenizer_glove)


class TestSearchIndexInvertedFile:
    def test_compute_exact_match(
        self,
        embed_glove: EmbeddingGensim,
        tokenizer_glove: TokenizerMoses,
        index: IndexInvertedFile,
    ):
        searcher = SearchIndexInvertedFile(index, tokenizer_glove, embed_glove)
        pattern_tokens = tokenizer_glove("the blues music")
        pattern = Pattern.build(
            pattern_tokens, embed_glove(pattern_tokens), [1.0] * len(pattern_tokens)
        )
        scores = searcher.compute_exact_match(pattern, embed_glove)
        assert list(scores.shape) == [len(pattern), len(embed_glove)]
        assert np.all(scores[np.arange(3), pattern_tokens] == 1.0)
        assert np.all(scores.sum() == 3.0)

    def test_compute_similarity(
        self,
        embed_glove: EmbeddingGensim,
        tokenizer_glove: TokenizerMoses,
        index: IndexInvertedFile,
    ):
        searcher = SearchIndexInvertedFile(index, tokenizer_glove, embed_glove)
        pattern_tokens = tokenizer_glove("the blues music")
        pattern = Pattern.build(
            pattern_tokens, embed_glove(pattern_tokens), [0.55] * len(pattern_tokens)
        )
        scores = searcher.compute_similarity(pattern, embed_glove)
        assert list(scores.shape) == [len(pattern), len(embed_glove)]
        assert np.all(scores[:, list(index.vocabulary)] >= 0.0)
        assert np.all(
            scores[
                :,
                list(
                    set(tokenizer_glove.tokens.keys()) - set(index.vocabulary.tolist())
                ),
            ]
            == 0.0
        )

    @pytest.mark.parametrize("use_hash", [False, True])
    def test_search(
        self,
        embed_glove: EmbeddingGensim,
        tokenizer_glove: TokenizerMoses,
        index: IndexInvertedFile,
        use_hash: bool,
    ):
        searcher = SearchIndexInvertedFile(
            index, tokenizer_glove, embed_glove, use_hash=use_hash
        )
        pattern_tokens = tokenizer_glove("the blues music")
        pattern = Pattern.build(
            pattern_tokens, embed_glove(pattern_tokens), [0.55] * len(pattern_tokens)
        )
        res: Generator[SearchIndexInvertedFile.Match] = searcher.search(pattern)
        matched = next(res)
        assert matched.begin == 2
        assert matched.end == 5
        with pytest.raises(StopIteration):
            next(res)

    @pytest.mark.parametrize("use_hash", [False, True])
    def test_search_start_position(
        self,
        tmp_path: Path,
        embed_glove: EmbeddingGensim,
        tokenizer_glove: TokenizerMoses,
        use_hash: bool,
    ):
        text_path = str(tmp_path / "text.txt")
        with open(text_path, mode="w") as f:
            print("I like the jazz music.", file=f)
            print("I have a pen.", file=f)
            print("I like the jazz music.", file=f)

        index_path = str(tmp_path / "index.bin")
        index_root = h5py.File(index_path, mode="w")
        index_group = index_root.create_group("index")

        index = IndexInvertedFile.build(index_group, text_path, tokenizer_glove)

        searcher = SearchIndexInvertedFile(
            index, tokenizer_glove, embed_glove, use_hash=use_hash
        )
        pattern_tokens = tokenizer_glove("the blues music")
        pattern = Pattern.build(
            pattern_tokens, embed_glove(pattern_tokens), [0.55] * len(pattern_tokens)
        )
        res: Generator[SearchIndexInvertedFile.Match] = searcher.search(
            pattern, start=6
        )
        matched = next(res)
        assert matched.begin == 13
        assert matched.end == 16
        with pytest.raises(StopIteration):
            next(res)
