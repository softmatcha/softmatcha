import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from softmatcha.tokenizers.moses import Tokenizer, TokenizerMoses

from .index_inverted import IndexInvertedFile


@pytest.fixture
def file_index(tmp_path: Path) -> h5py.Group:
    index_path = str(tmp_path / "index.bin")
    return h5py.File(index_path, mode="w").create_group("index")


@pytest.fixture
def tokenizer(tmp_path: Path):
    json_file = tmp_path / "vocab.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "i": 0,
                "'m": 1,
                "happy": 2,
                ".": 3,
                "hello": 4,
                ",": 5,
                "world": 6,
                "!": 7,
            },
            f,
        )

    return TokenizerMoses.build(TokenizerMoses.Config(str(tmp_path)))


@pytest.fixture
def file_path(tmp_path: Path) -> str:
    path = str(tmp_path / "text.txt")
    with open(path, mode="w") as f:
        print("i'm happy.", file=f)
        print("hello, world!", file=f)
        print("i'm happy.", file=f)
    return path


class TestIndexInvertedFile:
    def test_store_tokens(
        self, file_index: h5py.Group, file_path: str, tokenizer: Tokenizer
    ):
        counter = IndexInvertedFile._store_tokens(file_index, file_path, tokenizer)
        assert len(file_index["tokens"]) == 12
        assert np.array_equal(
            file_index["tokens"][()],
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int32),
        )
        assert len(file_index["line_offsets"]) == 3
        assert np.array_equal(
            file_index["line_offsets"][()], np.array([4, 8, 12], dtype=np.uint64)
        )

        # ID-8: <unk>
        assert dict(counter) == {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0}

    def test_record_byte_offsets(
        self, file_index: h5py.Group, file_path: str, tokenizer: Tokenizer
    ):
        IndexInvertedFile._store_tokens(file_index, file_path, tokenizer)
        IndexInvertedFile._record_byte_offsets(
            file_index,
            file_path,
            len(file_index["line_offsets"]),
        )
        assert len(file_index["byte_offsets"]) == 3
        assert np.array_equal(
            file_index["byte_offsets"][()], np.array([0, 11, 25], dtype=np.uint64)
        )

    def test_build_index(
        self, file_index: h5py.Group, file_path: str, tokenizer: Tokenizer
    ):
        counter = IndexInvertedFile._store_tokens(file_index, file_path, tokenizer)
        IndexInvertedFile._build_index(
            file_index, file_index["tokens"], counter, len(tokenizer)
        )
        assert len(file_index["inverted_lists"]["indptr"]) == len(tokenizer) + 1
        assert file_index["inverted_lists"]["indptr"][1] == 2
        assert np.array_equal(
            file_index["inverted_lists"]["indices"][
                file_index["inverted_lists"]["indptr"][0] : file_index[
                    "inverted_lists"
                ]["indptr"][1]
            ][()],
            np.array([0, 8]),
        )

    def test_build(self, file_index: h5py.Group, file_path: str, tokenizer: Tokenizer):
        index = IndexInvertedFile.build(file_index, file_path, tokenizer)

        assert len(index.inverted_lists) == len(tokenizer)
        assert np.array_equal(index.inverted_lists[0], np.array([0, 8]))
        assert np.array_equal(index.inverted_lists[8], np.array([]))
        assert np.array_equal(index.inverted_lists[2], np.array([2, 10]))
        assert np.array_equal(index.inverted_lists[5], np.array([5]))

    def test_get_line_number(
        self, file_index: h5py.Group, file_path: str, tokenizer: Tokenizer
    ):
        index = IndexInvertedFile.build(file_index, file_path, tokenizer)
        assert index.get_line_number(0) == 0
        assert index.get_line_number(1) == 0
        assert index.get_line_number(3) == 0
        assert index.get_line_number(4) == 1
        assert index.get_line_number(7) == 1
        assert index.get_line_number(8) == 2

    def test_get_byte_offset(
        self, file_index: h5py.Group, file_path: str, tokenizer: Tokenizer
    ):
        index = IndexInvertedFile.build(file_index, file_path, tokenizer)
        assert index.get_byte_offset(0) == 0
        assert index.get_byte_offset(1) == 11
        assert index.get_byte_offset(2) == 25
