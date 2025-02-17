import concurrent.futures
import json
import multiprocessing as mp
import pathlib
import sys
from multiprocessing.context import BaseContext

import pytest

from .moses import TokenizerMoses


class TestTokenizerMoses:
    @pytest.fixture
    def dict_dir(self, tmp_path: pathlib.Path):
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
        return str(tmp_path)

    def test_tokenize(self, dict_dir: str):
        tokenizer = TokenizerMoses.build(TokenizerMoses.Config(dict_dir))
        expected_tokens = ["i", "'m", "happy", "."]
        assert tokenizer.tokenize("I'm happy.") == expected_tokens

    def test_encode(self, dict_dir: str):
        text = "Hello, world!"
        tokenizer = TokenizerMoses.build(TokenizerMoses.Config(dict_dir))
        tokens = tokenizer.encode(tokenizer.tokenize(text))
        assert tokens == [4, 5, 6, 7]

    def test_call(self, dict_dir: str):
        text = "I'm happy."
        tokenizer = TokenizerMoses.build(TokenizerMoses.Config(dict_dir))
        tokens = tokenizer(text)
        assert tokens == [0, 1, 2, 3]

    @pytest.mark.parametrize(
        "mp_context",
        [
            mp.get_context("spawn"),
            pytest.param(
                mp.get_context("fork"),
                marks=pytest.mark.skipif(
                    sys.platform != "linux", reason="Not supported method."
                ),
            ),
        ],
    )
    def test_call_multiprocess(self, dict_dir: str, mp_context: BaseContext):
        texts = ["I'm happy.", "I'm sad."]
        tokenizer = TokenizerMoses.build(TokenizerMoses.Config(dict_dir))

        with concurrent.futures.ProcessPoolExecutor(
            mp_context=mp_context,
            initializer=tokenizer.build,
            initargs=(tokenizer.cfg,),
        ) as executor:
            tokens = list(executor.map(tokenizer, texts))
        assert tokens[0] == [0, 1, 2, 3]
        assert tokens[1] == [0, 1, 8, 3]
