from .transformers import TokenizerTransformers


class TestTokenizerTransformers:
    def test_build(self):
        tokenizer = TokenizerTransformers.build(
            TokenizerTransformers.Config("bert-base-uncased")
        )
        assert isinstance(tokenizer.dictionary, dict)

    def test_tokenize(self):
        tokenizer = TokenizerTransformers.build(
            TokenizerTransformers.Config("bert-base-uncased")
        )
        assert tokenizer.tokenize("Hello world!") == ["hello", "world", "!"]

    def test_encode(self):
        tokenizer = TokenizerTransformers.build(
            TokenizerTransformers.Config("bert-base-uncased")
        )
        assert tokenizer.encode(tokenizer.tokenize("Hello world!")) == [7592, 2088, 999]
