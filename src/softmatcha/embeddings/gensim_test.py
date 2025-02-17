import numpy as np

from softmatcha.tokenizers.gensim import TokenizerGensim

from .gensim import EmbeddingGensim

D = 300


class TestEmbeggingGensim:
    def test_load(self):
        embedding = EmbeddingGensim.load("glove-wiki-gigaword-300")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == D

    def test_embed(
        self, embed_glove: EmbeddingGensim, tokenizer_glove: TokenizerGensim
    ):
        texts = ["I'm so happy.", "Hello, world!"]
        embeddings = [
            embed_glove(tokenizer_glove.encode(tokenizer_glove.tokenize(text)))
            for text in texts
        ]
        assert len(embeddings) == 2
        assert list(embeddings[0].shape) == [5, D]
        assert list(embeddings[1].shape) == [4, D]
