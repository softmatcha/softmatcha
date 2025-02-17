import numpy as np

from softmatcha.tokenizers.transformers import TokenizerTransformers

from .transformers import EmbeddingTransformers


class TestEmbeddingTransformers:
    def test_load(self):
        embedding = EmbeddingTransformers.load("bert-base-uncased")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == 768

    def test_embed(
        self, embed_bert: EmbeddingTransformers, tokenizer_bert: TokenizerTransformers
    ):
        embeddings = [
            embed_bert(tokenizer_bert.encode(tokenizer_bert.tokenize(text)))
            for text in ["hello world!", "Amazing Bud Powell vol.1", "hello world!"]
        ]
        assert len(embeddings) == 3
        assert list(embeddings[0].shape) == [3, 768]
        assert list(embeddings[1].shape) == [6, 768]
        assert list(embeddings[2].shape) == [3, 768]
        assert np.allclose(embeddings[0], embeddings[2])
