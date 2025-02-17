import pytest

from softmatcha.embeddings import EmbeddingGensim
from softmatcha.struct import Pattern
from softmatcha.struct.token_embeddings import TokenEmbeddings
from softmatcha.tokenizers.moses import TokenizerMoses

from .quick import SearchQuick


class TestSearchQuick:
    def test_search_fullmatch_glove(
        self, embed_glove: EmbeddingGensim, tokenizer_glove: TokenizerMoses
    ):
        text_tokens = tokenizer_glove("Soft pattern match.")
        text_embeddings = embed_glove(text_tokens)
        pattern = Pattern.build(
            text_tokens, text_embeddings, [1 - 1e-5] * len(text_embeddings)
        )
        searcher = SearchQuick(pattern, tokenizer_glove, embed_glove)
        text = TokenEmbeddings(text_tokens, text_embeddings)
        res = searcher.search(text)
        assert next(res)
        with pytest.raises(StopIteration):
            next(res)

    def test_search_subseq_match(
        self, embed_glove: EmbeddingGensim, tokenizer_glove: TokenizerMoses
    ):
        text_tokens = tokenizer_glove("Soft pattern matching based on word embeddings")
        text_embeddings = embed_glove(text_tokens)
        pattern_tokens = tokenizer_glove("pattern matching")
        pattern_embeddings = embed_glove(pattern_tokens)
        pattern = Pattern.build(
            pattern_tokens, pattern_embeddings, [1 - 1e-5] * len(pattern_embeddings)
        )
        searcher = SearchQuick(pattern, tokenizer_glove, embed_glove)
        text = TokenEmbeddings(text_tokens, text_embeddings)
        res = searcher.search(text)
        assert next(res)
        with pytest.raises(StopIteration):
            next(res)

    def test_search_subseq_match_start_position(
        self, embed_glove: EmbeddingGensim, tokenizer_glove: TokenizerMoses
    ):
        text_tokens = tokenizer_glove(
            "Soft pattern matching based on word embeddings and normal hard pattern matching"
        )
        text_embeddings = embed_glove(text_tokens)
        pattern_tokens = tokenizer_glove("pattern matching")
        pattern_embeddings = embed_glove(pattern_tokens)
        pattern = Pattern.build(
            pattern_tokens, pattern_embeddings, [1 - 1e-5] * len(pattern_embeddings)
        )
        searcher = SearchQuick(pattern, tokenizer_glove, embed_glove)
        text = TokenEmbeddings(text_tokens, text_embeddings)
        res = searcher.search(text, start=2)
        matched = next(res)
        assert matched.begin == 10
        assert matched.end == 12
        with pytest.raises(StopIteration):
            next(res)

    def test_search_semantic_match_glove(
        self, embed_glove: EmbeddingGensim, tokenizer_glove: TokenizerMoses
    ):
        text_tokens = tokenizer_glove("He watched the shooting star.")
        text_embeddings = embed_glove(text_tokens)
        pattern_tokens = tokenizer_glove("saw a shooting star")
        pattern_embeddings = embed_glove(pattern_tokens)
        pattern = Pattern.build(
            pattern_tokens, pattern_embeddings, [0.5] * len(pattern_embeddings)
        )
        searcher = SearchQuick(pattern, tokenizer_glove, embed_glove)
        text = TokenEmbeddings(text_tokens, text_embeddings)
        res = searcher.search(text)
        assert next(res)
        with pytest.raises(StopIteration):
            next(res)

    def test_search_semantic_no_match_glove(
        self, embed_glove: EmbeddingGensim, tokenizer_glove: TokenizerMoses
    ):
        text_tokens = tokenizer_glove("He saw a television star.")
        text_embeddings = embed_glove(text_tokens)
        pattern_tokens = tokenizer_glove("saw a shooting star")
        pattern_embeddings = embed_glove(pattern_tokens)
        pattern = Pattern.build(
            pattern_tokens, pattern_embeddings, [0.5] * len(pattern_embeddings)
        )
        searcher = SearchQuick(pattern, tokenizer_glove, embed_glove)
        text = TokenEmbeddings(text_tokens, text_embeddings)
        res = searcher.search(text)
        with pytest.raises(StopIteration):
            next(res)

    def test_search_memorization(
        self, embed_glove: EmbeddingGensim, tokenizer_glove: TokenizerMoses
    ):
        text_tokens = tokenizer_glove(
            "Did you see stars yesterday? "
            "Do you see stars everyday? "
            "I saw many stars and also "
            "I heard that he watched the television star and saw a shooting star yesterday."
        )
        text_embeddings = embed_glove(text_tokens)
        pattern_tokens = tokenizer_glove("saw a shooting star")
        pattern_embeddings = embed_glove(pattern_tokens)
        pattern = Pattern.build(
            pattern_tokens, pattern_embeddings, [0.7] * len(pattern_embeddings)
        )
        searcher = SearchQuick(pattern, tokenizer_glove, embed_glove)
        text = TokenEmbeddings(text_tokens, text_embeddings)
        res = searcher.search(text)
        matched = next(res)
        assert matched is not None
        assert text_tokens[matched.begin : matched.end] == pattern_tokens
