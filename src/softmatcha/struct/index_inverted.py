from __future__ import annotations

import concurrent.futures
import logging
import math
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import Optional

import h5py
import numba as nb
import numpy as np

from softmatcha import stopwatch
from softmatcha.tokenizers import Tokenizer
from softmatcha.typing import NDArrayI32, NDArrayI64, NDArrayU16, NDArrayU64
from softmatcha.utils.io import buffer_lines

from .index import Index
from .sparse_matrix import SparseMatrix

logger = logging.getLogger(__name__)


@nb.njit(nb.int64[:](nb.uint64[:], nb.int64[:]), parallel=True, cache=True)
def _get_line_numbers_impl(
    line_offsets: NDArrayU64, token_positions: NDArrayI64
) -> NDArrayI64:
    n = token_positions.shape[0]
    outs = np.empty_like(token_positions)
    for i in nb.prange(n):
        outs[i] = np.searchsorted(line_offsets, token_positions[i], "right")
    return outs


@dataclass
class IndexInvertedFile(Index):
    """Inverted file index class.

    - tokens (NDArrayI32 | h5py.Dataset): Token ID sequence.
    - vocabulary (NDArrayI32): A set of token IDs that occured in the text.
    - inverted_lists (SparseMatrix): Inverted lists that record token positions.
    - token_offsets (NDArrayI64 | h5py.Dataset): Start positions of tokens in a line.
        Each element represents the character position in a line.
    - token_lengths (NDArrayU16 | h5py.Dataset): Token lengths that actually appeared in a text.
    - line_offsets (NDArrayU64): Offsets of line numbers.
        Each element has the token positions of the start of next lines.
    - byte_offsets (NDArrayU64): Offsets of byte offsets in the original file.
        Each element has the byte offsets of the start of lines in the original file.
    - jsonl_key (str, optional): The JSONL key to be indexed.
    """

    tokens: NDArrayI32 | h5py.Dataset
    vocabulary: NDArrayI32
    inverted_lists: SparseMatrix
    token_offsets: NDArrayI64 | h5py.Dataset
    token_lengths: NDArrayU16 | h5py.Dataset
    line_offsets: NDArrayU64
    byte_offsets: NDArrayU64
    jsonl_key: str | None = None

    @classmethod
    def _store_tokens(
        cls,
        index: h5py.Group,
        file_path: str,
        tokenizer: Tokenizer,
        jsonl_key: Optional[str] = None,
        num_workers: int = 8,
        buffer_size: int = 10000,
        chunk_size: int = 1024,
    ) -> Counter:
        """Store tokens and line offsets.

        Args:
            index (h5py.Group): File index.
            file_path (str): Indexed file.
            tokenizer (Tokenizer): Tokenizer.
            jsonl_key (str, optional): Key of texts to be indexed.
            num_workers (int): Number of workers.
            buffer_size (int): Buffer size.
            chunk_size (int): Chunk size of HDF5 storage.

        Returns:
            Counter: Frequency of each token type.
        """
        logger.info(f"Build an index: `{file_path}`.")
        tokens = index.create_dataset(
            "tokens",
            shape=(0,),
            dtype=np.int32,
            chunks=(chunk_size,),
            maxshape=(None,),
        )
        token_offsets = index.create_dataset(
            "token_offsets",
            shape=(0,),
            dtype=np.int64,
            chunks=(chunk_size,),
            maxshape=(None,),
        )
        token_lengths = index.create_dataset(
            "token_lengths",
            shape=(0,),
            dtype=np.uint16,
            chunks=(chunk_size,),
            maxshape=(None,),
        )
        line_offsets = index.create_dataset(
            "line_offsets",
            shape=(0,),
            dtype=np.uint64,
            chunks=(chunk_size,),
            maxshape=(None,),
        )
        counter = Counter()
        with concurrent.futures.ProcessPoolExecutor(
            num_workers, initializer=tokenizer.build, initargs=(tokenizer.cfg,)
        ) as executor:
            with stopwatch.timers["tokenize"]:
                line_offset = 0
                for buffer in buffer_lines(
                    file_path,
                    buffer_size=buffer_size * num_workers,
                    jsonl_key=jsonl_key,
                ):
                    symbol_sequences = list(
                        executor.map(tokenizer.tokenize, buffer, chunksize=buffer_size)
                    )
                    token_sequences = list(
                        executor.map(
                            tokenizer.encode, symbol_sequences, chunksize=buffer_size
                        )
                    )
                    lengths = [len(seq) for seq in token_sequences]
                    tokens.resize(len(tokens) + sum(lengths), axis=0)
                    flatten_token_sequences = list(chain.from_iterable(token_sequences))
                    tokens[-sum(lengths) :] = flatten_token_sequences
                    counter += Counter(flatten_token_sequences)
                    line_offsets.resize(len(line_offsets) + len(lengths), axis=0)
                    line_offsets[-len(lengths) :] = (
                        np.array(lengths, dtype=np.uint64).cumsum() + line_offset
                    )
                    line_offset = line_offsets[-1]

                    token_offsets_buffer = list(
                        chain.from_iterable(
                            tokenizer.get_span_start_positions(
                                line.lower(),
                                [sym.lower() for sym in symbol_sequence],
                            )
                            for line, symbol_sequence in zip(buffer, symbol_sequences)
                        )
                    )
                    token_offsets.resize(
                        len(token_offsets) + len(token_offsets_buffer), axis=0
                    )
                    token_offsets[-len(token_offsets_buffer) :] = token_offsets_buffer
                    token_lengths.resize(
                        len(token_lengths) + len(flatten_token_sequences), axis=0
                    )
                    token_lengths[-len(flatten_token_sequences) :] = [
                        len(symbol)
                        for symbol_sequence in symbol_sequences
                        for symbol in symbol_sequence
                    ]

        num_lines = len(index["line_offsets"])
        num_tokens = len(index["tokens"])
        logger.info(f"Number of lines: {num_lines:,}")
        logger.info(f"Number of tokens: {num_tokens:,}")

        for token in tokenizer.tokens:
            counter[token] += 0

        return counter

    @staticmethod
    def _record_byte_offsets_worker(
        file_path: str,
        start_offset: int,
        end_offset: int,
        read_size: int = 64 * 1024**2,
    ) -> NDArrayU64:
        """Read the byte offsets of lines within the given range.

        Args:
            file_path (str): Path to a text file.
            start_offset (int): The start offset of the range.
              This offset must be the beginning of a line.
            end_offset (int): The end offset of the range.
              This offset must be the beginning of a line.

        Returns:
            NDArrayI64: An array which stores the byte offsets of the beginning of lines.
        """
        offsets = [start_offset]
        with open(file_path, mode="rb") as f:
            f.seek(start_offset)
            i = start_offset
            while i < end_offset:
                if i + read_size < end_offset:
                    f.seek(i + read_size)
                    f.readline()
                    j = f.tell()
                    f.seek(i)
                else:
                    j = end_offset

                offsets += [
                    len(line) for line in f.read(j - i).splitlines(keepends=True)
                ]
                i = j
        return np.array(offsets[:-1], dtype=np.uint64).cumsum()

    @classmethod
    def _record_byte_offsets(
        cls, index: h5py.Group, file_path: str, num_lines: int, num_workers: int = 8
    ) -> None:
        """Record byte offsets of each line.

        Args:
            index (h5py.Group): File index.
            file_path (str): Indexed file.
            num_lines (int): Total number of lines.
            num_workers (int): Number of workers.
        """
        logger.info("Record byte offsets.")
        chunks = []
        with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, 2)
            file_size = f.tell()
            chunk_size = math.ceil(file_size / num_workers)

            i = 0
            j = chunk_size
            with stopwatch.timers["record_byte_offsets"]:
                while i < file_size:
                    f.seek(j)
                    line = f.readline()
                    j = f.tell()
                    chunks.append((i, j))
                    i, j = j, min(j + chunk_size, file_size)
                    if not line:
                        break

        byte_offsets = index.create_dataset(
            "byte_offsets", shape=(num_lines,), dtype=np.uint64
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            with stopwatch.timers["record_byte_offsets"]:
                futures = [
                    executor.submit(
                        cls._record_byte_offsets_worker, file_path, start, end
                    )
                    for start, end in chunks
                ]
                i = 0
                for future in futures:
                    res = future.result()
                    j = i + len(res)
                    byte_offsets[i:j] = res
                    i = j

    @classmethod
    def _build_index(
        cls,
        index: h5py.Group,
        tokens: h5py.Dataset,
        token_counter: Counter,
        chunk_size: int = 10**9,
    ) -> None:
        """Build a new inverted index from the given text tokens.

        Args:
            index (h5py.Group): File index.
            tokens (h5py.Dataset): Token sequence.
            token_counter (Counter): Frequency of each token type.
        """
        vocabulary, frequency = zip(*sorted(token_counter.items(), key=lambda k: k[0]))
        vocabulary = np.array(vocabulary, dtype=np.int32)
        frequency = np.array(frequency, dtype=np.int64)
        text_vocabulary = vocabulary[frequency > 0]
        vocabulary_size = max(vocabulary) + 1

        index.create_dataset("vocabulary", data=text_vocabulary)
        logger.info(f"Vocabulary size: {len(text_vocabulary):,}")

        def make_indptr(vocabulary: NDArrayI32, frequency: NDArrayI64) -> NDArrayI64:
            # The range from `indptr[v]` to `indptr[v+1]` records the positions of `v`.
            indptr = np.zeros(vocabulary_size + 1, dtype=np.int64)
            indptr[vocabulary + 1] = frequency
            return indptr.cumsum()

        invlists = index.create_group("inverted_lists")
        with stopwatch.timers["construct_invlists/indptr"]:
            indptr = make_indptr(vocabulary, frequency)
            invlists.create_dataset("indptr", data=indptr)
            logger.info(f"Done building indptr: {len(indptr):,}")

        num_tokens = len(tokens)
        indices = invlists.create_dataset(
            "indices", shape=(num_tokens,), dtype=np.int64
        )

        @nb.njit(nb.int64[:](nb.int32[:], nb.int64[:], nb.int64), cache=True)
        def make_indices(
            tokens: NDArrayI32,
            indptr: NDArrayI64,
            vocab_size: int,
        ) -> NDArrayI64:
            indices = np.zeros(len(tokens), dtype=np.int64)
            num_fills = np.zeros(vocab_size, dtype=np.int64)
            for i, token in enumerate(tokens):
                indices[indptr[token] + num_fills[token]] = i
                num_fills[token] += 1
            return indices

        # Fast but memory-consuming implementation:
        # indices[:] = construct_indices(tokens[()], indptr, vocabulary_size)

        # Memory efficient implementation:
        num_fills = np.zeros(vocabulary_size, dtype=np.int64)
        for i in range(0, num_tokens, chunk_size):
            with stopwatch.timers["construct_invlists/indices"]:
                buffer_tokens = tokens[i : i + chunk_size][()]
                buffer_vocab, buffer_freq = np.unique(buffer_tokens, return_counts=True)
                buffer_indptr = make_indptr(buffer_vocab, buffer_freq)
                buffer_indices = (
                    make_indices(buffer_tokens, buffer_indptr, vocabulary_size) + i
                )
                for v, c in zip(buffer_vocab, buffer_freq):
                    indices[indptr[v] + num_fills[v] : indptr[v] + num_fills[v] + c] = (
                        buffer_indices[buffer_indptr[v] : buffer_indptr[v + 1]]
                    )
                num_fills[buffer_vocab] += buffer_freq

    @classmethod
    def build(
        cls,
        index: h5py.Group,
        file_path: str,
        tokenizer: Tokenizer,
        jsonl_key: Optional[str] = None,
        num_workers: int = 8,
        buffer_size: int = 10000,
        chunk_size: int = 1024,
    ) -> IndexInvertedFile:
        """Build a new inverted index.

        Args:
            index (h5py.Group): File index.
            file_path (str): Indexed file.
            tokenizer (Tokenizer): Tokenizer.
            jsonl_key (str, optional): Key of texts to be indexed.
            num_workers (int): Number of workers.
            buffer_size (int): Buffer size.
            chunk_size (int): Chunk size of HDF5 storage.

        Returns:
            IndexInvertedFile: This class.
        """
        token_counter = cls._store_tokens(
            index,
            file_path,
            tokenizer,
            jsonl_key=jsonl_key,
            num_workers=num_workers,
            buffer_size=buffer_size,
            chunk_size=chunk_size,
        )
        cls._record_byte_offsets(
            index, file_path, len(index["line_offsets"]), num_workers=num_workers
        )
        cls._build_index(index, index["tokens"], token_counter, chunk_size=chunk_size)
        if jsonl_key is not None:
            index.create_dataset("jsonl_key", dtype=h5py.string_dtype(), data=jsonl_key)

        logger.info("Done building the index.")
        logger.info(f"Elapsed time: {stopwatch.timers.elapsed_time}")

        return cls(
            index["tokens"],
            index["vocabulary"][()],
            SparseMatrix(
                index["inverted_lists"]["indptr"][()],
                index["inverted_lists"]["indices"],
            ),
            index["token_offsets"],
            index["token_lengths"],
            index["line_offsets"][()],
            index["byte_offsets"][()],
            (jsonl_byte_key := index.get("jsonl_key", None))
            and jsonl_byte_key[()].decode(),
        )

    @classmethod
    def load(cls, index: h5py.Group, mmap: bool = False) -> IndexInvertedFile:
        """Load the index from a file.

        Args:
            index (h5py.Group): Path to a file.
            mmap (bool): Load the index on disk.

        Returns:
            IndexInvertedFile: This class.
        """
        self = cls(
            tokens=index["tokens"] if mmap else index["tokens"][()],
            vocabulary=index["vocabulary"][()],
            inverted_lists=SparseMatrix(
                index["inverted_lists"]["indptr"][()],
                index["inverted_lists"]["indices"]
                if mmap
                else index["inverted_lists"]["indices"][()],
            ),
            token_offsets=index["token_offsets"]
            if mmap
            else index["token_offsets"][()],
            token_lengths=index["token_lengths"]
            if mmap
            else index["token_lengths"][()],
            line_offsets=index["line_offsets"][()],
            byte_offsets=index["byte_offsets"][()],
            jsonl_key=(
                (jsonl_byte_key := index.get("jsonl_key", None))
                and jsonl_byte_key[()].decode()
            ),
        )

        logger.info(f"Vocabulary size: {len(self.vocabulary):,}")
        logger.info(f"Text length: {len(self.tokens):,}")
        logger.info(f"Number of lines: {len(self.line_offsets):,}")
        return self

    def get_token_span(self, token_position: int) -> int:
        """Get the span of the given token.

        Args:
            token_position (int): Position of the token.

        Returns:
            int: 0-indexed start position of the span.
        """
        return self.token_offsets[token_position].item()

    def get_token_spans(self, token_positions: NDArrayI64) -> NDArrayI64:
        """Get the span of the given token.

        Args:
            token_positions (NDArrayI64): Positions of the tokens.

        Returns:
            NDArrayI64: 0-indexed start positions of the spans.
        """
        return self.token_offsets[token_positions]

    def get_line_number(self, token_position: int) -> int:
        """Get the line number of the given token.

        Args:
            token_position (int): Position of the token.

        Returns:
            int: 0-indexed line number.
        """
        return np.searchsorted(self.line_offsets, token_position, "right").item()

    def get_line_numbers(self, token_positions: NDArrayI64) -> NDArrayU64:
        """Get the line number of the given token.

        Args:
            token_positions (NDArrayI64): Positions of the tokens.

        Returns:
            NDArrayI64: 0-indexed line numbers.
        """
        return _get_line_numbers_impl(self.line_offsets, token_positions)

    def get_byte_offset(self, line_number: int) -> int:
        """Get the byte offset of the start of line.

        Args:
            line_number (int): 0-indexed line number.

        Returns:
            int: Byte offset of the start of line in the original file path.
        """
        return self.byte_offsets[line_number]


@dataclass
class IndexInvertedFileCollection(Index):
    """Collection of inverted file indexes.

    - paths (list[str]): Original file paths.
    - indexes (list[IndexInvertedFile]): Inverted file indexes.
    """

    paths: list[str]
    indexes: list[IndexInvertedFile]

    @classmethod
    def build(
        cls,
        index_path: str,
        file_paths: list[str],
        tokenizer: Tokenizer,
        jsonl_key: Optional[str] = None,
        num_workers: int = 8,
        buffer_size: int = 10000,
        chunk_size: int = 1024,
    ) -> IndexInvertedFileCollection:
        """Build new inverted index collection.

        Args:
            index_path (str): Path to the index file.
            file_paths (list[str]): Paths to the text files.
            tokenizer (Tokenizer): Tokenizer.
            jsonl_key (str, optional): Key of texts to be indexed.
            num_workers (int): Number of workers.
            buffer_size (int): Buffer size.
            chunk_size (int): Chunk size of HDF5 storage.

        Returns:
            IndexInvertedFileCollection: This class.
        """
        with h5py.File(index_path, mode="w") as f:
            f.create_dataset(
                "paths",
                shape=(len(file_paths),),
                dtype=h5py.string_dtype(),
                data=file_paths,
            )
            index_root = f.create_group("indexes")
            file_indexes = []

            total_lines = 0
            total_tokens = 0
            for file_idx, file_path in enumerate(file_paths):
                file_index = IndexInvertedFile.build(
                    index_root.create_group(f"{file_idx}"),
                    file_path,
                    tokenizer,
                    jsonl_key=jsonl_key,
                    num_workers=num_workers,
                    buffer_size=buffer_size,
                    chunk_size=chunk_size,
                )
                file_indexes.append(file_index)
                total_lines += len(file_index.line_offsets)
                total_tokens += len(file_index.tokens)
        logger.info(f"Total number of lines: {total_lines:,}")
        logger.info(f"Total number of tokens: {total_tokens:,}")
        return cls(file_paths, file_indexes)

    @classmethod
    def load(cls, index_path: str, mmap: bool = False) -> IndexInvertedFileCollection:
        """Load the index from a file.

        Args:
            index_path (str): Path to an index file.
            mmap (bool): Load the index on disk.

        Returns:
            IndexInvertedFileCollection: This class.
        """
        index_collection = h5py.File(index_path, mode="r")
        paths = [path_byte.decode() for path_byte in index_collection["paths"][()]]
        indexes: list[IndexInvertedFile] = []
        for i, path in enumerate(paths):
            logger.info(f"Load the index of {path}")
            with stopwatch.timers["load/index"]:
                indexes.append(
                    IndexInvertedFile.load(
                        index_collection["indexes"][str(i)], mmap=mmap
                    )
                )
        return cls(paths, indexes)
