#!/usr/bin/env python3
import json
import logging
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from io import TextIOWrapper
from typing import Generator, Sequence, TypeVar

import numpy as np
import simple_parsing
import termcolor
from simple_parsing import field

from softmatcha import configs, stopwatch
from softmatcha.cli import highlight
from softmatcha.embeddings import Embedding, get_embedding
from softmatcha.search import Search, SearchIndex, SearchIndexInvertedFile
from softmatcha.struct import Pattern
from softmatcha.struct.index_inverted import (
    IndexInvertedFile,
    IndexInvertedFileCollection,
)
from softmatcha.tokenizers import Tokenizer, get_tokenizer

E = TypeVar("E", bound=Embedding)
S = TypeVar("S", bound=Search)


@dataclass
class SearcherArguments:
    """Searcher arguments."""

    # Pattern string.
    pattern: str = field(positional=True)
    # Path to the index.
    index: str = field()

    # Threshold for soft matching.
    threshold: float = 0.5
    # Start position to be searched.
    start_position: int = 0
    # Start line to be searched.
    start_line: int = 0
    # Load the index on disk.
    # This option will slow down the search time,
    # but will improve memory consumption.
    mmap: bool = False
    # Use hash-based implementation.
    use_hash: bool = False


def get_argparser(args: Sequence[str] | None = None) -> configs.ArgumentParser:
    parser = configs.get_argparser(args=args)
    parser.add_arguments(SearcherArguments, "searcher")
    parser.add_arguments(configs.OutputArguments, "output")
    return parser


def format_argparser() -> configs.ArgumentParser:
    parser = get_argparser()
    parser.preprocess_parser()
    return parser


def format_output(
    matches: list[SearchIndex.Match],
    line_num: int,
    file_pointer: TextIOWrapper,
    file_index: IndexInvertedFile,
    tokenizer: Tokenizer,
    output_cfg: configs.OutputArguments = configs.OutputArguments(),
) -> str:
    file_pointer.seek(file_index.get_byte_offset(line_num))
    raw_line = file_pointer.readline()
    line = raw_line.rstrip()
    if file_index.jsonl_key is None:
        jsonl = None
        text = line
    else:
        jsonl = json.loads(line)
        text: str = jsonl[file_index.jsonl_key]

    matched_span_positions: list[tuple[int, int]] = []
    matched_symbol_sequences: list[list[str]] = []
    for m in matches:
        matched_symbol_sequence = tokenizer.decode(m.tokens)
        matched_symbol_sequences.append(matched_symbol_sequence)
        matched_span_positions.append(
            (
                file_index.get_token_span(m.begin),
                file_index.get_token_span(m.end - 1)
                + file_index.token_lengths[m.end - 1].item(),
            )
        )

    if output_cfg.json:
        matched_spans: list[str] = []
        for span_start, span_end in matched_span_positions:
            matched_spans.append(text[span_start:span_end])
        return json.dumps(
            {
                "line_number": line_num + 1,
                "original_line": raw_line,
                "scores": [
                    [float(f"{x:.4f}") for x in m.scores.tolist()] for m in matches
                ],
                "matched_tokens": matched_symbol_sequences,
                "matched_spans": matched_spans,
                "matched_span_ranges": matched_span_positions,
                "matched_token_start_positions": [
                    [file_index.get_token_span(i) for i in range(m.begin, m.end)]
                    for m in matches
                ],
            },
            ensure_ascii=False,
        )

    prefix_string = ""
    if output_cfg.with_filename:
        prefix_string += termcolor.colored(
            f"{file_pointer.name}", "magenta"
        ) + termcolor.colored(":", "green")
    if output_cfg.line_number:
        prefix_string += termcolor.colored(f"{line_num + 1}:", "green")

    if output_cfg.only_matching:
        return "\n".join(
            [
                prefix_string + highlight(text[span_start:span_end])
                for span_start, span_end in matched_span_positions
            ]
        )
    else:
        p = 0
        highlighted_text = ""
        for span_start, span_end in matched_span_positions:
            highlighted_text += text[p:span_start]
            highlighted_text += highlight(text[span_start:span_end])
            p = span_end
        highlighted_text += text[p:]
        if jsonl is not None:
            jsonl[file_index.jsonl_key] = highlighted_text
            highlighted_text = (
                json.dumps(jsonl, ensure_ascii=False)
                .replace(r"\u001b[31m", "\033[31m")
                .replace(r"\u001b[0m", "\033[0m")
            )
        return prefix_string + highlighted_text


@dataclass
class Statistics:
    """Statistics of the search.

    - num_hit_lines (int): Number of matched lines.
    - num_hit_spans (int): Number of matched spans.
    """

    num_hit_lines: int
    num_hit_spans: int


@dataclass
class Result:
    """An item of search results.

    - text (str): The text which contains matched patterns.
    - line_number (int): Line number of the matched text.
    - stats (Statistics): Statistics of the search.
    """

    text: str
    line_number: int
    stats: Statistics


def search_stats(pattern: Pattern, searcher: SearchIndexInvertedFile) -> Statistics:
    matched_positions, score_matrix = searcher._find(pattern)
    line_numbers = searcher.index.get_line_numbers(matched_positions)
    return Statistics(len(np.unique(line_numbers)), len(matched_positions))


def search_texts(
    pattern: Pattern,
    file_path: str,
    searcher: SearchIndexInvertedFile,
    tokenizer: Tokenizer,
    output_cfg: configs.OutputArguments = configs.OutputArguments(),
    start_position: int = 0,
    start_line: int = 0,
) -> Generator[Result, None, None]:
    matched_positions, score_matrix = searcher._find(pattern)
    file_index = searcher.index
    line_numbers = file_index.get_line_numbers(matched_positions)
    line_boundaries = set(
        np.unique(line_numbers, return_index=True)[1].tolist() + [len(line_numbers)]
    )
    stats = Statistics(len(line_boundaries) - 1, len(matched_positions))

    matches = []
    with open(file_path, mode="r") as file_pointer:
        for i, (line_num, p) in enumerate(
            zip(line_numbers.tolist(), matched_positions)
        ):
            if p < start_position or line_num < start_line:
                continue

            matches.append(searcher._get_span(p, p + len(pattern), score_matrix))
            if i + 1 in line_boundaries:
                yield Result(
                    format_output(
                        matches,
                        line_num,
                        file_pointer,
                        file_index,
                        tokenizer,
                        output_cfg,
                    ),
                    line_num,
                    stats,
                )
                matches = []


def main(args: Namespace) -> None:
    output_cfg: configs.OutputArguments = args.output
    stopwatch.timers.reset(profile=output_cfg.profile)

    if output_cfg.log:
        logging_config_kwargs = {}
        if output_cfg.log == "-":
            logging_config_kwargs["stream"] = sys.stderr
        else:
            logging_config_kwargs["filename"] = output_cfg.log
        logging.basicConfig(
            format="| %(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level="INFO",
            force=True,
            **logging_config_kwargs,
        )
    else:
        simple_parsing.parsing.logger.setLevel(logging.ERROR)
        simple_parsing.wrappers.dataclass_wrapper.logger.setLevel(logging.ERROR)
        logging.disable(logging.ERROR)

    logger = logging.getLogger("search")

    logger.info(args)

    with stopwatch.timers["load/embedding"]:
        embedding_class = get_embedding(args.common.backend)
        embedding: Embedding = embedding_class.build(
            embedding_class.Config(args.common.model, mmap=args.searcher.mmap)
        )
    with stopwatch.timers["load/tokenizer"]:
        tokenizer_class = get_tokenizer(args.common.backend)
        tokenizer: Tokenizer = tokenizer_class.build(
            tokenizer_class.Config(
                args.common.model,
                **{
                    k: v
                    for k, v in asdict(args.tokenizer).items()
                    if k != "name_or_path"
                },
            )
        )

    indexes = IndexInvertedFileCollection.load(
        args.searcher.index, mmap=args.searcher.mmap
    )

    def _query(pattern_str: str, start_position: int = 0, start_line: int = 0):
        pattern_tokens = tokenizer(pattern_str)
        pattern_embeddings = embedding(pattern_tokens)
        pattern = Pattern.build(
            pattern_tokens,
            pattern_embeddings,
            [args.searcher.threshold] * len(pattern_embeddings),
        )
        logger.info(f"Pattern length: {len(pattern):,}")

        for file_path, file_index in zip(indexes.paths, indexes.indexes):
            searcher = SearchIndexInvertedFile(
                file_index, tokenizer, embedding, use_hash=args.searcher.use_hash
            )
            logger.info(f"Search: {file_path}")

            if output_cfg.quiet:
                stats = search_stats(pattern, searcher)
            else:
                stats = Statistics(0, 0)
                for res in search_texts(
                    pattern,
                    file_path,
                    searcher,
                    tokenizer,
                    output_cfg=output_cfg,
                    start_position=start_position,
                    start_line=start_line,
                ):
                    stats = res.stats
                    print(res.text)

            logger.info(f"Number of hit spans: {stats.num_hit_spans:,}")
            logger.info(f"Number of hit lines: {stats.num_hit_lines:,}")

    _query(
        args.searcher.pattern,
        start_position=args.searcher.start_position,
        start_line=args.searcher.start_line,
    )

    if output_cfg.profile:
        print(
            f"elapsed_time\t{json.dumps(stopwatch.timers.elapsed_time)}",
            file=sys.stderr,
        )
        print(f"ncalls\t{json.dumps(stopwatch.timers.ncalls)}", file=sys.stderr)


def cli_main() -> None:
    args = get_argparser().parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
