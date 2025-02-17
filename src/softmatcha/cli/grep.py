#!/usr/bin/env python3

import json
import logging
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Sequence, TypeVar

import simple_parsing
from simple_parsing import choice, field
from termcolor import termcolor

from softmatcha import configs, stopwatch
from softmatcha.cli import highlight
from softmatcha.embeddings import Embedding, get_embedding
from softmatcha.search import (
    SearchNaive,
    SearchQuick,
    SearchScan,
)
from softmatcha.struct import Pattern, TokenEmbeddings
from softmatcha.tokenizers import Tokenizer, get_tokenizer
from softmatcha.utils import io as io_utils

E = TypeVar("E", bound=Embedding)
S = TypeVar("S", bound=SearchScan)


@dataclass
class SearcherArguments:
    """Searcher arguments."""

    # Pattern string.
    pattern: str = field(positional=True)
    # Path to a text file.
    text_file: str = field(positional=True, nargs="*", default="-")
    # Search method.
    search: str = choice(*["naive", "quick"], default="quick")
    # Threshold for soft matching.
    threshold: float = 0.5
    # Start position to be searched.
    start_position: int = 0


def get_argparser(args: Sequence[str] | None = None) -> configs.ArgumentParser:
    parser = configs.get_argparser(args=args)
    parser.add_arguments(SearcherArguments, "searcher")
    parser.add_arguments(configs.OutputArguments, "output")
    return parser


def format_argparser() -> configs.ArgumentParser:
    parser = get_argparser()
    parser.preprocess_parser()
    return parser


def main(args: Namespace) -> None:
    stopwatch.timers.reset(profile=args.output.profile)

    if getattr(args.output, "log", None):
        logging_config_kwargs = {}
        if args.output.log == "-":
            logging_config_kwargs["stream"] = sys.stderr
        else:
            logging_config_kwargs["filename"] = args.output.log
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
            embedding_class.Config(args.common.model)
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

    pattern_tokens = tokenizer(args.searcher.pattern)
    pattern_embeddings = embedding(pattern_tokens)
    pattern = Pattern.build(
        pattern_tokens,
        pattern_embeddings,
        [args.searcher.threshold] * len(pattern_embeddings),
    )

    searcher: SearchScan = {"naive": SearchNaive, "quick": SearchQuick}[
        args.searcher.search
    ](pattern, tokenizer, embedding)

    num_tokens = 0
    num_lines = 0
    for file_path in args.searcher.text_file:
        for i, line in enumerate(io_utils.read_lines(file_path)):
            with stopwatch.timers["tokenize"]:
                text_symbols = tokenizer.tokenize(line)
                span_starts = tokenizer.get_span_start_positions(
                    line.lower(), text_symbols
                )
                text_tokens = tokenizer.encode(text_symbols)
            text_embeddings = embedding(text_tokens)
            num_tokens += len(text_embeddings)
            text = TokenEmbeddings(text_tokens, text_embeddings)
            matches = list(searcher.search(text, start=args.searcher.start_position))
            if len(matches) <= 0:
                continue

            token_lengths = [len(symbol) for symbol in text_symbols]
            matched_span_positions: list[tuple[int, int]] = []
            matched_symbol_sequences: list[list[str]] = []
            for m in matches:
                matched_symbol_sequence = tokenizer.decode(m.tokens)
                matched_symbol_sequences.append(matched_symbol_sequence)
                matched_span_positions.append(
                    (
                        span_starts[m.begin],
                        span_starts[m.end - 1] + token_lengths[m.end - 1],
                    )
                )

            prefix_string = ""
            if not args.output.json:
                if args.output.with_filename or (
                    len(args.searcher.text_file) > 1 and not args.output.no_filename
                ):
                    prefix_string += termcolor.colored(
                        f"{file_path}", "magenta"
                    ) + termcolor.colored(":", "green")
                if args.output.line_number:
                    prefix_string += termcolor.colored(f"{i + 1}:", "green")

            if args.output.json:
                d = {
                    "path": file_path,
                    "line_number": i + 1,
                    "original_line": line,
                    "matched_tokens": matched_symbol_sequences,
                    "matched_spans": [line[s:e] for s, e in matched_span_positions],
                    "scores": [
                        [float(f"{x:.4f}") for x in m.scores.tolist()] for m in matches
                    ],
                }
                print(json.dumps(d, ensure_ascii=False))
            elif args.output.only_matching:
                for span_start, span_end in matched_span_positions:
                    print(prefix_string + highlight(line[span_start:span_end]))
            else:
                p = 0
                highlighted_text = ""
                for span_start, span_end in matched_span_positions:
                    highlighted_text += line[p:span_start]
                    highlighted_text += highlight(line[span_start:span_end])
                    p = span_end
                highlighted_text += line[p:]
                print(prefix_string + highlighted_text)
            num_lines += 1

    if args.output.profile:
        print(
            f"elapsed_time\t{json.dumps(stopwatch.timers.elapsed_time)}",
            file=sys.stderr,
        )
        print(f"ncalls\t{json.dumps(stopwatch.timers.ncalls)}", file=sys.stderr)
        print(f"nlines\t{num_lines:,}", file=sys.stderr)
        print(f"ntokens\t{num_tokens:,}", file=sys.stderr)
        print(f"ntokens/sentence\t{num_tokens/num_lines:.1f}", file=sys.stderr)
        if isinstance(searcher, SearchQuick):
            print(f"table_size\t{len(searcher.shift_table)}", file=sys.stderr)


def cli_main() -> None:
    args = get_argparser().parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
