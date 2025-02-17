#!/usr/bin/env python3
import logging
import os
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Sequence

import simple_parsing
from simple_parsing import field

from softmatcha import configs, stopwatch
from softmatcha.struct import IndexInvertedFileCollection
from softmatcha.tokenizers import get_tokenizer

simple_parsing.parsing.logger.setLevel(logging.ERROR)
simple_parsing.wrappers.dataclass_wrapper.logger.setLevel(logging.ERROR)

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stderr,
)
logger = logging.getLogger("softmatcha.cli.build_inverted_index")


@dataclass
class IndexerArguments:
    """Indexer arguments."""

    # Paths to the input files.
    inputs: list[str] = field(positional=True, nargs="+")
    # Path to an index.
    index: str = field(alias=["-o"])
    # Specify the JSONL key of texts to be indexed.
    # If not specified this option, the inputs will be treated as plain text.
    jsonl_key: str | None = field(default=None)
    # Number of workers.
    num_workers: int = field(default=8)
    # Buffer size.
    buffer_size: int = field(default=10000)
    # Chunk size of HDF5 storage.
    chunk_size: int = field(default=1024)


def get_argparser(args: Sequence[str] | None = None) -> configs.ArgumentParser:
    parser = configs.get_argparser(args=args)
    parser.add_arguments(IndexerArguments, "indexer")
    return parser


def format_argparser() -> configs.ArgumentParser:
    parser = get_argparser()
    parser.preprocess_parser()
    return parser


def main(args: Namespace) -> None:
    logger.info(args)
    stopwatch.timers.reset(profile=True)

    input_paths = [os.path.abspath(input_path) for input_path in args.indexer.inputs]
    tokenizer_class = get_tokenizer(args.common.backend)

    tokenizer = tokenizer_class.build(
        tokenizer_class.Config(
            name_or_path=args.common.model,
            **{k: v for k, v in asdict(args.tokenizer).items() if k != "name_or_path"},
        )
    )

    IndexInvertedFileCollection.build(
        args.indexer.index,
        input_paths,
        tokenizer,
        jsonl_key=args.indexer.jsonl_key,
        num_workers=args.indexer.num_workers,
        buffer_size=args.indexer.buffer_size,
        chunk_size=args.indexer.chunk_size,
    )

    logger.info(f"Elapsed time: {stopwatch.timers.elapsed_time}")
    logger.info(f"Total time: {sum(stopwatch.timers.elapsed_time.values())}")
    logger.info(f"ncalls: {stopwatch.timers.ncalls}")


def cli_main() -> None:
    args = get_argparser().parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
