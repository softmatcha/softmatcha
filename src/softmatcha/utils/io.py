import fileinput
from typing import Generator

import simdjson
import tqdm


def read_lines(
    inputs: str = "-", jsonl_key: str | None = None
) -> Generator[str, None, None]:
    """Read and yield lines.

    Args:
        inputs (str): Input file. `-` reads from the standard input.
        jsonl_key (str, optional): If specified, read the text from the key.

    Yields:
        str: A line.
    """
    if jsonl_key is not None:
        parser = simdjson.Parser()
        with fileinput.input(files=[inputs], mode="rb") as f:
            for line in f:
                yield parser.parse(line).at_pointer(f"/{jsonl_key}")
    else:
        with fileinput.input(
            files=[inputs], mode="r", openhook=fileinput.hook_encoded("utf-8")
        ) as f:
            for line in f:
                yield line.rstrip()


def buffer_lines(
    inputs: str = "-",
    buffer_size: int = 1000,
    jsonl_key: str | None = None,
) -> Generator[list[str], None, None]:
    """Read and yield lines.

    Args:
        inputs (str): Input file. `-` reads from the standard input.
        buffer_size (int): Buffer size.
        jsonl_key (str, optional): If specified, read the text from the key.

    Yields:
        list[str]: Buffered lines.
    """
    if jsonl_key is not None:
        parser = simdjson.Parser()
        with fileinput.input(files=[inputs], mode="rb") as f:
            buffer = []
            for line in tqdm.tqdm(f):
                buffer.append(parser.parse(line).at_pointer(f"/{jsonl_key}"))
                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []
            if len(buffer) > 0:
                yield buffer
    else:
        with fileinput.input(
            files=[inputs], mode="r", openhook=fileinput.hook_encoded("utf-8")
        ) as f:
            buffer = []
            for line in tqdm.tqdm(f):
                buffer.append(line.rstrip())
                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []
            if len(buffer) > 0:
                yield buffer
