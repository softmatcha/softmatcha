from __future__ import annotations

import abc
from dataclasses import dataclass

from softmatcha.typing import PathLike


@dataclass
class Index:
    """Index base class to store the indexed information."""

    @classmethod
    @abc.abstractmethod
    def build(cls, *args, **kwargs) -> Index:
        """Build new index.

        Returns:
            Index: This class.
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: PathLike, *args, **kwargs) -> Index:
        """Load the index from a file.

        Args:
            path (PathLike): Path to a file.

        Returns:
            Index: This class.
        """
