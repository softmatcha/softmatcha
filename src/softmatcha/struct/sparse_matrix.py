from __future__ import annotations

from dataclasses import dataclass

import h5py

from softmatcha.typing import NDArrayI64


@dataclass
class SparseMatrix:
    """CSR sparse matrix for inverted indexes.

    This implementation omits values and only stores indptr and indices.

    - indptr (NDArrayI64): The number of elements for each row.
    - indices (NDArrayI64 | h5py.Dataset): The element indices that are not null.
    """

    indptr: NDArrayI64
    indices: NDArrayI64 | h5py.Dataset

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def __getitem__(self, i: int) -> NDArrayI64:
        return self.getrow(i)

    def getrow(self, i: int) -> NDArrayI64:
        return self.indices[self.indptr[i] : self.indptr[i + 1]]
