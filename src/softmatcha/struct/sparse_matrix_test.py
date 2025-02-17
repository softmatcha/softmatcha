from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sparse_matrix import SparseMatrix

INDPTR = np.array([0, 3, 0, 1])
INDICES = np.array([0, 3, 5, 4])


@dataclass
class TestSparseMatrix:
    def test_len(self):
        sparse_matrix = SparseMatrix(INDPTR, INDICES)
        assert len(sparse_matrix) == INDPTR - 1

    def test_getitem(self):
        sparse_matrix = SparseMatrix(INDPTR, INDICES)
        assert sparse_matrix[0] == np.array([0, 3, 5])
        assert sparse_matrix[1] == np.array([], dtype=np.int64)
        assert sparse_matrix[2] == np.array([4])

    def test_getrow(self):
        sparse_matrix = SparseMatrix(INDPTR, INDICES)
        assert sparse_matrix[0] == np.array([0, 3, 5])
        assert sparse_matrix[1] == np.array([], dtype=np.int64)
        assert sparse_matrix[2] == np.array([4])
