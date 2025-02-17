import time

import numpy as np
import numpy.linalg as LA

from . import functional as F

N = 10
K = 3
D = 64


def test_normalize():
    v = np.random.rand(N, D)
    assert not np.allclose(LA.norm(v, axis=-1), np.ones(N))
    assert np.allclose(LA.norm(F.normalize(v), axis=-1), np.ones(N))


def test_matmul_impl():
    N, Q, D = 3, 5, 8
    A = np.random.rand(N, D).astype(np.float32)
    q = np.random.rand(Q, D).astype(np.float32)
    res = F.matmul(A, q)
    expected = np.einsum("nd,qd->nq", A, q)
    assert np.allclose(res, expected)


def test_matmul_impl_speed():
    N, Q, D = 3, 100_000, 256
    A = np.random.rand(N, D).astype(np.float32)
    q = np.random.rand(Q, D).astype(np.float32)

    F.matmul(A, q)
    s = time.perf_counter()
    for _ in range(10):
        F.matmul(A, q)
    e = time.perf_counter()
    time_fast = e - s

    np.einsum("nd,qd->nq", A, q)
    s = time.perf_counter()
    for _ in range(10):
        np.einsum("nd,qd->nq", A, q)
    e = time.perf_counter()
    time_naive = e - s
    assert time_fast <= time_naive
