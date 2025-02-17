import time
from contextlib import nullcontext

import pytest

from .stopwatch import Stopwatch, StopwatchDict


class TestStopwatch:
    def test__init__(self) -> None:
        timer = Stopwatch()
        assert timer._acc_time == 0.0

    def test_reset(self) -> None:
        timer = Stopwatch()
        assert timer._acc_time == 0.0
        timer._acc_time = 1.0
        assert timer._acc_time != 0.0
        timer.reset()
        assert timer._acc_time == 0.0

    def test___call__(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timer = Stopwatch()
        for _ in range(10):
            with timer:
                t += 1.0
        assert timer.elpased_time == 10.0

    def test_ncalls(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timer = Stopwatch()
        for _ in range(20):
            with timer:
                t += 2.0
        assert timer.elpased_time == 40.0
        assert timer.ncalls == 20


class TestStopwatchDict:
    def test__init__(self) -> None:
        timers = StopwatchDict()
        assert len(timers) == 0

    def test_reset(self) -> None:
        timers = StopwatchDict()
        names = ["A", "B"]
        assert all([timers[name]._acc_time == 0.0 for name in names])
        for t in timers.values():
            t._acc_time = 1.0
        assert all([timers[name]._acc_time != 0.0 for name in names])
        timers.reset()
        assert all([isinstance(timers[name], nullcontext) for name in names])

    def test___call__(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timers = StopwatchDict()
        for _ in range(10):
            with timers["A"]:
                t += 1.0
        for i in range(3):
            with timers["B"]:
                t += 3.0
        assert timers.elapsed_time == {"A": 10.0, "B": 9.0}
        assert timers.ncalls == {"A": 10, "B": 3}

    def test_nest(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timers = StopwatchDict()
        for _ in range(10):
            with timers["A"]:
                for _ in range(3):
                    with timers["B"]:
                        t += 3.0
        assert timers.elapsed_time == {"A": 90.0, "B": 90.0}
        assert timers.ncalls == {"A": 10, "B": 30}
