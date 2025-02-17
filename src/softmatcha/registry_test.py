import pytest

from . import registry


def test_registry():
    register, get_cls = registry.setup("test")
    REGISTRY = registry.REGISTRIES["test"]
    assert "mock1" not in REGISTRY

    @register("mock1")
    class MockClass1: ...

    assert "mock1" in REGISTRY
    assert issubclass(REGISTRY["mock1"], MockClass1)
    assert issubclass(get_cls("mock1"), MockClass1)

    @register("mock2")
    class MockClass2: ...

    assert "mock2" in REGISTRY
    assert issubclass(REGISTRY["mock2"], MockClass2)
    assert issubclass(get_cls("mock2"), MockClass2)

    assert registry.get_registry("test") == {"mock1": MockClass1, "mock2": MockClass2}

    with pytest.raises(ValueError):

        @register("mock1")
        class MockClass3: ...
