"""Sanity checks for the runtime package bootstrap."""


def test_runtime_package_is_importable() -> None:
    import resilient_ai_runtime

    assert resilient_ai_runtime.__doc__
