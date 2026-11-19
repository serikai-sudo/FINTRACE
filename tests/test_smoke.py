import fintrace


def test_version_exists() -> None:
    assert isinstance(fintrace.__version__, str)
    assert fintrace.__version__
