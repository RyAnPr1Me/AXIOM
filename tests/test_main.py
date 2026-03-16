"""Tests for the AXIOM main module."""

from axiom import __version__
from axiom.main import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_main(capsys) -> None:
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "AXIOM"
