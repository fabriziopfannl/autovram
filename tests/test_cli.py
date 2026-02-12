from __future__ import annotations

from typer.testing import CliRunner

from autovram.cli.app import app


def test_cli_info() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["info", "--json"])
    assert result.exit_code == 0
    assert "compute" in result.stdout.lower()


def test_cli_doctor() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "system summary" in result.stdout.lower()
