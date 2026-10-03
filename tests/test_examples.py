# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The scripts in ``examples/`` run, so that what the documentation shows keeps working."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
SCRIPTS = sorted(EXAMPLES.glob("0*.py"))


def test_there_is_an_example_for_each_topic() -> None:
    assert [script.stem[:2] for script in SCRIPTS] == [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
    ]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.stem)
def test_an_example_runs_and_says_something(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()
