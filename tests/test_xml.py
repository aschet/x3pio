# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from helpers import build, unzip
from x3pio import X3pFormatError
from x3pio.codec.xml import NAMESPACE, parse_datums


def test_parse_datums_reads_columns_and_missing_points() -> None:
    values = parse_datums(["1;2;3", "", "4e-6;-5;+6.5"], 3)
    assert values.shape == (3, 3)
    assert values[0].tolist() == [1.0, 2.0, 3.0]
    assert values[1].tolist() != values[1].tolist()  # NaN
    assert values[2].tolist() == [4e-6, -5.0, 6.5]


def test_parse_datums_is_lenient_about_numbers() -> None:
    values = parse_datums(["1.", " 2 ", "3"], 1)
    assert values[:, 0].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(("text", "columns"), [("1;2", 1), ("1", 2), ("1;;2", 3), ("a", 1)])
def test_parse_datums_rejects_wrong_content(text: str, columns: int) -> None:
    with pytest.raises(X3pFormatError):
        parse_datums([text], columns)


def test_written_document_layout() -> None:
    text = unzip(build())["main.xml"].decode("utf-8")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<p:ISO5436_2 ')
    assert f'xmlns:p="{NAMESPACE}"' in text
    assert f'xsi:schemaLocation="{NAMESPACE} {NAMESPACE}/ISO5436_2.xsd"' in text
    assert "<Revision>ISO25178-72:2017/DAM1</Revision>" in text
    assert text.endswith("</p:ISO5436_2>\n")


def test_numbers_are_written_with_a_round_tripping_representation() -> None:
    text = unzip(build())["main.xml"].decode("utf-8")
    assert "<Increment>1e-06</Increment>" in text
    assert "<Increment>2e-06</Increment>" in text
