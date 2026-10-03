# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import x3pio
from helpers import SURFACE, edit_main, stacked, sub
from x3pio import DataStorage, DataType, X3pFile
from x3pio.cli import main


def _write(path: Path, **kwargs: object) -> Path:
    x3pio.write(path, SURFACE, x_scale=1e-6, y_scale=2e-6, **kwargs)  # type: ignore[arg-type]
    return path


def test_cli_info_prints_header(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path / "surface.x3p")

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "FeatureType     = SUR" in output
    assert "SizeX           = 3" in output
    assert "SizeY           = 2" in output
    assert "SizeZ           = 1" in output
    assert "CX              = incremental, Increment=1e-06, Offset=0" in output
    assert "CZ              = absolute, float64, Increment=1, Offset=0" in output
    assert "Manufacturer    = x3pio" in output
    assert "Corrections" not in output


def test_cli_info_prints_the_storage_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["info", str(_write(tmp_path / "a.x3p", storage=DataStorage.XML))]) == 0
    assert "Storage         = xml" in capsys.readouterr().out
    assert main(["info", str(_write(tmp_path / "b.x3p"))]) == 0
    assert "Storage         = binary" in capsys.readouterr().out


def test_cli_info_prints_a_point_cloud(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "cloud.x3p"
    x3pio.write_points(path, np.zeros((5, 3)))

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "FeatureType     = PCL" in output
    assert "ListDimension   = 5" in output
    assert "SizeX" not in output


def test_cli_info_prints_rotation_extensions_and_missing_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    x3p = X3pFile.loads(_write(tmp_path / "a.x3p").read_bytes())
    x3p.metadata = None
    x3p = x3p.with_placement(x3pio.Placement.from_axis_angle([0.0, 0.0, 1.0], np.pi))
    x3p.extensions["http://www.vendor.com/mypath/ext.xml"] = b"1"
    x3p.save(tmp_path / "b.x3p")

    assert main(["info", str(tmp_path / "b.x3p")]) == 0

    output = capsys.readouterr().out
    assert re.search(r"^Rotation +\= -1 \S+ 0 \S+ -1 0 0 0 1$", output, re.MULTILINE)
    assert re.search(r"^Record2 +\= \(none\)$", output, re.MULTILINE)
    assert "= http://www.vendor.com/mypath/ext.xml" in output


def test_cli_info_prints_not_recorded_for_unusable_dates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "a.x3p", storage=DataStorage.XML)
    path = tmp_path / "placeholder.x3p"
    path.write_bytes(edit_main(source.read_bytes(), sub(r"<Date>[^<]*</Date>", "<Date>N/A</Date>")))

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert re.search(r"^Date +\= \(not recorded\)$", output, re.MULTILINE)
    assert re.search(r"^CalibrationDate +\= \(not recorded\)$", output, re.MULTILINE)


def test_cli_convert_changes_the_storage_form(tmp_path: Path) -> None:
    source = _write(tmp_path / "source.x3p", storage=DataStorage.XML)
    destination = tmp_path / "destination.x3p"

    assert main(["convert", "-s", "binary", str(source), str(destination)]) == 0

    converted = x3pio.read(destination)
    assert converted.storage is DataStorage.BINARY
    assert np.array_equal(stacked(converted), stacked(x3pio.read(source)), equal_nan=True)


def test_cli_convert_keeps_the_storage_form_if_none_is_given(tmp_path: Path) -> None:
    source = _write(tmp_path / "source.x3p", storage=DataStorage.XML)
    destination = tmp_path / "destination.x3p"

    assert main(["convert", str(source), str(destination)]) == 0

    assert x3pio.read(destination).storage is DataStorage.XML


def test_cli_convert_changes_the_z_type(tmp_path: Path) -> None:
    source = _write(tmp_path / "source.x3p")
    destination = tmp_path / "destination.x3p"

    assert main(["convert", "-t", "int16", str(source), str(destination)]) == 0

    converted = x3pio.read(destination)
    assert converted.header.z.data_type is DataType.INT16
    np.testing.assert_allclose(
        stacked(converted),
        stacked(x3pio.read(source)),
        atol=converted.header.z.increment,
        equal_nan=True,
    )


def test_cli_convert_repairs_a_file_with_deviations(tmp_path: Path) -> None:
    sample = Path(__file__).parent / "interop_samples" / "opengps" / "ISO5436-sample3.x3p"
    destination = tmp_path / "destination.x3p"

    assert main(["convert", str(sample), str(destination)]) == 0


def test_cli_convert_fails_on_metadata_that_cannot_be_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blob = edit_main(
        (_write(tmp_path / "a.x3p", storage=DataStorage.XML)).read_bytes(),
        sub(r"<Date>[^<]*</Date>", "<Date>N/A</Date>"),
    )
    source = tmp_path / "placeholder.x3p"
    source.write_bytes(blob)
    destination = tmp_path / "destination.x3p"

    assert main(["convert", str(source), str(destination)]) == 1
    assert "no date" in capsys.readouterr().err
    assert not destination.exists()

    assert main(["convert", "--fix", str(source), str(destination)]) == 0
    assert x3pio.read(destination).metadata is None


def test_cli_reports_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", str(tmp_path / "missing.x3p")]) == 1
    assert "x3pio: error:" in capsys.readouterr().err


def test_cli_reports_a_file_that_is_not_x3p(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.x3p"
    path.write_bytes(b"not a zip")
    assert main(["info", str(path)]) == 1
    assert "not a zip archive" in capsys.readouterr().err


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == f"x3pio {x3pio.__version__}"


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_module_is_runnable(tmp_path: Path) -> None:
    path = _write(tmp_path / "surface.x3p")
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "x3pio", "info", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "SizeX" in result.stdout
