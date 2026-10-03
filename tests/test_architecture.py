# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The layers of the package depend on each other in one direction only.

The model (``x3pio.model``) knows nothing of how a file is stored. The codec (``x3pio.codec``)
turns bytes into the types of the model and back. The application and the command line use both.
The file types read and write through a codec that the package registers in the port of the model,
so the model imports the codec nowhere, not even inside of a function.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import x3pio

PACKAGE = Path(x3pio.__file__).parent

#: What a layer must not import when the module is loaded.
FORBIDDEN = {
    "model": ("x3pio.codec", "x3pio.application", "x3pio.cli"),
    "codec": ("x3pio.application", "x3pio.cli"),
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _loaded_imports(path: Path) -> set[str]:
    """Return the modules that importing ``path`` imports, not those imported by a call."""
    package = (
        _module_name(path).rsplit(".", 1)[0] if path.name != "__init__.py" else _module_name(path)
    )
    found: set[str] = set()

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                    base = ".".join([*parts, base] if base else parts)
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.If) and "TYPE_CHECKING" not in ast.unparse(node.test):
                visit(node.body)  # an import for the type checker only is not a dependency
                visit(node.orelse)

    visit(ast.parse(path.read_text(encoding="utf-8")).body)
    return found


@pytest.mark.parametrize("layer", list(FORBIDDEN))
def test_a_layer_does_not_import_the_layers_above_it(layer: str) -> None:
    modules = sorted((PACKAGE / layer).glob("*.py"))
    assert modules
    for path in modules:
        imported = _loaded_imports(path)
        for name in imported:
            assert not name.startswith(FORBIDDEN[layer]), f"{path.name} imports {name}"


def _every_import(path: Path) -> set[str]:
    """Return the modules that ``path`` imports anywhere, also inside of functions."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            if node.module in (
                None,
                "x3pio",
            ):  # the names may be modules, as in ``from .. import codec``
                found.update(f"{base}.{alias.name}" for alias in node.names)
            else:
                found.add(base)
    return found


def test_the_model_imports_the_codec_nowhere() -> None:
    for path in sorted((PACKAGE / "model").glob("*.py")):
        for name in _every_import(path):
            assert "codec" not in name.split("."), f"{path.name} imports {name}"


def test_the_package_registers_the_codec_for_the_file_types() -> None:
    from x3pio.codec import container
    from x3pio.model import codec_port

    assert codec_port.codec() is container


def test_the_checker_sees_an_import_of_a_layer() -> None:
    # The same check on a module that does import the codec: the application.
    assert any(
        name.startswith("x3pio.model") for name in _loaded_imports(PACKAGE / "application.py")
    )
    assert any(
        name.startswith("x3pio.model")
        for name in _loaded_imports(PACKAGE / "codec" / "container.py")
    )
    assert not any(
        name.startswith("x3pio.codec") for name in _loaded_imports(PACKAGE / "model" / "files.py")
    )


def test_the_model_needs_nothing_but_numpy() -> None:
    allowed = ("x3pio", "numpy", "__future__")
    stdlib = set(__import__("sys").stdlib_module_names)
    for path in sorted((PACKAGE / "model").glob("*.py")):
        for name in _loaded_imports(path):
            root = name.split(".")[0]
            assert root in stdlib or name.startswith(allowed), f"{path.name} imports {name}"
