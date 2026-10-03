# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Entry point of ``python -m x3pio``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
