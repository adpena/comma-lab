#!/usr/bin/env bash
# Render a Manim scene with the TeX toolchain on PATH (BasicTeX installs to
# /Library/TeX/texbin, which a non-login shell may not pick up via path_helper).
#
#   ./render.sh -qh scenes/scene01_separatrix.py Separatrix
#   ./render.sh -qm scenes/scene02_hardest_frame.py HardestFrame
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"
cd "$(dirname "$0")"
exec .venv/bin/manim "$@"
