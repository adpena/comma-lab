# Cross-platform / cross-OS scripting — canonical reference

**Standing directive (operator 2026-07-07):** *"We should write all of our
scripts to be cross platform and cross OS by default."*

Our fleet spans **macOS** (BSD coreutils, system bash 3.2), **Linux** (GNU
coreutils — Vast/Modal/CI), and **Windows via WSL2** (bat00, a Linux
userland). A script that reaches for a GNU-only idiom silently breaks on the
macOS control host we run on every day. Default to POSIX-portable; when a
convenience is GNU-only, use the BSD-first dual form or move the logic to
Python.

Binding memory: `cross_platform_cross_os_scripts_by_default_20260707`.

## The portable idioms (copy from here)

| Need | ❌ GNU-only (breaks on macOS) | ✅ Portable |
|---|---|---|
| shebang | `#!/bin/bash` (macOS pins bash 3.2) | `#!/usr/bin/env bash` |
| file size in bytes | `stat -c%s "$f"` | `stat -f%z "$f" 2>/dev/null \|\| stat -c%s "$f"` |
| mtime | `stat -c%Y "$f"` | `stat -f%m "$f" 2>/dev/null \|\| stat -c%Y "$f"` |
| in-place edit | `sed -i 's/a/b/' f` (BSD needs `-i ''`) | write tmp + `mv`, or do it in Python |
| show non-printing chars | `cat -A f` | `sed -n l f` |
| resolve abs path | `readlink -f "$p"` | `(cd "$(dirname "$p")" && printf '%s/%s\n' "$(pwd)" "$(basename "$p")")` |
| listing with fixed time | `ls --time-style=…` | `stat` per file, or Python `os.stat` |
| pattern search (PCRE) | `grep -P` | `grep -E` (ERE is portable) |
| version sort | `sort -V` | Python `packaging.version`, or accept lexicographic |
| date arithmetic | `date -d '…'` | Python `datetime` |
| array-from-lines | `mapfile`/`readarray` | `while IFS= read -r line; do …; done` |
| find + format | `find … -printf` | `find … -print0 \| while …`, or Python `os.walk` |

## Rules of thumb

1. **Prefer Python over shell** for anything beyond a few piped commands.
   `pathlib`/`os`/`shutil`/`subprocess` are the most portable layer we own, and
   they carry the determinism spine the project already binds.
2. **Shebang `#!/usr/bin/env bash`**, and if you rely on bash ≥ 4 features
   (associative arrays, `${x^^}`), guard with a version check — macOS system
   bash is 3.2. Most of our scripts don't need them; keep it that way.
3. **`set -euo pipefail`** is fine (shebang is bash). Do NOT assume `pipefail`
   under a `#!/bin/sh` shebang — it is not POSIX.
4. **Every `stat`/`sed -i`/`readlink` reflex → the BSD-first dual form above**,
   or Python.
5. **Paths:** `.venv/bin/python` is Unix-venv layout; Windows-native venv is
   `.venv/Scripts/python.exe`. WSL2 is Unix so the fleet is covered — but if a
   script must run on Windows-native (no WSL), branch on the OS.
6. **Tailscale addressing only** for remote nodes (per CLAUDE.md fleet rule) —
   never raw LAN IPs/hostnames, which are also not portable across networks.

## Known debt (2026-07-07 audit)

~145 checked-in `.sh` files use at least one GNU idiom; most `remote_lane_*.sh`
already carry the `stat -c … || stat -f …` dual form (correct). A bounded
sweep to normalize the remainder (bare `stat -c`, `#!/bin/bash` shebangs) is a
named follow-up, not a silent gap. New scripts must be born portable per the
table above.
