---
name: set -euo pipefail + grep -q causes false-positive pipeline failures
description: `cmd | grep -q PATTERN` exits 0 on first match, closing the pipe early. Under set -euo pipefail, the upstream cmd's SIGPIPE-induced non-zero exit then propagates as a failed pipeline, even though grep matched. Workaround: capture cmd output first, then grep without pipeline risk.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified 2026-04-27.** Both Lane A and Lane B setup scripts failed at ffmpeg validation:
```bash
"$FFMPEG_NEW" -encoders 2>&1 | grep -qi svtav1 || {
    echo "FATAL: ffmpeg-master lacks libsvtav1" >&2
    exit 1
}
```
Even though `ffmpeg -encoders` clearly listed `libsvtav1` and a manual `grep -qi svtav1` returned exit 0, the script died with the FATAL message.

**Root cause:** Under `set -euo pipefail`:
1. `grep -q PATTERN` finds the first match and **closes the pipe**.
2. Upstream `ffmpeg` continues writing → SIGPIPE → non-zero exit.
3. `pipefail` says: "any pipeline component failed → pipeline failed."
4. So the pipeline reports non-zero exit even though grep matched.
5. `|| { exit 1; }` then fires the failure path.

This is a known bash trap with `grep -q` on commands that write a lot of output. Same trap applies to `head`, `tail -n N` (when -N is small), `awk 'cond {exit}'`, etc.

**Workarounds:**

1. **Capture output first** (preferred, idiomatic):
   ```bash
   OUT=$("$cmd" 2>&1)
   if ! echo "$OUT" | grep -q PATTERN; then echo FATAL; exit 1; fi
   ```

2. **Disable pipefail for the check**:
   ```bash
   set +o pipefail
   "$cmd" 2>&1 | grep -q PATTERN || { echo FATAL; exit 1; }
   set -o pipefail
   ```

3. **Use `grep` without -q** (consume full input):
   ```bash
   "$cmd" 2>&1 | grep PATTERN > /dev/null || { echo FATAL; exit 1; }
   ```

**How to apply:** any time you write `cmd | grep -q ...` inside a `set -euo pipefail` script, audit it. Prefer the capture-first idiom (workaround 1) — it's the most readable and avoids the SIGPIPE issue entirely.

**Cost of this trap today:** ~$0.30 of GPU spend on Lane A + Lane B that died at the validation stage before any actual work. Both fixed in `scripts/remote_setup_full.sh` via the capture-first idiom.
