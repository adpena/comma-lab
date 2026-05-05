---
name: 2026-04-26 SHIRAZ pose loader + self-match deadlock
description: Two bug classes that burned 21h of A100 idle time. Permanent prevention via content-detecting pose loader + cross-language preflight scanner.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Two bug classes detected on 2026-04-26 cost ~21 hours of fully-paid A100 time:

**Bug 1 — Suffix-based pose loader.** A wrapper renamed `optimized_poses_partial.pt` (torch.save pickle) → `optimized_poses.bin` (raw fp16). `auth_eval_renderer.py` called `torch.frombuffer(..., dtype=float16).reshape(-1, 6)` based on suffix alone and crashed after 7 min of mask extraction with `RuntimeError: shape '[-1, 6]' is invalid for input of size 7862`.

**Bug 2 — Self-matching `pgrep -f TOKEN` wait loop.** `bash -c "while pgrep -f train_distill > /dev/null; do sleep 60; done; bash run_pipeline.sh"` matched its OWN bash -c argv (the string contains "train_distill"), looped forever, GPU sat idle.

**Why:** Suffix dispatch is fragile because wrappers blindly `cp foo.pt foo.bin`. `pgrep -f` is fragile because the wait loop's argv is also a process command line. Both look correct in code review.

**How to apply:**
- Use `tac.submission_archive.load_optimized_poses(path, pose_dim=N, expected_n_pairs=600)` for ALL pose loading. Content-detects pickle vs raw, validates count.
- NEVER `cp .pt .bin` — convert via `save_poses_binary()` or have producer emit `.bin` directly.
- NEVER `pgrep -f X` for synchronization. Use pidfile, `pgrep -x <executable>` (exact name), or unique cookie.
- Preflight at `src/tac/preflight.py:_scan_text_for_dangerous_patterns()` catches all 3 patterns in both bash files AND Python f-strings/string constants.
- 19 property tests in `src/tac/tests/test_pose_loader_and_preflight.py` pin the contract.

Layered defense: producer emits canonical .bin + meta sidecar; consumer validates by content; codebase scanner catches the patterns at the source. Each layer alone would catch the bug.
