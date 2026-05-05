---
name: EVERY script crashes on first run — always smoke test before committing to long runs
description: qat_finetune.py, train on Vast.ai, inflate_renderer.py, auth_eval.py — ALL crashed on first execution. Wrong args, wrong formats, wrong paths, wrong dtypes. ALWAYS run a 5-10 epoch smoke test.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Pattern observed across the ENTIRE session (2026-04-21/22):

1. auth_eval: wrong constructor arg (upstream_root vs upstream_dir)
2. inflate_renderer.py: --device flag doesn't exist
3. train_distill.py: --checkpoint "" → torch.load("") → FileNotFoundError
4. qat_finetune.py: torch.load on ASYM .bin → UnpicklingError
5. inflate_renderer.py: video_names.txt had "0" not "0.mkv"
6. mask encoding: int8 * 63 overflow

**Why:** We write code, review it, verify syntax, even fix 8 bugs from
adversarial review — but we don't RUN it until we need results.

**How to apply:**
- ALWAYS run a 5-10 epoch smoke test before launching any long run
- ALWAYS test the full e2e pipeline (train→export→inflate→evaluate) locally
  before deploying to Vast.ai
- The smoke test is the FIRST action after writing, not the last
- "Syntax OK" and "review passed" are NOT the same as "it runs"
