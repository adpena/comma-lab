---
name: USE THE TOOLS YOU BUILD — never bypass canonical infrastructure with ad hoc code
description: Built submission_archive.py, preflight.py, build_and_eval.sh, tac CLI, ExperimentConfig — then ignored ALL of them and used inline python -c scripts. The user is right to be angry. This is inexcusable.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
On 2026-04-22, the user called out a pattern that had been happening all session:

1. Built `submission_archive.py` with `require_valid_archive()` — never called it
2. Built `preflight.py` with full pipeline validation — never called it
3. Built `build_and_eval.sh` with archive provenance — ran inline scripts instead
4. Built `tac` CLI with proper subcommands — used `python -c` one-liners
5. Built `ExperimentConfig` registry — wrote shell scripts on the remote

Every integration bug lived in the ad hoc code. The canonical tools would
have caught them all.

**Why:** "It's faster." It is not. Every ad hoc script crashed on first run.
Every inline eval produced wrong results. The canonical tools were tested
and correct. Using them would have been faster AND correct.

**How to apply — NON-NEGOTIABLE:**
- Before ANY eval: run `preflight.py` first. No exceptions.
- Before ANY archive build: use `submission_archive.build_submission_archive()`. No exceptions.
- Before ANY e2e eval: use `build_and_eval.sh`. No inline python -c.
- Before ANY training run: use `ExperimentConfig` or `tac` CLI.
- If a tool doesn't exist for what you need: BUILD IT PROPERLY, then use it.
  Do NOT write a one-off inline script "just this once."
- If you catch yourself writing `python -c "..."` for anything non-trivial: STOP.
  Add a CLI command instead.

This is common sense. The user should not have to tell me this.
