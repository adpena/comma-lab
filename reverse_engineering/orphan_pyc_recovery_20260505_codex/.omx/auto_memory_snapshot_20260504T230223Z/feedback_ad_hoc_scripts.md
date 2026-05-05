---
name: Ad hoc scripts are the root cause of integration bugs — use the canonical CLI
description: We built tac CLI, ExperimentConfig, profiles, preflight — then bypassed them all with inline scripts and shell one-liners. Every integration bug was in ad hoc code.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
We have canonical infrastructure:
- `tac` CLI with subcommands (cli.py)
- `ExperimentConfig` registry (deploy/vastai/experiments.py)
- `DistillConfig` dataclass (train_distill.py)
- Training profiles (profiles.py)
- Archive validation (submission_archive.py)
- Pipeline preflight (preflight.py)

We ALSO have 6+ ad hoc shell scripts, 10+ inline `python -c` blocks,
and 3+ remote `cat > script.sh` hacks. Every integration bug was in
the ad hoc code, not the canonical tools.

**Why:** "It's faster to write a one-off." It is — until it crashes 4 times.

**How to apply:**
- Every experiment should be an ExperimentConfig, not a shell script
- Every eval should go through `tac eval` or `build_and_eval.sh`, not inline Python
- Preflight should run automatically before any expensive operation
- The tac CLI should have `tac train`, `tac qat`, `tac pose-tto`, `tac eval` subcommands
- If you're writing `nohup python3 -u -c "..."`, STOP and add a CLI command instead
