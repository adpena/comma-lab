---
name: Openpilot Config Pattern
description: openpilot avoids CLI flag mismatches by sharing config via Params store + cereal pub-sub, never subprocess CLI flags. Our pipeline.py should move to shared config objects.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
openpilot avoids the subprocess CLI flag mismatch problem entirely — they never pass config via CLI flags between processes. Their architecture:
1. **Params** — persistent key-value store, all processes read at startup
2. **cereal** — Cap'n Proto pub-sub over shared memory for live config updates
3. **Manager** — orchestrates process lifecycle, doesn't pass args, processes discover config themselves

They use `dataclasses`, `functools`, `itertools` throughout — but NOT `click` (only 1 widget file). `argparse` exists only in dev tools, never in production process communication.

**Why:** Our pipeline.py constructs subprocess CLI commands that must perfectly match each target script's argparse — this is the exact fragility pattern openpilot eliminated. We've fixed 6+ bugs in this category (wrong flags, missing flags, type mismatches between pipeline.py and target scripts).

**How to apply:** Phase 2 refactor: pipeline.py should share config via Python import (shared dataclass), not subprocess string construction. Instead of `cmd = ["python", "train_distill.py", "--base-ch", str(cfg.base_ch)]`, do `from experiments.train_distill import main; main(DistillConfig(...))`. The config dataclass IS the contract. For now, test_pipeline_args.py regression tests catch these bugs.
