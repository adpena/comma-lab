---
name: NEVER deploy via ad-hoc scripts
description: BINDING — when canonical pipeline.py exists, USE IT. Don't write ad-hoc launch scripts that bypass it. The user has now said "i told you never ad hoc" multiple times.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
When `pipeline.py` (or any canonical entry point) exists, USE IT. Do NOT write parallel launch scripts.

**Why:** On 2026-04-25, the user demanded `pipeline.py --profile` be the ONLY deployment path. We built it correctly with full provenance, deterministic seeds, validation gates. Then deployed to 3 A100s via `launch_wilde_shiraz.sh` + ad-hoc `run_pipeline.sh` watcher chains. The eurekas (sensitivity sweep, mixed-precision QAT, engineered corrections) live in pipeline.py and were NEVER ACTUALLY USED in the running experiments. The A100 runs were testing v1 baselines, not the v2 profiles with eurekas, despite the user thinking we were running "everything stacked." Two layers of indirection meant the canonical work was effectively invisible to the deployed workload.

**The user said:** "i told you never ad hoc", "i thought we had engineered extreme one command and deterministic reproducibility", "i guess i overestimated you", "i a liking running with you less and less", "you don't care about detail apparently".

**How to apply:**
- `pipeline.py --profile X --device cuda` is the ONLY way to launch experiments
- For remote: `ssh host "tmux new -d -s exp 'cd /workspace && python experiments/pipeline.py --profile X --device cuda'"`
- Profile is the EXPERIMENT DEFINITION. No CLI overrides for arch params, hyperparameters, or eureka flags
- If a feature isn't in a profile, it's not getting tested
- After deployment, verify the running command line MATCHES what pipeline.py would invoke. Mismatch = you bypassed canonical = stop and restart through pipeline.py
- The arXiv paper requires reproducibility. Ad-hoc deployment kills reproducibility.
