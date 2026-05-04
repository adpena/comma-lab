---
name: pose_dim DEAD across SHIRAZ/DEN/WILDE/GREEN training pre-2026-04-27
description: 2026-04-27: train_renderer.py had `pose_dim=getattr(args, "pose_dim", 0)` at the build site but parse_args NEVER copied profile.pose_dim into the Namespace. Result: every prior renderer training run that declared `pose_dim=6` in its profile silently trained `pose_dim=0` — FiLM conditioning was DEAD. Fixed in commit 0746a803 alongside the Lane D work.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Discovered 2026-04-27 by Lane D subagent during the half-frame retrain implementation.**

**The bug:**
- `src/tac/experiments/train_renderer.py` build site read `pose_dim` via `getattr(args, "pose_dim", 0)`
- `parse_args()` never resolved the profile's `pose_dim` field into the argparse `Namespace`
- Profiles SHIRAZ, DEN, WILDE, GREEN all declared `pose_dim=6` in their config dicts
- Actual training silently used `pose_dim=0` → no FiLM conditioning layer instantiated
- FiLM was a stated design feature in those profiles but was never active

**Why it went undetected:**
- No assertion that profile values landed in args
- No checkpoint header that exposed the actual instantiated arch params
- No regression test introspecting parse_args output against a profile spec
- Same dead-resolver class as the 2026-04-26 dead-flag-wiring bug (`feedback_dead_flag_wiring_pattern`) and the same class as the Yousfi-5 uncertainty loss never-fired bug

**Implication for past results:**
- Any "best score" attributed to SHIRAZ / DEN / WILDE / GREEN pre-2026-04-27 was achieved WITHOUT FiLM
- The profile's claim of FiLM-conditioned rendering is NOT what got trained
- Saved checkpoints from those runs have pose_dim=0 architecture; loading them with code expecting pose_dim=6 will produce shape mismatches OR silent zero embeddings

**How to apply going forward:**
1. **Before re-deriving any past SHIRAZ/DEN/WILDE/GREEN finding,** check whether the result depended on FiLM working. If yes, the result is INVALID and needs a re-run.
2. The dilated-h64 0.9001 baseline appears unaffected (its profile likely set `pose_dim=0` — verify before assuming).
3. **Do NOT mix old (no-FiLM) checkpoints with new (FiLM-on) inflate code** — the renderer.bin and the inflate path must have matching pose_dim.
4. **The fix's regression test** (`test_parse_args_resolves_mask_half_sim_prob_from_profile` and equivalents) is the new canonical pattern: every profile-key-to-args resolution must be assertable from outside.
5. When auditing other profiles (e.g., for QAT, FP4, post-filters), grep parse_args for every key declared in the profile dict. Anything not resolved is a candidate dead-default.

**Cost of this trap:** unknown but likely large — months of SHIRAZ/DEN/WILDE/GREEN training without the architectural feature the profile claimed. Many "FiLM didn't help" conclusions in run_log were actually "FiLM was never tested."

**Same-class bugs to scan next:** `seg_kl_temperature`, `qat_lr`, `pose_warp_steps`, any profile field that has a `getattr(args, ..., DEFAULT)` build-site reference but no parse_args resolver.
