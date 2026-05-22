# Codex Findings: Recovered Results Config And Synergy Audit

timestamp_utc: 2026-05-22T05:01:51Z
lane_id: lane_codex_modal_recovery_config_synergy_audit_20260522
author: codex
verdict: PROCEED_WITH_CONFIG_FIXES

## Scope

Read-only adversarial audit of harvested Modal results and recent local
artifacts for engineering bugs, non-authoritative score traps, and useful
stacking/synergy signals.

## Findings

### HFV9 CPU/CUDA Pair

HFV9 has clean paired exact-eval evidence but is not promotional:

- CPU exact eval: `0.32067828057415293`
- CUDA exact eval: `0.33713201858942626`
- archive bytes: `178553`
- archive SHA: `9a32b1311da1076b1659ff6652481383527279905c8a135eeedff6ec913888ac`
- runtime content tree: `20a96c8c...`

The method is a hard regression. Use it only as an axis/custody sanity pair.

Custody cleanup: top-level `contest_auth_eval.json` carries
`archive_size_bytes` but not top-level `archive_sha256`; consumers must descend
into provenance. The CPU provenance also records `gpu_model` as a
`FileNotFoundError` string. Normalize these in future harvest schemas so
archive identity is not hidden behind nested provenance.

### A100 Grayscale LUT

This was a training/export pipeline failure, not a method score:

- Modal wrapper timed out after `14400.84s` with rc `124`.
- No OOM signal was found.
- `best.pt` exists and loads cleanly.
- Best validation occurred at epoch 30 (`best_val_lagrangian=12.175071716308594`).
- The run continued to epoch 550 and validation had degraded to about `93.996147`.
- No `0.bin`, `archive.zip`, submission dir, auth-eval payload, or inflate
  manifest was emitted.

Highest-confidence bug: the pipeline lacks export-on-best and early-stop
custody. It burned almost the full 4h wrapper after the best checkpoint, then
lost byte-closed archive custody.

Next action: build/export from recovered `best.pt` in a new custody directory,
run archive/inflate smoke and cheap advisory scoring, then dispatch exact
`[contest-CUDA]` only if the advisory result is not dead. A rerun should use
early stop or a 40-60 epoch cap, not 2000 epochs under a 4h timeout.

### NSCS06 v8 Modal Retries

The NSCS06 v8 artifacts are config/guard failures, not substrate evidence:

- rc22 runs failed immediately because full mode was still blocked as an L0
  scaffold: only smoke was supported and `_full_main` was unavailable.
- a later rc1 accepted `full` but invoked CPU full training and failed the
  trainer guard: `--device cpu is permitted only with --smoke`.

Next action: fix dispatch/config so Modal full mode implies CUDA device, and
add a pre-dispatch guard rejecting `mode=full` with CPU. Retry once after the
guard lands.

### DP1

DP1 produced the most useful negative anchor:

- later training retries produced byte-smaller archives:
  baseline `25730` bytes, procedural `18298` bytes;
- paired auth-eval memo records baseline `4.25` vs procedural `90.33`;
- the `-7432` archive-byte saving is overwhelmed by distortion.

Verdict: naive post-training procedural codebook substitution is blocked.
Useful next work is either score-opacity audit or joint retraining with the
procedural seed present during training.

Packaging bug: refire auth-eval directories failed because the runtime imported
`tac.substrates.pretrained_driving_prior.inflate`, but that module was missing
from the shipped runtime. Fix runtime closure before using DP1 refires as
durable contest JSON custody.

### Harvest Summary Authority

`experiments/results/_modal_harvest_summary.json` is a recovery index, not score
authority. Several rows have null lane/out-dir/score/score-claim fields. Any
promotion/ranking decision must descend into per-call artifacts and auth-eval
payloads.

## Prioritized Actions

1. Add/export grayscale LUT `best.pt` to a byte-closed archive in a fresh
   custody directory; run archive/inflate smoke and cheap advisory score.
2. Add export-on-best and early-stop/cap controls to the grayscale LUT training
   pipeline before another paid run.
3. Add a pre-dispatch NSCS06 guard for `mode=full` plus CPU, and fix Modal full
   mode to use CUDA.
4. Fix DP1 runtime closure so auth-eval refires include
   `src/tac/substrates/pretrained_driving_prior/inflate.py`.
5. Normalize auth-eval top-level archive SHA custody where schemas currently
   require consumers to descend into nested provenance.

## Non-Authority

This memo does not promote any recovered result. It separates recovery,
configuration, and score authority so harvested artifacts can guide the next
engineering tranche without becoming false leaderboard evidence.
