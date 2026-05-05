# Artifact recovery is canonical — recover BEFORE destroy

**2026-04-28 · binding non-negotiable**

## The bug class

Lane RM-d crashed at the auth-eval stage AFTER 3.5h of training on a Vast.ai
RTX 4090. The canonical fail-loud destroy path fired immediately — wiping the
trained renderer, masks, optimized poses, and run.log. $1.16 burned with
ZERO recoverable artifacts. The trained model was probably worth running auth
eval against locally / on Modal — we'll never know.

This is NOT a one-off. Every time `destroy_instance()` fires after an
auth-eval crash (NVDEC bad host AFTER training, OOM at archive build, etc.)
we lose the work product. Multiply by the lanes destroyed today:

| Lane | Cost burned | Recoverable artifact lost? |
|------|-------------|------------------------------|
| Lane RM-d | $1.16 | YES — trained renderer + masks + poses |
| Lane SAUG-V2 (codex DNS) | $0.50 | N/A — never trained |
| Lane W (Iceland) | $0.30 | N/A — never started |
| ~5 zombies before launcher V4 | $0.30 | N/A — boot failures |

Total recovery-eligible loss today: ~$1.16. Multiplied across the next 30 days
of lanes, this is real money.

## The canonical fix

**`tools/recover_lane_artifacts.py`** wraps SSH/SCP to pull every
archive-relevant file from the remote BEFORE destroying. Wired into
`scripts/launch_lane_on_vastai.py:destroy_instance()` as the default path.

```python
def destroy_instance(instance_id, *, recover=True, lane_label=None,
                     recovery_timeout_s=600):
    if recover:
        recover_before_destroy(instance_id, lane_label, ...)
    # then vastai destroy
```

Patterns recovered (`tools/recover_lane_artifacts.RECOVERY_PATTERNS`):
* `renderer.bin*`, `masks.mkv`, `optimized_poses.pt*`
* `archive*.zip`, `*best*.pt`, `checkpoint*.pt`
* `run.log`, `train.log`, `setup.log`, `heartbeat.log`, `provenance.json`
* `RESULT_JSON*.json`, `auth_eval*.json`, `report.txt`

Recovery is best-effort: tight per-call timeouts (30s SSH, 5min SCP, 10min
overall). Every call is wrapped to NEVER block destroy.

The `--no-recover` opt-out exists for instances known-unreachable (NVDEC bad
boot, network split). Default is recover-first.

## Local follow-up

`tools/auth_eval_local.py` accepts the recovery dir and runs the canonical
contest_auth_eval pipeline locally:

```bash
python tools/recover_lane_artifacts.py 12345 --lane-label lane_rm_d
python tools/auth_eval_local.py \
    --archive-dir experiments/results/recovered_12345_lane_rm_d/workspace
```

Or push to Modal:

```bash
modal run experiments/modal_auth_eval.py \
    --archive experiments/results/recovered_12345_lane_rm_d/workspace/archive.zip
```

## Tests

* `src/tac/tests/test_recover_lane_artifacts.py` — 13 tests covering SSH
  unreachable, ssh-CLI missing, happy path, classification, deadline cutoff.
* `src/tac/tests/test_auth_eval_local.py` — 10 tests covering archive
  packing determinism, GT video presence, F5 config.env guard.

## Workflow change

Before today: destroy fires → artifacts lost → operator pays for next
training run from scratch.

After today: destroy fires → recover SSH/SCP runs (~30s-5min) →
`experiments/results/recovered_<id>_<label>/recovery_metadata.json` records
what was found → operator can run auth_eval_local OR modal_auth_eval against
the same bytes that were on the remote.

## Key insight

The launcher's `destroy_instance()` was acting as a strict fail-loud guard,
but that's the WRONG default for cost paranoia. The right default is
"recover what you can, then destroy" — fail-loud for the destroy itself,
best-effort for the recovery. The two concerns are orthogonal.
