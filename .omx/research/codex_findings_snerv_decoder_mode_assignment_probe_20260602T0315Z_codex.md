# Codex Findings: SNeRV Decoder Mode Assignment Probe

UTC: 2026-06-02T03:15Z
Lane: snerv_decoder_mode_assignment_probe_20260602
Axis: [macOS-CPU advisory]
Artifact: .omx/research/snerv_decoder_mode_assignment_probe_1pair_20260602T0315Z.json

## Verdict

NO-GO for promotion or exact dispatch. GO for local SNeRV fit/rate triage.

This landing adds a receiver-decoded mixed decoder mode assignment probe for the
SNeRV stack. It compares the existing magnitude heuristic against explicit
per-kernel decoder modes while keeping the receiver archive parser and scoring
surface single-sourced through `run_snerv_advisory`.

The probe is intentionally fail-closed:

- `score_claim=false`
- `frontier_score_claim=false`
- `rank_or_kill_eligible=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `exact_or_full_video_launched=false`

## Empirical Smoke

Command:

```bash
.venv/bin/python tools/probe_snerv_decoder_mode_assignments.py \
  --n-pairs 1 \
  --levels 1 \
  --bits-per-coeff 2.0 \
  --step-map-coder-bins 4 \
  --mode-plan magnitude_heuristic \
  --mode-plan fp16,int4,int4 \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --out .omx/research/snerv_decoder_mode_assignment_probe_1pair_20260602T0315Z.json
```

Result:

- Best advisory plan: `explicit_int42_fp161`
- Best advisory `score_linf`: 2.9824514626437555
- Magnitude heuristic: `score_linf=3.616345827198948`, modes `{fp16: 2, int4: 1}`
- Explicit `fp16,int4,int4`: `score_linf=2.9824514626437555`, modes `{fp16: 1, int4: 2}`
- Both candidates had `receiver_archive_replay_verified=true`
- Explicit plan reduced decoder payload bytes from 45 to 34 and decoder section bytes from 884 to 862 on this 1-pair smoke.

Interpretation: explicit mixed precision can matter even at tiny scale, but this
is a local advisory signal only. It is not a proof of pair-robust detector fit.

## Fresh-Eyes Sidecar

A read-only xhigh fresh-eyes agent independently flagged the same operating
frame:

- Main is the only production truth; stale lane checkouts must not drive
  frontier decisions.
- Current SNeRV/PACT-VQ work is fit-limited, not authority-leaky.
- PR101 CPU claim state still needs reconciliation before further exact work.
- Existing SNeRV package guards are healthy and fail closed.

No files were edited by the sidecar.

## Verification

```bash
.venv/bin/ruff check \
  src/tac/analysis/snerv_decoder_mode_assignment_probe.py \
  tools/probe_snerv_decoder_mode_assignments.py \
  src/tac/tests/test_snerv_decoder_mode_assignment_probe.py

.venv/bin/python -m pytest src/tac/tests/test_snerv_decoder_mode_assignment_probe.py
```

Both passed.

Lane registry mutation was performed through `tools/lane_maturity.py`.
`tools/lane_maturity.py validate` still fails on 104 pre-existing missing
historical evidence paths; the new lane entries themselves were present and
coherent after re-check.

## Remaining Blockers

- Full-600 byte-closed receiver proof missing.
- Paired contest CPU/CUDA auth eval missing.
- Contest archive zip packaging missing.
- PR101 storage-order CPU recovery remains pending and continues to block new
  full-video/exact/CUDA launches.
- The probe does not solve SNeRV detector fit; it only gives a reusable local
  way to compare explicit decoder protection plans.

## Next Step

Use this probe as the local receiver-decoded scorer-loop for broader bounded
mode-plan searches, then pivot from hand-written plans to learned/nonlinear
decoder QAT if the pair-robust contraction keeps collapsing.
