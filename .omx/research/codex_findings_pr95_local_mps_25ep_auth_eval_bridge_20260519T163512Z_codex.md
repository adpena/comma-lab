# Codex Findings: PR95 Local MPS Training Auth-Eval Bridge

UTC: 2026-05-19T16:35:12Z
Agent: Codex
Evidence grade: `[macOS-CPU advisory]` for the auth-eval run; no score claim.

## Finding

The PR95 public-intake HNeRV source stack is trainable on local Apple MPS with
`PYTORCH_ENABLE_MPS_FALLBACK=0`, and the same emitted weights/archive replay
through `experiments/contest_auth_eval.py` gives the same/comparable score as
the training-side PR95 score computation.

This establishes a useful authority bridge:

- Local MPS is still not a contest axis and remains non-promotable.
- But PR95's training-side score is predictive enough to use as a cheap
  convergence and curriculum-search signal when the exact same emitted archive
  is later replayed through byte-closed auth eval.
- The promotion boundary remains paired `[contest-CPU]` and `[contest-CUDA]`
  exact auth eval on contest-compliant hardware/runtime.

## Source And Custody

PR95 source root:
`experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon`

Public archive custody anchor:

- path: `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/archive.zip`
- bytes: `178417`
- sha256: `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`

Local run:

- path: `experiments/results/pr95_local_mps_source_faithful_stage1_25ep_20260519T161718Z`
- seed: `1234`
- device requested/selected: `mps` / `mps`
- fallback: `PYTORCH_ENABLE_MPS_FALLBACK=0`
- source tree sha256: `11a18e07427f57e3e9ac963902c2a21083eca62fcd52a949aa66406dff3ae2db`

## Training Probe Result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python tools/run_pr95_local_training_probe.py \
  --device mps \
  --stage-epochs 1=25 \
  --eval-every 5 \
  --output-dir experiments/results/pr95_local_mps_source_faithful_stage1_25ep_20260519T161718Z
```

Result:

- best ep: `25`
- training-side best score: `1.8250676379821704`
- payload bytes: `228423`
- wall seconds: `450.53339295799924`
- authority flags: `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`, `rank_or_kill_eligible=false`

Convergence checkpoints:

- ep5: `41.8828`
- ep10: `5.3380`
- ep15: `3.0913`
- ep20: `2.2801`
- ep25: `1.8251`

## Packaging

The codec re-emission produced:

- path: `submission_archive/0.bin`
- bytes: `228423`
- sha256: `0cdc96c7e34aa2cdbf41adf3d7e150203b84a0f03d1a3dfd51a3dfb9341ccb1d`

The first macOS default `zip -j` packet added extra fields and deflated a dense
payload, producing a larger archive. The corrected packet used stored mode and
stripped extra fields:

```bash
(cd experiments/results/pr95_local_mps_source_faithful_stage1_25ep_20260519T161718Z/submission_archive \
  && zip -X -0 -j ../archive.zip 0.bin)
```

Corrected archive:

- path: `experiments/results/pr95_local_mps_source_faithful_stage1_25ep_20260519T161718Z/archive.zip`
- bytes: `228531`
- sha256: `fd85d3178549d7cfee4ceaa31078ac2ce13c74c58d16ae7047755fbb0732891e`
- member: `0.bin`
- compression method: stored
- extra field length: `0`

## Auth-Eval Bridge Result

Command:

```bash
RUN=experiments/results/pr95_local_mps_source_faithful_stage1_25ep_20260519T161718Z
RUNTIME=experiments/results/pr95_runtime_upload_20260519T1626Z
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python experiments/contest_auth_eval.py \
  --archive "$RUN/archive.zip" \
  --inflate-sh "$RUNTIME/submissions/hnerv_muon/inflate.sh" \
  --upstream-dir upstream \
  --device cpu \
  --work-dir "$RUN/local_cpu_advisory_eval" \
  --json-out "$RUN/local_cpu_advisory_eval/contest_auth_eval.json" \
  --keep-work-dir
```

Result:

- canonical score: `1.8251394122531366`
- displayed score: `1.83`
- average PoseNet distortion: `0.01569166`
- average SegNet distortion: `0.01276843`
- rate unscaled: `0.006086776496638518`
- archive bytes: `228531`
- inflate elapsed seconds: `36.42334191700047`
- evaluate elapsed seconds: `442.4226857090007`
- inflated aggregate sha256:
  `949f1682194ca85e2ffb15aa16f9d5b648a16d4b55caa47f29b53f343b38d695`
- runtime tree sha256:
  `56fdfacf30cfcef996f5b69cad4e55c62e76cc9b3bbe0cd0fd32dff257565f23`
- runtime content tree sha256:
  `83362e65f21774defb706aaac4e2df2afe6de6f8a4531abe0f08ce64c11c6ae2`

Score delta:

- training-side score: `1.8250676379821704`
- auth-eval canonical score: `1.8251394122531366`
- absolute delta: `0.0000717742709662`

The delta is explained by archive packaging bytes/rate contribution and normal
score recomputation precision, not a runtime-shape or scorer-path failure.

## Blockers And Corrections

- Direct upstream `compress.sh` is path-broken in this checked-out PR95 source:
  `train.py` writes `hnerv_muon/ckpts/run_*`, while `compress.sh` searches under
  `hnerv_muon/src/ckpts/run_*`.
- Direct upstream `train.py` only selects CUDA or CPU; the committed local probe
  harness is the MPS path.
- Modal upload should use a clean source-root runtime shaped as
  `submissions/hnerv_muon/...`, not the `hnerv_muon` directory by itself,
  because `inflate.sh` derives `ROOT="$HERE/../.."` and imports
  `submissions.hnerv_muon.inflate`.
- MPS/macOS CPU remains advisory only. Do not rank, kill, promote, or claim
  frontier movement from this result.

## Next Action

Use local MPS for longer PR95 curriculum timing/convergence probes, then dispatch
only byte-closed archive/runtime packets that clear the local replay ladder.

Immediate high-signal next probes:

1. Run a longer source-faithful PR95 stage-1 window to estimate asymptotic local
   convergence and seconds per epoch.
2. Run source-faithful multi-stage PR95 with `--full-curriculum` or explicit
   stage overrides once stage-1 timing is understood.
3. For any promising packet, package with stored extra-field-free ZIP, build the
   clean source-root runtime, then run paired `[contest-CPU]` and
   `[contest-CUDA]` exact auth eval before any score or rank claim.
