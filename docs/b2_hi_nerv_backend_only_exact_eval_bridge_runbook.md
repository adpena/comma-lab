# B2 exact-eval bridge runbook — HiNeRV backend-only archive → contest score

**Tool:** `tools/run_hi_nerv_backend_only_b2_exact_eval.py`
**Tests:** `src/tac/tests/test_run_hi_nerv_backend_only_b2_exact_eval.py` (18 NO-FAKE tests)
**Lane:** `lane_b2_bridge_20260609`
**Schema emitted:** `hi_nerv_backend_only_exact_eval.v1`

This is the B2 handoff: it turns a HiNeRV **backend-only** `archive.zip` (the
sidecar-stripped double-win packet from
`tools/build_hi_nerv_backend_only_archive.py`) into an **exact contest score** so
a B1 pilot archive can be arbitrated against the public frontier the moment it
lands.

## What it does (pipeline)

Identical to the contest `upstream/evaluate.sh` flow, reused via the canonical,
fully-hardened `experiments/contest_auth_eval.py`:

```
archive.zip (member x or 0.bin)
  -> emit canonical x/0.bin-accepting HiNeRV runtime dir (inflate.sh + inflate.py
     + vendored src/tac/substrates/hi_nerv + _shared; NO scorer imports)
  -> inflate.sh <archive_dir> <inflated_dir> <file_list>   (per-video .raw)
  -> upstream/evaluate.py --device {cpu,cuda}               (d_seg / d_pose / rate)
  -> parse exact final score
  -> emit hi_nerv_backend_only_exact_eval.v1 JSON
```

It does **not** reimplement `evaluate.py` or the inflate decode. It reuses
`contest_auth_eval.py` (archive.zip → inflate.sh → evaluate.py → score with full
ZIP integrity, runtime-tree custody, and device→evidence-grade tagging) plus the
canonical HiNeRV decode-only runtime modules.

## AUTHORITATIVE vs ADVISORY (NON-NEGOTIABLE)

| Axis | Where | Tag | Authoritative? |
|---|---|---|---|
| local macOS CPU | this laptop | `[macOS-CPU advisory]` | **NO** — pipeline-works + runtime only |
| Linux x86_64 CPU | Modal/Vast/Lightning CPU | `[contest-CPU]` | **YES** — public leaderboard axis |
| NVIDIA T4 CUDA | Modal/Vast | `[contest-CUDA]` | **YES** — promotion/ranking axis |
| MPS | any | — | **NEVER** (noise) |

The downstream `contest_auth_eval.py` evidence contract stamps the axis
automatically (macOS CPU ⇒ `score_claim=false`, `promotable=false`). The
authoritative B2 result requires **BOTH** Linux-x86_64-CPU **AND** T4-CUDA on the
**same archive bytes** (paid + deferred).

A 1-pair smoke archive (`num_pairs < 600`) cannot satisfy the contest's
600-sample assertion. For such archives the bridge runs the **inflate stage**
end-to-end (the PR106 dep-closure-bug-class half) and reports
`pipeline_inflate_ok_evaluate_requires_600_pairs` honestly — pipeline-works is a
real claim; there is no score claim.

## Advisory-local validation (free; proves the pipeline runs)

```bash
.venv/bin/python tools/run_hi_nerv_backend_only_b2_exact_eval.py \
    --archive $PACT_TIER1/<candidate>/archive_backend_only.zip \
    --replay-row $PACT_TIER1/<candidate>/hi_nerv_backend_only_exact_replay.json \
    --device cpu \
    --work-root $PACT_TIER1/b2_bridge_work_<utc> \
    --out-row $PACT_TIER1/b2_bridge_work_<utc>/hi_nerv_backend_only_exact_eval.json
```

- `--work-root`: prefer SSD (`$PACT_TIER1/...`, then `$PACT_TIER2/...`) or repo
  `.omx/tmp`. The bridge **refuses** the system `/tmp` tree (disk hygiene).
- `--inflate-python` (default `.venv/bin/python`): the interpreter the inflate.sh
  subprocess uses (`${PYTHON}`). It **must** satisfy the contest dep-closure
  (`brotli + torch + numpy`). Bare homebrew `python3` lacks brotli/torch — the
  PR106 `ModuleNotFoundError` bug class.
- Inflated `.raw` frames are **certified rebuildable** (deterministic via
  inflate.sh on the archive sha) and **auto-deleted** after eval. `--keep-work-dir`
  preserves them.

**Validated 2026-06-09** on the real `hinerv_backend_only_candidate_20260608`
archive (member `x`, `HIV1`, 9 064 bytes, `num_pairs=1`):
`inflate_ok=true`, `inflate_returncode=0`, raw output `6 104 016` bytes
(= `2 × 874 × 1164 × 3`, frame-aligned, 2 frames), `inflate_elapsed≈0.6-0.9s`
(well within the 30-min budget), `verdict=pipeline_inflate_ok_evaluate_requires_600_pairs`,
`axis_tag=[macOS-CPU advisory]`, `score_claim=false`.

## AUTHORITATIVE dual-axis dispatch (paid; deferred until a B1 600-pair archive)

Print the exact recipe any time:

```bash
.venv/bin/python tools/run_hi_nerv_backend_only_b2_exact_eval.py --print-dual-axis-recipe
```

Run **both** on the **same** archive bytes (apples-to-apples), after operator
approval + lane claim:

```bash
# Axis 1 — contest-CPU (public leaderboard), Linux x86_64 (Modal CPU ~$0.06):
.venv/bin/python -u experiments/contest_auth_eval.py \
    --archive <B1_600pair_archive.zip> \
    --inflate-sh <runtime_dir>/inflate.sh \
    --video-names-file upstream/public_test_video_names.txt --upstream-dir upstream \
    --device cpu \
    --json-out <linux_cpu_workdir>/hi_nerv_backend_only_exact_eval_contest_cpu.json \
    --work-dir <linux_cpu_workdir>

# Axis 2 — contest-CUDA (promotion), NVIDIA T4/equiv (Modal T4 ~$0.30-0.60):
.venv/bin/python -u experiments/contest_auth_eval.py \
    --archive <B1_600pair_archive.zip> \
    --inflate-sh <runtime_dir>/inflate.sh \
    --video-names-file upstream/public_test_video_names.txt --upstream-dir upstream \
    --device cuda \
    --json-out <t4_cuda_workdir>/hi_nerv_backend_only_exact_eval_contest_cuda.json \
    --work-dir <t4_cuda_workdir> \
    --expected-runtime-tree-sha256 <sha-from-cpu-run>

# Step 3 — compliance gate before any judge-facing/public submission:
.venv/bin/python scripts/pre_submission_compliance_check.py \
    --contest-final --strict --archive <B1_600pair_archive.zip> \
    --expected-archive-sha256 <sha256> --expected-archive-size-bytes <bytes> \
    --auth-eval-json <contest_cuda_or_cpu_json>
```

The `<runtime_dir>/inflate.sh` is emitted by an advisory `--work-root` run of the
B2 bridge (it vendors the hermetic x/0.bin-accepting HiNeRV runtime). Lane claim
before any paid dispatch:

```bash
tools/claim_lane_dispatch.py claim --lane-id lane_b2_bridge_20260609 \
    --platform modal --agent <agent> --instance-job-id <job> --status dispatched_dual_axis
```

When the B1 archive is a real 600-pair packet, the bridge auto-detects
`num_pairs == 600` and runs the **full** `contest_auth_eval.py` pipeline itself —
it becomes one command:

```bash
.venv/bin/python tools/run_hi_nerv_backend_only_b2_exact_eval.py \
    --archive <B1_600pair_archive.zip> --device cpu \
    --work-root $PACT_TIER1/b2_<utc> \
    --out-row $PACT_TIER1/b2_<utc>/exact_eval.json
```
(and again with `--device cuda` for the CUDA axis).

## Frontier arbitration

Frontier to beat (re-derive from `tools/refresh_canonical_frontier.py`; do NOT
hardcode): the leaderboard ranks by **contest-CPU**. Once a B1 archive has a
`[contest-CPU]` score from this bridge, compare it to the canonical frontier
pointer; the `[contest-CUDA]` score is the internal promotion axis.

## Dep-closure note (PR106 bug class)

The HiNeRV `inflate.sh` honours `${PYTHON:-python3}`. On a clean contest machine
the self-contained runtime venv/uv must provide `brotli + torch + numpy`; bare
`python3` does not. The B2 bridge's advisory path points `${PYTHON}` at a venv
that satisfies the closure. For a paid Linux/T4 dispatch, the runtime tree shipped
into the work dir must include these deps (the canonical `contest_auth_eval.py`
runs under the repo venv / uv).
