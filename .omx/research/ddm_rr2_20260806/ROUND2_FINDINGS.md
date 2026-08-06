# DDM RR2 Round 2 PR130 Lift Wave Adversarial Review

Date: 2026-08-06
Reviewer: ddm_rr2
Verdict: NOT-CLEAN
Clean counter: 0/3
Scorer use: none; this review used no n600 scorer slot.

## Scope And Method

This was a fresh-eyes adversarial review of the PR130 lift wave, not a restatement
of the Round 1 findings. I read the governing Pact rules, the operating craft
handoff, the Round 1 review memo, the ET4 cache repair path, the MX1/MX2/HB1/EH1
receipts and launch artifacts, the PR130 lift source tree, the experiment
drivers, the vendored-source manifests, and the imported PR130 upstream source.

No new exact score, d_seg, or d_pose result is claimed here. All findings below
are custody/control-flow findings from file contents, launch receipts, tests,
and source-line tracing.

## Findings

| ID | Severity | Axis | Type | Status |
| --- | --- | --- | --- | --- |
| RR2-F1 | CRITICAL | call-site tracing / measured-quantity isolation | machine-readable fire order incomplete | QUEUED to MAIN |
| RR2-F2 | CRITICAL | resume/crash path | live long job is weight-warm-start, not deterministic resume | QUEUED to MAIN |
| RR2-F3 | MEDIUM | resume/crash path / default override hunting | MX1 resume accepts mismatched command metadata silently | QUEUED to MAIN |
| RR2-F4 | MEDIUM | parity adjudication leak path | MLX train telemetry can be mistaken for authority unless verifier is forced | QUEUED to MAIN |

### RR2-F1: MX1 Round 1 F1 Was Fixed In Prose, Not In The Executable Ticket

Evidence:

- `.omx/research/ddm_mx1_20260806/LAUNCH_TICKET.md:74` records the Round 1
  amendment that a valid MX1 decision needs two n32 arms: ARM-CAP GT-to-GT and
  ARM-VEH tq1c-to-GT.
- `.omx/research/ddm_mx1_20260806/LAUNCH_TICKET.md:25` still publishes a single
  fire-order command using the tq1c input cache and GT target cache.
- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_ticket.json:39`
  exposes only one `argv_n32` command, again tq1c input to GT target.
- `.omx/research/ddm_mx1_20260806/NEXT_IF_RESUMED.md:25` repeats the same
  single-arm n32 command.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:510` through
  `experiments/ddm_mx1_pr130_semantic_renderer.py:628` emits one `argv_n32` and
  one `argv_n120` from the current `--input-cache`/`--target-cache` values.

Finding:

Round 1 correctly identified the token-source conflation and amended the prose,
but the artifacts a launcher would actually consume still encode only the
vehicle arm. A MAIN agent following the JSON or the top fire-order block can run
ARM-VEH only, never running the GT-to-GT receiver-capacity discriminator. That
allows a Row-1/MX1 conclusion to keep conflating receiver capacity with
tq1c-to-GT correction reach.

Required fire order:

Regenerate the machine-readable ticket and the resumed instructions with two
explicit commands:

- `argv_n32_arm_cap`: input cache = GT parent labels, target cache = GT parent
  labels, separate run directory, same seed and pair selection policy.
- `argv_n32_arm_veh`: input cache = tq1c/public-wire parent labels, target cache
  = GT parent labels, separate run directory, same seed and pair selection
  policy.

Do not dispatch n120 until the arm being scaled is explicitly selected from the
two n32 results. Add a ticket-shape test or static guard so this cannot regress
to a one-arm ticket.

### RR2-F2: HB1's Live Driver Is Not Crash-Resumable

Evidence:

- `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/hpac_our_labels_driver.sh:5`
  claims the trainer writes `<save>.latest.pt` and that the driver resumes by
  passing it as `--init`.
- `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/hpac_our_labels_driver.sh:20`
  through `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/hpac_our_labels_driver.sh:39`
  selects `.latest.pt` as `--init` and launches a 60-epoch HPAC self-compress
  stage.
- `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_hpac_self_compress.py:99`
  loads only `initial["state_dict"]` from `--init`.
- `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_hpac_self_compress.py:132`
  through `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_hpac_self_compress.py:135`
  constructs fresh optimizer and scheduler state on every invocation.
- `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_hpac_self_compress.py:169`
  through `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_hpac_self_compress.py:171`
  starts a fresh epoch loop from epoch 1.
- `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_hpac_self_compress.py:213`
  through `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_hpac_self_compress.py:217`
  writes latest checkpoints containing model weights, config, and history, but
  no optimizer state, scheduler state, RNG state, current epoch, or best-state
  lineage sufficient for deterministic continuation.
- `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/launch/launch_manifest.json:1`
  records the detached HPAC-on-our-labels launch.

Finding:

The HB1 driver is running a long stage under a false resume claim. Restarting
from `.latest.pt` is a weight warm-start, not a deterministic crash-resume from
disk. A crash after partial progress followed by the scripted restart would
reset optimizer, scheduler, RNG/generator, and epoch accounting while training
for another 60 epochs. Any output from that lineage is not clean enough to serve
as a canonical row or a clean premise for a downstream pack/decode stage.

Required fire order:

Treat the current HB1 output as quarantined/advisory unless MAIN explicitly
chooses a restart policy. The clean path is to add a real `--resume-from` path
that persists epoch, optimizer, scheduler, RNG/generator state, best state,
history, config, input/target source hashes, and atomic stage checkpoints under
distinct names. Then relaunch from a clean checkpoint surface rather than from
the current warm-start-only path.

### RR2-F3: MX1 Resume Does Not Fail Closed On Metadata Mismatch

Evidence:

- `experiments/ddm_mx1_pr130_semantic_renderer.py:371` loads a resume checkpoint
  and trusts its model/optimizer state.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:382` through
  `experiments/ddm_mx1_pr130_semantic_renderer.py:388` then rebuilds caches and
  selected pair IDs from the current command-line arguments.
- `experiments/ddm_mx1_pr130_semantic_renderer.py:457` through
  `experiments/ddm_mx1_pr130_semantic_renderer.py:487` stores pair IDs and some
  score-claim metadata, but not enough input-cache, target-cache, init, seed,
  source, and pair-selection hashes to reject a mismatched resume.
- `src/tac/pr130_lift/mlx_semantic_renderer.py:403` through
  `src/tac/pr130_lift/mlx_semantic_renderer.py:446` loads checkpoint metadata
  but leaves compatibility enforcement to callers.

Finding:

An MX1 resume can silently continue optimizer/model state under a different
`--input-cache`, `--target-cache`, `--seed`, `--pairs`, `--bits`, `--init`, or
pair-selection mode. That is a resume-integrity bug even if the current launch
ticket is typed correctly, because the next operator command can accidentally
splice incompatible state into a claimed n32/n120 result.

Required fire order:

Store and validate input/target cache paths and SHA-256s, init checkpoint SHA,
source repo/head, seed, pair-selection mode, pair IDs, model config, optimizer
config, and total-step horizon. Resume must fail closed when any identity field
differs or when the checkpoint step already exceeds the requested horizon.

### RR2-F4: MX1 Keeps MLX Telemetry In The Result Path Without A Forced CPU-Torch Verdict Step

Evidence:

- `experiments/ddm_mx1_pr130_semantic_renderer.py:446` through
  `experiments/ddm_mx1_pr130_semantic_renderer.py:455` records `d_seg_batch`
  from the MLX training-time SegNet adapter.
- `.omx/research/ddm_mx1_20260806/PARITY.md:101` through
  `.omx/research/ddm_mx1_20260806/PARITY.md:108` correctly states that MLX
  parity telemetry is not authority and that a CPU-torch verifier must decide
  PR130-lift verdicts.
- The launch ticket does not force a separate CPU-torch post-train verifier
  command before any scaled decision.

Finding:

The policy text is right, but the artifact flow still leaves a low-friction leak
path: a downstream reader can copy MLX `d_seg_batch` from the result JSON as if
it were the verdict. This is not an observed false claim in the receipts I read;
it is an unguarded path from advisory telemetry to decision language.

Required fire order:

Add an explicit CPU-torch post-train verifier command to the ticket and result
contract, and label the MLX telemetry fields with an axis such as
`[macOS-MLX research-signal]`. The scale-up gate should require the CPU-torch
verifier receipt path, not just the MLX training result path.

## Mandatory Assumption Challenge

The hidden assumption Round 1 did not close is:

> A prose amendment is operationally equivalent to a machine-readable launch
> artifact.

RR2-F1 refutes that assumption. The Markdown amendment says the right thing, but
the JSON ticket, the resumed instructions, and the generator code still encode a
one-arm launch. The executable shape, not the prose, is what determines whether
the receiver-capacity discriminator is actually measured.

A second assumption exposed by HB1 is:

> A latest checkpoint accepted by `--init` is equivalent to crash-resume.

RR2-F2 refutes that assumption. A model-only latest file is a warm-start unless
it preserves enough training state to continue deterministically.

## Borrowed-Substrate Accounting

PR130 artifacts remain borrowed substrate unless they are receiver-closed,
byte-closed, and scored under our exact authority. The imported PR130 source
head inspected here is `2f94596fc6991c55dd777fb2e43418430ef88243e`. Vendored
source manifests matched their declared upstream-file hashes where checked; the
local wrappers add provenance headers and Pact-specific guards, which is allowed
only if our receipts keep external PR130 scores separate from our own vehicle
rows. This review found no new exact Pact score and makes no claim that PR130's
public/external score is our frontier.

## Default Override Hunt

The PR130 semantic Stage-08 source defaults were traced in the imported
`scripts/e2e.py`: bits 4, 6000 steps, batch size 2, eval batch size 8, eval every
250, lr 2e-7, CE 0, softplus -999, seed 20260716. The MX1 driver defaults match
that family and correctly fail closed when MLX is unavailable for parity/train.
The override risk found by this review is not a wrong literal default; it is
that the launch ticket contains only one argument vector, so the intended ARM-CAP
default set is absent rather than overridden.

For ET4, the batch-shape repair is scoped and honest: the repair rebuilds the
parent argmax cache through batch-1 inference, preserves the original batch-16
cache, records the old/new SHA-256s, and labels first-8 prefix evidence as not
banked. I did not find a new ET4 cache-lineage blocker in the bounded review.

## Resume And Crash Path Verdict

MX1: has checkpoint files and optimizer persistence through the MLX helper, but
needs metadata identity validation before it is clean for scaled continuation.

MX2: the launch ticket already blocks dispatch on the lack of a true
`--resume-from`; RR2 does not add a new finding there.

HB1: NOT-CLEAN. The live HPAC path is warm-start-only and must not be treated as
crash-resumable.

ET4: the cache repair is non-training and uses atomic replacement with old-cache
preservation; no new resume blocker found.

## Parity And Authority

No MPS or MLX result is promoted here. MLX parity remains training-adjacent
telemetry only. The only authority for a promoted row remains the exact
byte-closed archive evaluated through the contest CPU/CUDA path and the official
receiver/scorer stack. RR2 did not run that path.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/pr130_lift/tests/test_mx1_pr130_lift.py \
  src/tac/pr130_lift/tests/test_mx2_pose_lift.py -q
```

Result: 9 passed in 0.61s. A non-authority MLX atexit message reported no Metal
device.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import importlib
mods = [
    "experiments.ddm_mx1_pr130_semantic_renderer",
    "experiments.ddm_et4_rebuild_parent_argmax_cache",
    "src.tac.pr130_lift.mlx_semantic_renderer",
    "src.tac.pr130_lift.pose.repack_race",
    "src.tac.pr130_lift.pose.source_loader",
    "src.tac.pr130_lift.pose.mlx_pose_carrier",
]
for mod in mods:
    importlib.import_module(mod)
    print("import ok", mod)
PY
```

Result: all six imports succeeded.

## Recall Evidence

Beyond the supplied charter and its seeded paths, I ran bounded recall queries
against the live Pact research and state surfaces before deciding:

```bash
rg -n "PR130|HPAC|CPR1|semantic renderer|pose carrier|eureka|batch_shape_is_part|F1|Row-1|ARM-CAP|ARM-VEH" \
  .omx/research/CANONICAL_RESEARCH_INDEX* \
  .omx/research/sub015_DAG_* \
  .omx/state/main_hot_state.md \
  .omx/state/canonical_task_status.jsonl \
  .omx/state/operator_p0_ledger.jsonl \
  .omx/research/ddm_*_20260806*.md \
  .omx/research/ddm_*_20260806/*.md
```

The recall hit live PR130 state in `main_hot_state.md`, the PR130 DAG entries
for external-score separation and "proceed with revisions," the MX1/MX2/HB1/ET4
receipts, and the batch-shape continuity amendments. It changed the review by
forcing RR2 to treat PR130 public numbers as external until exact Pact custody
exists, and by focusing on executable launch artifacts rather than prose.

```bash
.venv/bin/python tools/list_canonical_equations.py --json | rg -n "PR130|MX1|MLX|HPAC|batch|authority|MPS|CUDA|CPU"
```

This found no PR130-specific equation that superseded the live receipts, but it
reinforced the existing authority split: MLX/MPS telemetry is advisory and
cannot be promoted into an exact row.

## Follow-On Disposition

FOLDED: ET4 batch-seam repair remains scoped to cache instrument identity; no
additional ET4 blocker found in this review.

QUEUED: RR2-F1 requires MAIN to regenerate the MX1 machine-readable two-arm
ticket and resumed instructions before any n32/n120 dispatch.

QUEUED: RR2-F2 requires MAIN/operator handling of the live HB1 warm-start-only
job. Current outputs should be quarantined/advisory unless relaunched under a
true resume contract.

QUEUED: RR2-F3 and RR2-F4 require MX1 result-contract hardening before a scaled
decision can be clean.

## Own-Vehicle Frontier Honesty

No lower exact score landed in this review. The own-vehicle frontier remains:

`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`
