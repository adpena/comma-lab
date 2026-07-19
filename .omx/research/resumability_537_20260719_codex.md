# Task #537 resumability seal — Codex handoff (2026-07-19)

## Verdict and authority

`IMPLEMENTATION_READY / REQUIRED_REAL_MLX_PROOF_BLOCKED_ENVIRONMENT_NO_METAL_DEVICE`.

This branch has no launch, score, submission, or pointer-promotion authority. The canonical
`0.1910828242 [contest-CPU]` pointer is unchanged. The sacred run directory was read only. MAIN
must review the branch diff and rerun the real n24 probe on its Metal-capable host before treating
the mandatory crash/resume proof as complete.

Canonical lane `resumability_537_20260719` is honestly L1: `impl_complete` and
`strict_preflight` are marked; `real_archive_empirical` remains false until the n24 proof is green.

The implementation and handoff follow `docs/operating_manual_craft_handoff.md`: authority and
scope are explicit, evidence is durable, a missing proof is named literally, and the next owner has
an exact executable continuation rather than an inferred success.

## Landed implementation

1. Periodic checkpointing now writes rolling compatibility aliases plus distinct atomic
   `levelset_periodic_{ema,resume}_<stage>_ep<N>.npz` pairs. `--ckpt-retain-per-stage M` prunes only
   the new periodic grammar for the current stage; rolling, BEST, Polyak, stage-boundary, final, and
   unknown files are outside the deletion grammar.
2. Every new resume sidecar records a semantic schema, stage, RNG, and an explicit event-ledger row.
   Active event controllers without persisted keys refuse checkpoint creation. Normal continuation
   requires live weights, EMA, optimizer moments, full NumPy RNG state, event custody, and stage/epoch
   position. Legacy full sidecars pass only through directly present event keys and emit a loud
   `legacy_semantic_keys` compatibility row.
3. Optimizer restoration is exact-keyset-only. A subset, superset, or restore exception refuses;
   it cannot silently continue with fresh moments. `--warm-start-weights-only` is the sole
   intentional state-drop path.
4. Intentional weights-only re-treatment re-anchors active resume-round events to
   `resume_epoch + adam_v_variance_warmup_epochs(beta2, ceil(num_pairs/accum_pairs))`. Normal
   bit-faithful continuation is an identity operation and retains its saved event ledger/anchors.
5. `src/tac/preflight.py` has a warn-only live-trainer save-path scan, wired into
   `preflight_all(strict=False)`. The resume registry's static canonical set now also acknowledges
   the already-live `rate_rolling_telemetry` controller, closing its pre-existing static mismatch.
6. Two reusable probes landed: a governed, directly typed-DSL real n24 crash/resume probe and a
   read-only sacred-layout compatibility/refusal probe. Both use success-only cleanup; the real
   probe preserves failure metadata and writes a machine-readable blocker receipt.

## Evidence

### Green local evidence

- `python -m py_compile` for trainer, preflight, and both probes: PASS.
- Focused Task #537 helper tests: `14 passed`.
- New save-path preflight plus existing resume-registry gate: `30 passed`.
- Whole `test_levelset_checkpoint_resume.py`: the Task #537 additions pass; four older `main()`
  tests still stop at the repository DSL admission guard before reaching their assertions. This
  diff does not touch those tests' admission setup or guard call.
- Sacred-layout replay receipt:
  `.omx/research/resumability_537_sacred_layout_replay_receipt_20260719.json`, SHA-256
  `b0fe9783ce50e177b4f1aace31d6016af955dbcbe547bce8d85cfb990d0e4b11`.
  It proves the source/copy hash is
  `f94347356f7da39972a3f2596846e2713a9f96d9b2274fa803664f43e6e37bf8`, the real legacy layout
  passes via direct semantic keys, the optimizer-stripped copy refuses, scratch was deleted, and
  the source hash stayed unchanged.

### Mandatory real proof: exact blocker, not a pass

Receipt: `.omx/research/resumability_537_n24_crash_resume_receipt_20260719.json`, SHA-256
`24a9aaf0b162ddb22ffbc0c1c57c695a9882b3c7de84e8aab64dce5d3eb60d47`.

The probe compiled and independently reverified three exact typed-DSL launch units for the real
221,996,742-byte `gt_n24.npz`. Governed admission and DSL compile admission both passed. Before the
first epoch, importing `mlx.nn` reached `mx.compile` and refused with:

`RuntimeError: [metal::load_device] No Metal device available.`

The attempted CPU selection cannot make this installed MLX build import `mlx.nn` without Metal in
the current execution environment. No epoch, checkpoint, score, or source mutation occurred. The
receipt records all nine launch/provenance/manifest hashes and preserves only those small generated
failure artifacts. This is `BLOCKED_ENVIRONMENT_NO_METAL_DEVICE`, not evidence against the
implementation and not completion of the mandatory proof.

MAIN continuation on a Metal-capable host:

```text
/Users/adpena/Projects/pact/.venv/bin/python \
  tools/probe_resumability_537_real_smoke.py \
  --python /Users/adpena/Projects/pact/.venv/bin/python
```

MAIN must require the regenerated receipt to say `all_pass=true` and review its continuous/resumed
live+EMA hashes, optimizer/RNG/event/stage hashes, SIGKILL epoch, byte-close periodic load, retention
bound, final preserved pair, and cleanup row before landing/promoting this task.

## Self-review log (2 of 5 allowed rounds)

1. Structural diff review plus focused helper/preflight/registry tests. Corrected canonical static
   custody for `rate_rolling_telemetry`.
2. Adversarial restore and real-proof review. Extinguished partial optimizer restoration, routed the
   proof through a direct `TypedWitnessConfig` plus recomputed DSL artifacts, and recorded the exact
   no-Metal blocker without synthetic substitution.

## MAIN landing review checklist

- Review `base..codexwt/resumability_537_20260719_20260719T070411Z`; trainer is a hot file.
- Confirm the legacy semantic-key allowlist against the sacred receipt and ensure no weaker deploy
  checkpoint can enter normal continuation.
- Confirm event re-anchoring is limited to explicit weights-only re-treatment and that exact
  continuation remains ledger-authoritative.
- Run the exact n24 command above on Metal and require an all-pass receipt before merging.
- Preserve pointer and contest-axis separation; this task has no score claim.
