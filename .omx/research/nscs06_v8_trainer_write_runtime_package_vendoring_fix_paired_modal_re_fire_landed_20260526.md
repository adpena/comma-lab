# NSCS06 v8 `_write_runtime` package-vendoring fix + paired Modal re-fire landed 2026-05-26

**Lane:** `lane_nscs06_v8_trainer_write_runtime_package_vendoring_fix_20260526` L1
**Subagent:** `nscs06-v8-trainer-write-runtime-package-vendoring-fix-plus-modal-re-fire-pr111-unblock-20260526`
**Predecessor (PARTIAL_SUCCESS):** commit `d4ac491fa` (subagent `aa0601e603067ed99`)
**Sister (trainer-v3-wire-in):** commit `5685f1a0c`
**Operator approval:** 2026-05-26 verbatim *"all are approved"*

---

## Bug class fingerprint

`experiments/train_substrate_nscs06_v8_chroma_lut.py` `_write_runtime` invoked
`shutil.copy2(PROCEDURAL_CODEBOOK_GENERATOR_SOURCE, ...)` against
`src/tac/procedural_codebook_generator.py` (single file path). The canonical
surface was refactored 2026-05-26 to a **PACKAGE** at
`src/tac/procedural_codebook_generator/` with `__init__.py` re-exporting
`derive_codebook_from_seed` from 8 submodules. The trainer's single-file
copy targeted a non-existent path; FileNotFoundError on Modal worker;
auth_eval blocked despite v3 archive sha `f187df3653...` emitting structurally.

**Empirical anchor** (canonical Modal call_id ledger
`.omx/state/modal_call_id_ledger.jsonl`):

- Pre-fix call_id `fc-01KSJTVTAX5JBSF0F45P6H3XKJ` (HEAD `a0040599d`):
  `rc=1`, trainer crashed at line 402 with
  `FileNotFoundError: NSCS06 v8 vendoring failed: procedural codebook
  generator missing: /tmp/pact/src/tac/procedural_codebook_generator.py`.

Per Catalog #307: IMPLEMENTATION-LEVEL falsification of the vendoring
mechanism; PARADIGM intact (v3 trainer-wire-in produces correct archive
sha; chroma LUT substrate paradigm unchanged).

## The fix (~19 LOC / 1 file)

Commit `e278a497038d` via canonical serializer
(`tools/subagent_commit_serializer.py` per Catalog #117/#157/#174;
POST-EDIT working-tree sha
`240cbbc3db23abd39b4f3998b8677bb19733ce7b03cc64633e5f9a902c751a21`).

```diff
- # The procedural_codebook_generator is the cross-substrate canonical helper
- # (sister grayscale_lut + DP1 + VQ-VAE pattern). Vendor it under the codec
- # subdir as well so the inflate runtime is fully self-contained.
- PROCEDURAL_CODEBOOK_GENERATOR_SOURCE = (
-     REPO_ROOT / "src" / "tac" / "procedural_codebook_generator.py"
- )
+ # The procedural_codebook_generator is the cross-substrate canonical helper
+ # (sister grayscale_lut + DP1 + VQ-VAE pattern). The canonical surface is a
+ # PACKAGE (directory with __init__.py + 8 submodules) per the 2026-05-26
+ # canonical refactor; vendor it as a tree under the codec subdir so the
+ # inflate runtime's relative import `from .procedural_codebook_generator
+ # import derive_codebook_from_seed` resolves against the package's __init__.py
+ # re-export. Per Catalog #295 the submission tree is self-contained (no
+ # tac.* imports from the judge's runtime).
+ PROCEDURAL_CODEBOOK_GENERATOR_SOURCE = (
+     REPO_ROOT / "src" / "tac" / "procedural_codebook_generator"
+ )

- # Vendor procedural_codebook_generator.py alongside.
- if not PROCEDURAL_CODEBOOK_GENERATOR_SOURCE.is_file():
-     raise FileNotFoundError(...)
- shutil.copy2(PROCEDURAL_CODEBOOK_GENERATOR_SOURCE, ...)
+ # Vendor procedural_codebook_generator/ PACKAGE alongside.
+ if not PROCEDURAL_CODEBOOK_GENERATOR_SOURCE.is_dir():
+     raise FileNotFoundError(...)
+ shutil.copytree(
+     PROCEDURAL_CODEBOOK_GENERATOR_SOURCE,
+     vendored_dir / "procedural_codebook_generator",
+     ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
+ )
```

Existing patched relative import in vendored inflate.py
(`from .procedural_codebook_generator import derive_codebook_from_seed`)
now resolves against the package's `__init__.py` re-export.

## Local CPU pre-dispatch parity verdict: PASS

Full `_write_runtime` emission test (tmp dir):

- 8 submodules + `__init__.py` vendored (~109 KB total vs ~17 KB previously
  with the missing package path)
- `__pycache__` excluded (helper `shutil.ignore_patterns`)
- `PACKAGE_IMPORT_OK`: vendored package importable from submission tree
- `INFLATE_IMPORT_OK`: vendored inflate.py's patched relative import resolves
- Top-level submission `inflate.py` exits `rc=2` with canonical Catalog #146
  usage message (proves import phase completes cleanly)

Test suites:
- 206 / 206 NSCS06 v8 substrate tests PASS
- 69 / 69 canonical equation #344 + seed_derived_codebook tests PASS

## Paired Modal re-fire

| Axis | Call ID | HEAD | Status |
|------|---------|------|--------|
| CUDA T4 | `fc-01KSJWKFJMGF67MAC45MV00R4D` | `e278a497038d` | DISPATCHED (in flight) |
| CPU paired | held by CUDA lane claim per Catalog #246 paired-dispatch helper | — | enforced post-CUDA via canonical paired-dispatch chain |

**Dispatch envelope:** cost band p10/p50/p90 = $0.00/$0.07/$0.20 (N=8
empirical_posterior). Well within $0.50-1.00 operator-approved envelope.

**Discipline:** Catalog #117/#157/#174 serializer; Catalog #166 sentinel-files
threaded (6 sentinels); Catalog #202 paired-env clean-tree bypass
(`OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 +
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1`) with verbatim
operator rationale; Catalog #199 paired-env operator-authorize bypass
(`OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 +
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=1.00`); Catalog #245 ledger
registered (event_type=dispatched, status=dispatched); Catalog #339
fail-closed registration; Catalog #270 dispatch_optimization_protocol PASS.

## Final scores per axis vs T3 predicted band

**Status:** in flight; harvest pending via
`.venv/bin/python experiments/modal_recover_lane.py --call-id
fc-01KSJWKFJMGF67MAC45MV00R4D`.

T3 PR110-stacking-pivot-ordering predicted band per sister symposium
(commit `d4ac491fa`): **ΔS ∈ [-0.0027, -0.0015]** from canonical
equation #26 closed form
`ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706`.

PR111 candidacy verdict per Catalog #307 framework deferred until paired
auth_eval completes:
- **If empirical ΔS within predicted band**: PARADIGM-VALIDATED → PR111
  candidate (NSCS06 v8 first-decision in T3 PR110-stacking ordering)
- **If empirical ΔS outside predicted band**: IMPLEMENTATION-LEVEL
  reclassification → root-cause analysis + canonical equation #26
  context refinement (NOT paradigm refutation)

## Canonical equation #344 anchor

Will be appended on harvest via
`tac.canonical_equations.update_equation_with_empirical_anchor` for
equation #26 (`procedural_codebook_from_seed_compression_savings_v1`,
IN-DOMAIN context `nscs06_v8_chroma_lut`). Sister context registration
for v3 variant (`v3_procedural_seed_with_cls_stream`) operator-routable
per Catalog #344 protocol if predicted-vs-empirical residual > 2σ.

## Canonical-vs-frontier-push decision per Catalog #344

Anchor is canonical equation refinement (existing IN-DOMAIN context;
mechanism already registered). Per CLAUDE.md "Long-burn score-lowering
campaign default": frontier-push decision deferred to T3 council
post-harvest per PR110-stacking-pivot-ordering verdict.

## 6-hook wire-in per Catalog #125

1. **Sensitivity-map**: N/A (trainer-engineering fix; no signal axis change)
2. **Pareto constraint**: N/A (no new constraint)
3. **Bit-allocator hook**: N/A (archive grammar unchanged)
4. **Cathedral autopilot dispatch hook**: ACTIVE — empirical anchor will
   land in `.omx/state/modal_call_id_ledger.jsonl` (Catalog #245); cathedral
   consumers auto-discovered per Catalog #335 will absorb the result
5. **Continual-learning posterior update**: ACTIVE — paired auth_eval
   verdict feeds canonical equation #26 anchor via
   `tac.canonical_equations.update_equation_with_empirical_anchor`
6. **Probe-disambiguator**: N/A (single mechanism per equation #26
   IN-DOMAIN context; no multi-interpretation surface)

## Drift surface declaration (MLX ↔ CUDA bidirectional)

Trainer is numpy + Pillow only (no torch / no CUDA / no MPS); no MLX↔CUDA
drift surface exists. v3 variant uses MLX iteration substrate at compress
time (`src/tac/substrates/nscs06_v8_chroma_lut/tests/test_mlx_iteration.py`),
but inflate runtime is deterministic numpy reconstruction from procedural
seed; per CLAUDE.md "MLX portable-local-substrate authority" the MLX path
remains research-signal-only and does not contaminate the contest auth-eval
axis.

## Operator-routable next step

1. **Harvest CUDA arm**: `.venv/bin/python experiments/modal_recover_lane.py
   --call-id fc-01KSJWKFJMGF67MAC45MV00R4D` when complete
2. **Dispatch paired CPU arm** after CUDA lane releases per Catalog #246
   paired-dispatch helper (the recipe enforces paired-CUDA+CPU as a
   DISPATCH-CHAIN concern, not a dispatch-blocker)
3. **Compare paired CPU vs CUDA scores** per CLAUDE.md "Submission auth
   eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
4. **PR111 candidacy decision** per Catalog #307 + T3 PR110-stacking
   ordering verdict

## Cross-references

- T3 PR110-stacking-pivot-ordering symposium: `feedback_t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
- PARTIAL_SUCCESS predecessor: commit `d4ac491fa`
- Sister trainer-v3-wire-in: commit `5685f1a0c`
- Canonical equation #26: `src/tac/canonical_equations/procedural_codebook_savings.py`
- Canonical Modal call_id ledger: `.omx/state/modal_call_id_ledger.jsonl`
- Catalog #361 sister bug class (Modal artifact filter package
  preservation): same META class at the harvester-filter surface
- Catalog #295 (submission inflate runtime self-containment): the parity
  invariant that drove the fix design
