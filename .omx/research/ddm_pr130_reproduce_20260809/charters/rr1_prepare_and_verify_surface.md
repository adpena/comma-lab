# ddm_rr1 — recursive adversarial review, LAYER 1: the PREPARE + VERIFY surface

**Operator directive 2026-08-09:** *"The Recursive adversarial Review must check every step of every
stage of everything, recursive fractal of our port to upstream PR one thirty."*

You are a FRESH-EYES adversarial reviewer. MAIN did round 1 alone and found real defects in his own
landings — that is the weakest configuration the protocol allows. Your job is to find what his frame
could not.

## THE OBJECT UNDER REVIEW

We reproduced PR130's chain on this Metal host. `BASE = PR130 CPR1 S = 0.172141297491896447`
`[contest-CUDA, DALI GT, n600]`, archive 191,052 B sha
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.

Intake (READ-ONLY, never write, never `git add` inside):
`/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`
`SOURCE_REPO_HEAD = e34f31bc4969042c0051ac81aa3c56884419a231`

The landed ledger you are auditing: `.omx/research/ddm_pr130_reproduce_20260809/OFF_THE_SHELF_VS_PORTED.md`
(commit 3856788c96) and `THROUGHPUT_ROOT_CAUSE.md` (d28fde10f5).

## YOUR SCOPE — layer 1, element-deep, loop-until-dry

1. **`scripts/audit_repo.py`** — read every check. What does it actually assert? What does it NOT?
   Report the DENOMINATOR (checks defined vs checks that can fire vs checks that fired).
2. **`scripts/verify.sh`'s 4 pytest files** — `code/test_carrier_codec.py`, `tests/test_e2e_plan.py`,
   `tests/test_official_report.py`, `tests/test_provenance.py`. MAIN reported "24 passed" as if it
   were the verification. **Adversarial hypothesis to test: the 24 are structural/plan assertions and
   the only load-bearing check in the whole verify path is the single byte-identity comparison inside
   `reproduce.sh`.** Confirm or refute with the actual assertions. If any test would still pass with
   the payload replaced by zeros, say so.
3. **`prepare` / `build_gt_cache_official.py`** — the GT cache lineage. Two caches exist
   (`gt_cache_600.pt`, `gt_cache_600_official_ada.pt`). What produces each, what decodes the video
   (DALI? PyAV? AV?), and does the semantic leg consume a DIFFERENT cache than the carrier leg? This
   connects to task #906 (chroma siting / DALI-vs-AV) — if the two caches come from different
   decoders, every cross-leg metric comparison has an unstated confound.
4. **`compileall`** — is it a real check or a no-op in this context?

## ROUND-1 FINDINGS — do NOT re-derive these, build past them

- F1: `reproduce.sh` reads BANKED artifacts (`artifacts/base/`, `artifacts/hpac/`). The byte-identity
  closes the ASSEMBLY tail only, not training.
- F3: `encode-tokens` consumed the BANKED hpac checkpoint (`canonical_hpac_checkpoint()` default).
- F4: the shipped semantic checkpoint's embedded `config` describes an ANCESTOR run (steps 3000,
  lr 1e-3, `amp: True`, path `…w96_b2…`) while the file is `…b4_qat4_tail6k_lr2e7` and the trainer
  has NO autocast. Architecture fields survive (strict load succeeded); schedule/precision are stale.
- REFUTED already: device-is-inert (A1), width-provenance (A3-arch), double-eval (A4).

## OPTIMAL FORM

- **Reference form:** a full-coverage static audit of every assertion in the verify path, with
  denominators reported per the #50 vacuity law (`skip == green`, `--no-codebase` == 502 gates unrun,
  "5 healthy workers vs TRUE COUNT 0"). A count without its denominator is not a finding.
- **SCOPE reductions (legal):** you may read rather than execute where execution needs Metal — arms
  have no Metal device (the pp2 lesson). Static source reading at full coverage is the reference form
  for this layer; nothing here needs a GPU.
- **MECHANISM reductions (must be declared TOY-BRACKET):** sampling a subset of the 24 tests instead
  of reading all; inferring what a test asserts from its NAME instead of its body; reporting "N passed"
  as evidence of anything.
- **Provenance pins:** intake `SOURCE_REPO_HEAD e34f31bc4969042c0051ac81aa3c56884419a231`; ledger
  commit 3856788c96.

## NON-NEGOTIABLES

- Intake is READ-ONLY. Copy out; never edit in place, never `git add` inside it.
- MPS/MLX are NEVER score authority. You have no Metal device; do not claim device measurements.
- No number without a locatable receipt. If you cannot re-derive it from disk, say ABSENT — do not
  restate it. (Round 1's most serious finding was exactly this class: a d_pose figure with no receipt.)
- Every negative gets a verdict_scope (INSTANCE / FORMULATION / FAMILY). One failed formulation is
  not a dead family.
- Report the DENOMINATOR on every count.
- Commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per
  file, tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By` trailer.
- `REVIEW_GATE_OVERRIDE=1` is FORBIDDEN with `.py` files (use `review_tracker.py mark-file` ×2);
  acceptable for `.md`/`.json`.

## DELIVERABLE

One memo at `.omx/research/ddm_pr130_reproduce_20260809/RR1_PREPARE_VERIFY_AUDIT.md`:
per-element table {element · what it asserts · what it does NOT · denominator · verdict+scope},
then the ranked findings with falsifiers, then an explicit "what I could not check and why."
Honest non-findings required — if a surface is clean, say clean and show the coverage that proves it.
