# ddm_b2e landing — train-for-editability levers + edit-replay admission, and the charter OBJECT RE-PIN

Date: 2026-08-16 · Arm: ddm_b2e (supervised Opus bridge) · Owner of the FIRE: MAIN
Charter: `.omx/research/ddm_b2e_train_for_editability_burn2_charter_20260816.md`
Commit: `ec83c44223` (4 files, 1,947 insertions)
Axis: `[byte-only scorer-free]` throughout. **No score claim. No training launched. No Modal spend.
No n600 scorer pass. Scorer slot untouched** (verified free at start and never used).

---

## HEADLINE

The charter's regime thesis is **sound and unchanged**. Its *build target* is **falsified**, and the
falsification is material: as written, the charter would have fired a multi-day Metal burn that
trained **the wrong object** and could not possibly have made the mp2 edits free.

Three facts, each MEASURED, not argued:

1. **`experiments/ddm_rx2_mc36_label_hpac.py` is not a trainer.** Its sha256 matches the charter pin
   exactly (`4a0db3c8fd42…`), so this is not a wrong-file mix-up — the charter genuinely pinned this
   file. It is a Stage-0 telemetry + token-cache materializer: 347 lines, no model, no optimizer, no
   loop, and a single positional argument `{preflight,telemetry,cache,stage0}`. There is no argparse
   surface to add a lever to. "rx2" in this lineage is a **profile name** (`rx2_mc36`), not a trainer.

2. **The real HPAC trainer is `tools/train_ddm_cl1_hpac_capacity.py`** (1,389 lines, 38 flags,
   `--profile rx2_mc36`). It produced ep0634.

3. **But the tensors mp2 edited are not in ep0634.** The archive member carries **three** sections —
   `hpac`, `semantic`, `carrier` (`ddm_mp2_mixed_precision_receiver_close.py::split_member`, L145-173).
   ep0634 holds the **cl1 token model** (`conv_a`, `conv_b1`, `conv_b2`, `conv_past`, `spm_dw`,
   `spm_pw`, 5-class `head` — 37 state keys). The mp2 edits hit the **`SemanticTokenRenderer`** — a
   38-tensor object (`token_embed`, `frame_embed`, `coord_mix`, `blocks.{0..3}.{dw,pw,norm,film}`,
   `head`) trained by `src/tac/pr130_lift/train_semantic_quantized_resumable.py`.

**The executable proof of (3):** point the new edit-replay harness at the charter's own named
warm-start object and it refuses, fail-closed:

```
$ ddm_b2e_edit_replay_admission.py replay --checkpoint …/retained/epoch_0634.pt
B2EError: checkpoint state is not a SemanticTokenRenderer state: missing token_embed.weight
```

That is a measurement, not an opinion. The ns1 tensor names decode it: ns1 §A's
`18_blocks_1_film_weight` and `01_frame_embed_weight` come from mp2's retained dump naming
`f"{index:02d}_{safe_name}.npy"` (`ddm_mp2_mixed_precision_receiver_close.py` L449) — index 18 of the
38-tensor **semantic** state is literally `blocks.1.film.weight`. I listed the retained dump directly:
38 files, `00_token_embed_weight.npy` … `37_head_bias.npy`. None of them is a cl1 tensor.

**Consequence for the burn:** "warm-start ep0634 and train the weights to tolerate the mp2 edits" is
incoherent as written — ep0634 does not contain those weights. The correct burn warm-starts the
**semantic** trainer. MAIN owns that re-pin; this arm did not make it unilaterally.

---

## THE MECHANISM, SHARPENED (why the regime thesis survives)

ns1 said "the e960 burn ran QAT on TOKENS but never on the semantic WEIGHTS." The measured refinement:
the semantic trainer **does** run weight QAT — at **uniform int4**, and it refuses anything else:

```python
# src/tac/pr130_lift/train_semantic_quantized_resumable.py:708-709
if args.bits != 4:
    parser.error("--bits must be 4 because the deployed semantic packer is int4-only")
```

So the object was trained for a **uniform q4** grid and then edited post-hoc onto a **mixed q3/q4**
grid it had never seen. That is a sharper and more actionable statement than "never QAT'd": the cure
is not "add QAT," it is "**train on the grid you will deploy**" — the eval_roundtrip principle moved
into weight space. F2 implements exactly that.

---

## WHAT I BUILT

### 1. `src/tac/pr130_lift/editability_levers.py` — the five levers, on the CORRECT object

| lever | state | mechanism |
|---|---|---|
| **F1** weight-perturb robustness | BUILT | Per-step noise measured in each tensor's own quantiser steps, straight-through, with the pose-critical FiLM family up-weighted. |
| **F2** mixed q3/q4 weight QAT | BUILT | STE fake-quant on the **exact** mp2 mixed map. |
| **F3** FiLM structured row dropout | BUILT | Whole-row inverted dropout over `blocks.{1,2,3}.film.weight`, with top-N protection. |
| **F4** carrier rank penalty | BUILT | Nuclear-norm **normalised by Frobenius**, so it penalises spectral *concentration* rather than magnitude (an un-normalised nuclear norm degenerates into weight decay — a real trap avoided). |
| **F5** gate-aware conditioning | **DECLARED UNBUILT** | Blocker below. Never stubbed. |

Two design points worth flagging to MAIN because they are judgement calls, not derivations:

- **F1's default FiLM multiplier is `sqrt(93.7) ≈ 9.68`, not `93.7`.** ns1 measured the anisotropy at
  93.7× in Δd_pose per unit relative perturbation. Weighting by the *full* ratio equalises measured
  damage, which at this anisotropy collapses the FiLM rows outright; the square root equalises damage
  in the metric where perturbation enters quadratically, which is the regime quantiser noise lives in.
  This is **DERIVED-with-a-judgement-call**, and it is the first thing the arm matrix should sweep.
- **F2 uses `weight_qat_high_bits` as F1's step reference even when F2 is off**, so "one sigma" always
  means "one deployed quantiser step."

**Byte-identity contract, enforced and tested.** Every lever is default-off. When off: no tensor is
read, and — the subtle one — **no random number is drawn**. Active levers draw from a *dedicated*
`torch.Generator`, never the global stream, so a lever A/B is a clean 2×2 rather than a confounded one
(the base trainer's own sampling is bit-identical either way). `test_inactive_config_draws_no_randomness`
and `test_active_levers_do_not_disturb_the_global_rng_stream` pin both halves.

Each lever also emits an `activation_ledger()` row carrying `{active, params, reason_if_off}` and a
`steps_applied` counter, so "configured on but never fired" is detectable rather than silent orphaned
signal.

### 2. `experiments/ddm_b2e_edit_replay_admission.py` — the admission instrument

Stages `replay` / `pairs` / `admit`. It re-applies the **exact** mp2 edit constructions (reusing the
shipped `sd1`/`sm3`/`mz2` primitives — never reimplementing them), reports weight-space and byte
deltas, emits the seeded stratified pair set, and adjudicates measured pose against the pre-registered
bar.

**Pre-registered bar (fixed, in code, test-pinned at 50.0):** per edit, using each model's OWN base so
a shifted burn-2 base cannot flatter the result,

```
excess   = pose_edited / pose_base − 1
collapse = calibration_excess / burn2_excess     ADMIT iff collapse ≥ 50
```

Calibration excesses from the pinned MP2 adjudication: 3.959 (q3/q4), 3.638 (keep87), 2.767 (marginal).
Verdicts: `REGIME_THESIS_SUPPORTED` / `PARTIAL` / `INSTANCE_REFUTED` / `PENDING_MEASUREMENT`.

**Subset-bias handling (m96).** Pairs are chosen by seeded **stratified** sampling, never a prefix
(the smoke's n32 spans ids 4→598). Every bounded read is stamped: it **may refute** an admission
claim, it may **never** be the sole basis for granting one. The n600 advisory leg stays MAIN-fired.

### 3. SMOKE RESULT — the harness reproduces the ns1 §A calibration EXACTLY

Run on the real frontier archive (`…/ep0634/retained/candidate/archive.zip`, 182,759 B, sha
`80d9c8c6…`, pins verified):

| edit | ns1 §A ‖ΔW‖ (rel) | b2e replay | tensors touched |
|---|---|---|---|
| keep75∖keep87 | 0.2615 (0.35%) | **0.2615 (0.3479%)** | 3 FiLM ✓ |
| keep87 | 0.2102 (0.28%) | **0.2102 (0.2797%)** | 3 FiLM ✓ |
| mixed q3/q4 | 21.41 (28.5%) | **21.4141 (28.4908%)** | 4 (3 FiLM + frame_embed) ✓ |
| global base L2 | 75.16 | **75.1613** | 38 ✓ |

Four independent quantities reproduced to 4 decimals. This is the strongest available check that the
replay constructions are the shipped ones — and it independently re-derives the object finding, since
every touched tensor is a semantic-section tensor.

### 4. Tests — 66, all passing, behaviour not constants

`src/tac/tests/test_editability_levers.py` (39) + `experiments/tests/test_ddm_b2e_edit_replay_admission.py` (27).
The load-bearing ones:

- `test_deployed_fake_quant_matches_shipped_quantizer` — F2's forward output equals
  `sd1.quantized_tensor`'s restored tensor at **rtol=0, atol=0**, across 3 tensor shapes × {3,4} bits.
  Without this, F2 would be a proxy pretending to be the deployed grid.
- `test_prune_ladders_are_nested_and_row_structured` — keep87 and the keep75 marginal band are
  disjoint and match the shipped row order exactly.
- `test_constants_match_shipped_sets` — fails loudly if `SELECTED_MIXED_Q3_NAMES` / `PRUNE_NAMES` drift.
- Review-pass regressions: buffers untouched, parameter type restored, nested scopes clean.

Every test would fail if the implementation were replaced by a marker-returning stub.

---

## CHECKPOINT VERIFICATION (charter question, answered MEASURED)

`/Volumes/VertigoDataTier/pact/ddm_hv1_harvest_compose/retained/epoch_0634.pt`, 1,103,503 B,
sha256 `5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec` — **matches the charter pin.**

**The optimizer state IS retained.** Top-level keys:

```
schema=ddm_cl1_hpac_capacity_checkpoint.v2 · epoch=634 · phase=discrete_qat · qat_start=481
state_dict(37, EMA shadow) · live_state_dict(37) · ema(4) · ema_policy(6)
optimizer_state_dict(2)  ← present · scheduler_state_dict(8) · rng(5)
best(11) · history(318) · run_identity(24) + sha · resume_lineage(1) · causal_state_sha256
```

So the **wd3 law (Adam state carries ~3× pose descent) is available** — for the cl1 token object. The
caveat MAIN must hold: this is the token model's optimizer state. If the burn re-pins to the semantic
trainer, the relevant question becomes whether the *semantic* run's optimizer state was retained, which
is a different artifact and is **NOT verified by this arm** (NEXT_IF_RESUMED #1).

---

## SEALED TICKET STATE

`.omx/research/ddm_b2e_sealed_launch_ticket_20260816.md` — **BLOCKED_ON_OBJECT_REPIN**, not
READY_TO_FIRE. The blocker is not budget or apparatus; it is that two of the charter's four launch
pins name objects that cannot do the job. Gate checklist in the ticket.

---

---

# ADDENDUM — MAIN's re-pin EXECUTED (same session, commit `277fc58d13`)

MAIN adjudicated the premise falsification CORRECT and decided both blocked gates. Results:

## The warm-start object is INHERITED — the case MAIN anticipated

**`b489c735…` = sha256 of `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/pr135/canonical/semantic.wans1` (36,051 B). EXACT MATCH.**

Chain: `mz2._load_records()` decodes `F12_BODY`, asserts equality with that canonical file, and those
records ARE the shipped semantic state (mp2's `parser="legacy"` returns the template unchanged).

**We never trained the semantic weights.** They are PR135 intake weights. Measured consequences:

- hb1 `checkpoints/{gt,tq1c}/` holds **only HPAC token models** — all 10 score 2/5 on
  SemanticTokenRenderer key overlap (coincidental `frame_embed`/`head`), none semantic.
- **No optimizer state exists anywhere for the semantic object.** Fresh Adam.
- **The wd3 warm-carry law does NOT apply.** This is a genuine downgrade vs the charter's assumption
  and the window must be priced for it.
- It also sharpens ns1: the object wasn't "never QAT'd", it was QAT'd *by PR135 at uniform q4* and our
  edits move it to a mixed q3/q4 grid it has never seen.

## The `--init` object — located, 37/38 exact, NOT bit-exact

`semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`, sha `3948ccfc…` — **exactly** sd1's pinned
`EXPECTED_CHECKPOINT_SHA256`. `q4(init)` reproduces the shipped state on **37/38 tensors exactly**.

`blocks.3.film.weight` differs in **2 of 1,536 elements** (rows 87, 189). Checked and **not** rounding
tie-breaks: normalized 7.0015 ships as code 5; normalized 0.6167 ships as code **−1** (sign flip).
No zeroed rows, so not a prune. Reading: 2 deliberately-modified elements in the pose-critical FiLM
family, likely a downstream targeted correction or a marginally later ship. Almost certainly immaterial
to training, but **the warm-start is not bit-exact and the ticket does not claim it is.**

## Wiring landed

F1–F4 are default-off flags on `train_semantic_quantized_resumable.py`. Two findings shaped it:

1. **`quantized_forward` already fake-quantizes every parameter via `functional_call` at a uniform
   `bits`.** A naive parameter swap would have **double-quantized**. The levers therefore OWN the
   quantization on the lever path, which calls `render_float` (identical tail, no second quantize).
2. **The `base_bits` guard is load-bearing.** If quantization applied only when F2 is on, enabling F1
   or F3 alone would have *silently dropped QAT* — an effect that would look like a lever result but
   would actually be the absence of quantization. `applied(base_bits=…)` now always quantizes, at the
   mixed map when F2 is on and at `--bits` when it is off. Verified: levers-off `parameter_overrides`
   equals PR130's `quantized_forward` parameters **exactly**.

`--bits` stays hard-pinned at 4; F2 requests the mixed grid through its own flag, so the uniform path
is untouched.

**Additive resume guard.** `_reconcile_additive_resume_config` drops a lever key from the comparison
only when the checkpoint predates it AND the current run leaves it inert. Active lever vs pre-lever
parent still refuses; a vanished key still refuses; a non-lever difference still refuses. The guard is
extended, never bypassed. 17 tests pin it.

Totals: **83 tests green**, ruff clean. The whole-`src/tac/tests` collection error
(`conflicting in-process LawRef evaluator for realization_breakeven_bytes_v1`) is **pre-existing** —
reproduced with my change stashed out.

## LIVE HYPOTHESES

1. **The regime thesis is intact and now sharper.** The object was trained for uniform q4 and edited
   onto mixed q3/q4 it never saw; F2 removes exactly that mismatch. Falsifiable by the pre-registered
   50× bar.
2. **F1's FiLM up-weighting is the highest-variance choice in the build.** `sqrt(93.7)` vs `93.7` vs
   `1.0` is a 3-point sweep that plausibly decides whether the burn works. Sweep it; do not assume it.
3. **The two-object structure may itself be a rate lever nobody has priced.** `hpac` (token model),
   `semantic` (renderer), `carrier` — three sections, three independent editability regimes. ns1's
   §A screen was measured on the semantic section only. Whether the *carrier* section (F4's 22,032 B
   target) has the same ~94× anisotropy is **unmeasured**.
4. **F1 and F2 may be partially redundant.** Quantisation-shaped noise at one q4 step and STE
   fake-quant at q3 both push toward grid-tolerance. A 2×2 (not the diagonal) is required.

## DEAD ENDS (do not re-walk)

- Adding levers to `experiments/ddm_rx2_mc36_label_hpac.py`. No argparse surface, no model, no loop.
- Warm-starting a *semantic* editability burn from ep0634. The harness refuses it, fail-closed, and it
  is right to.
- Reimplementing the edit constructions. `sd1.pack_semantic_state` + the mp2 row-prune ladder are the
  shipped ones and the replay reproduces ns1 to 4 decimals by reusing them.
- Decoding the base semantic state via `sd1.unpack_semantic_state`: the hv1 base section rides the
  **legacy** parser, where `expected_state(parser="legacy")` returns the template unchanged. Using the
  SD1M parser raises `truncated signed code stream` (hit and fixed during the smoke).

## NEXT_IF_RESUMED (updated after the re-pin)

1. ~~Locate the semantic warm-start + optimizer state.~~ **DONE** — inherited PR135 intake; no
   optimizer state anywhere; `--init` pinned at sha `3948ccfc…` with a documented 2-element delta.
2. ~~MAIN decides the re-pin; wire the levers; handle the resume guard.~~ **DONE** (`277fc58d13`).
3. **Close PIN-1 (`--challenge-root`) and PIN-2 (`--cache`)** — the two mechanical lookups blocking
   the F2-alone row. sd1 pins the cache sha as `EXPECTED_OFFICIAL_ADA_CACHE_SHA256`; find the local
   artifact matching it.
4. **Take the ~50-step timing + memory smoke** at the real config. Wall-clock is deliberately NOT
   derived in the ticket — I have no measurement and refuse to invent one.
5. **Investigate the 2-element `blocks.3.film.weight` delta.** 2 elements of 1,536, one a sign flip,
   in the pose-critical family. Either a downstream targeted correction (a PR101-style single-element
   sidecar) or a later ship. If it is a *correction*, it is itself evidence that targeted 2-element
   FiLM edits are viable — which would be a live rate lever, not a curiosity.
6. **Build F5 (gate-aware conditioning).** Blocker unchanged: needs the js8 gated application
   distribution, not derivable from any receipt this arm holds. Named, not stubbed.
7. **Sweep F1's FiLM multiplier** {1.0, sqrt(93.7), 93.7} and run the F1×F2 2×2 off-diagonal.
8. **Wire the scorer leg** via the proven `ddm_mp2_advisory_queue.py --manifest --output-root`; add
   manifest emission to `replay` rather than reimplementing a scorer loop.
9. `--out-dir` retention is opt-in and unenforced against the SSD tiers; add a storage-tier guard
   before any retention run.
10. **Phase-B (#850 cap-lift) remains NOT BUILT** — still requires #850 and qs5 at source.

## STORES CONSULTED

ns1 memo (sha `91741c06…`) · MP2_ADVISORY_ADJUDICATION.json (sha `54228227…`, read + pinned into the
harness) · the retained mp2 `semantic_state` dumps (38 tensors, listed directly) · the b2e charter ·
`ddm_hv1_harvest_compose.py` CANDIDATES · ep0634 checkpoint (loaded + key-inspected) · the frontier
archive (sha-verified) · `ddm_rx2_mc36_label_hpac.py` (read in full) ·
`tools/train_ddm_cl1_hpac_capacity.py` · `train_semantic_quantized_resumable.py` ·
`ddm_mp2_{semantic_receiver,mixed_precision_receiver_close}.py` · `ddm_mz2_*` · `ddm_sd1_*` ·
`ddm_sm3_*` · `semantic_renderer_oracle.py` · `upstream/modules.py` (read-only) ·
`tools/codex_arm_queue.py status` (scorer slot free).

**Not consulted, and it matters:** the wd2/wd3 verdict memos, the js8 handoff, the #850 row and the
qs5 verdict memo were consulted only through ns1's summaries of them, not at source. The Phase-B
harness (charter deliverable 3) depends on #850 and qs5 at source and is therefore **NOT BUILT** —
declared unbuilt with that exact blocker rather than guessed at.
