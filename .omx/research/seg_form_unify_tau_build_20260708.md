# `--seg-form-unify-tau` — dissolve CE→tau_softplus into ONE continuous L_τ (build)

- **UTC:** 20260708 · **Agent:** UNIFY-TAU BUILD (Opus) — recovery-landed · **[no-triality]**
- **review_status:** recovery-written, fresh-eyes-reviewed(1)
- **Authority:** `[macOS advisory / build + unit tests]` — `$0`, NO launch, run dirs read-only.
  Pointer UNMOVED contest-CPU **0.19109982**. This is a MEANS (a build + a pre-registered A/B), not a
  score row. Pointer moves only through a byte-closed `upstream/evaluate.py` n600 exact row.
- **Charter:** `.omx/research/witness_native_schedule_derivation_20260709.md` Phase-3 element-1 BUILD
  (the CONTINUOUS verdict; `L_τ = τ·logsumexp(φ/τ) − φ_y`; τ=1 ≡ CE, τ→0 ≡ max-margin; HYBRID restart rec).

## STORES CONSULTED
- The derivation memo above (Phase 1 §1.2 the loss-family derivation; Phase 2 element-1 STRUCTURAL
  DIVERGENCE; Phase 3 the ~60–100 LOC BUILD + geometric flip + KEEP list + pre-registered A/B).
- `corpus_query` / grep over the incumbent trainer: `_seg_form_for_epoch` (L1817, the hard
  `ce`→`tau_softplus`→`l7_softplus` dispatch, comment "PR95 d_seg sequence"), the `make_loss_fn` seg
  branches (`ce` / `tau_softplus` / `l7_softplus` / `margin_hinge`), the pre-existing `tau_override`
  loss param (FEED-ca intra-stage anneal), `_softmax_temp_for_epoch` (the render τ anneal incl.
  `geometric`), the event-triggered controller (`_evt_resolve_seg_form`), the curriculum
  transition-easing (`--stage-transition-rewarmup-epochs` / `--...-reset-moments`) and the Muon-entry
  easing (`muon_switched`, a SEPARATE trigger).
- MEASURED constants (harvested as physics, from snippets): through-R per-stage d_seg
  **CE 0.01045→0.005443, τ_softplus(0.3)→0.004563**; the FEED-ft ep300 loss-form bump 0.0056→0.020
  (3.4×) on the amort decoder; knee **τ\*≈0.31**.

## WHAT LANDED (completion vs the 6-part charter)
1. **`--seg-form-unify-tau` (store_true, default OFF).** When ON, `_seg_form_for_epoch` short-circuits
   to a CONSTANT `"unify_tau"` for every epoch (BEFORE the `--curriculum` branches), so no discrete
   CE→tau_softplus boundary exists. The dissolved boundary's transition-easing (spike-guard clear / LR
   rewarmup / moment-reset) never fires — it triggers only on `seg_form != prev_seg_form`, and
   `prev_seg_form` is initialised to `"unify_tau"` so the constant form never crosses a boundary. The
   event-triggered controller is bypassed (`seg_form = "unify_tau"; _evt_event = None`). The
   **Muon-entry easing is a SEPARATE trigger (`muon_switched`) — intact**.
2. **Numerical equivalence + the mapping (never fudged):** see §MAPPING MATH below. Unit-tested.
3. **`--tau-anneal-shape geometric`** — VERIFIED already present (`choices=["cosine","geometric",
   "cosine_hold"]`, log-spaced `τ = start·(end/start)^prog`, with the `>0` fail-closed guard). No build
   needed; the derivation's element-2 flip is a 0-LOC config choice.
4. **DSL `SegFormUnifyTau()` Lever factory** (`overrides={"--seg-form-unify-tau": True}`, `epochs_delta
   0`) + `lever_registry.completeness()` maps `--seg-form-unify-tau` (AST-derived; asserted NOT in
   `.unmapped`).
5. **Mutual exclusion:** `validate_seg_form_unify_tau_config` REFUSES an explicit
   `--tau-softplus-start-epoch` when unify is ON (detected on the CLI string, not the value — argparse
   default 300 is indistinguishable from a user 300). Fail-LOUD `ValueError`, not silent. The sister
   `--l7-start-epoch` is inert-but-not-gated (documented; defaults to a `>=epochs` "never" sentinel and
   never contradicts).
6. **This memo.**

Recovery gap-fills (prior builder DIED ~63 tool-uses, work uncommitted): (a) the post-loop `final_form`
checkpoint tag now yields `"unify_tau"` under unify (was mis-tagging the final ckpt with the stale event
stage-0 `"ce"` when unify+event-triggered co-occur — cosmetic tag, not correctness/byte-identity;
guarded by `_unify_tau_on` so DEFAULT-OFF stays byte-identical); (b) this memo.

## MAPPING MATH — CE / tau_softplus are the τ endpoints of ONE family (documented, not fudged)
Per-pixel kernel `_seg_unify_tau_perpixel(φ, y, τ) = τ·logsumexp_k(φ_k/τ) − φ_y`.
- **τ = 1:** `= logsumexp(φ) − φ_y` = the `ce` branch's per-pixel base **EXACTLY** — `φ/1.0 ≡ φ` and
  `1.0·logsumexp ≡ logsumexp` for finite IEEE floats, so at τ=1 the `unify_tau` branch (which carries
  the SAME `w = 1+hinge·exp(−clip(margin))`, `apply_mw`, `seg_pixel_w` wrapping as the `ce` branch)
  reproduces the FULL `ce` branch bit-for-bit.
- **τ → 0:** `τ·logsumexp(φ/τ) → max_k φ_k`, so `L_τ → max_k φ_k − φ_y = ReLU(−m)`,
  `m = φ_y − max_{k≠y}φ_k` (the top1−top2 margin) — the max-margin / perceptron / hinge form.
- **vs the incumbent `tau_softplus = τ·softplus(−m/τ)`:** that branch is the **TOP-2 REDUCTION** of the
  multi-class `L_τ` — keep only the two dominant logits in the logsumexp and
  `τ·logaddexp(φ_y/τ, φ_r/τ) − φ_y = τ·softplus(−m/τ)`. They **COINCIDE exactly** as τ→0 and in the
  2-class case; at moderate τ (e.g. 0.31) the full-multiclass `L_τ ≥ tau_softplus`, the gap being the
  sub-runner-up logsumexp mass over the other classes. So `L_τ` is the **PARENT** whose top-2 marginal
  IS `tau_softplus` — NOT a rescaling of it at general τ. (Unit tests: τ=1≡CE across seeds; τ=1e-3→ReLU(−m)
  within 5e-2; 2-class limit ≡ tau_softplus at τ∈{1,0.5,0.31,0.1}; `L_τ ≥ tau_softplus` at τ∈{0.31,0.5}.)

## DEFAULT-OFF byte-identity (the sharpest edge — live run-1 pid 63069 resumes re-importing THIS module)
Every new path is guarded so flag-absent is byte-identical: `_seg_form_for_epoch` short-circuit is
`if getattr(args,"seg_form_unify_tau",False): return "unify_tau"` (False ⇒ falls through to the unchanged
discrete dispatch); `_unify_tau_on = False` ⇒ `_uni_tau = None` ⇒ `base_loss(..., tau_override=None, ...)`,
and `tau_override` is a PRE-EXISTING loss param (default None ⇒ `tau_use = tau_softplus_tau`) so passing
`None` explicitly ≡ the old call that omitted it; `prev_seg_form`, the in-loop `seg_form` assignment, the
coupling cell write, and `final_form` are each gated on `_unify_tau_on`. A run-1 resume from its frozen
`launch.sh` (which lacks the flag) therefore takes the exact discrete path unchanged.

## PRE-REGISTERED FALSIFICATION (settled ONLY by a measured through-R n600 d_seg trajectory)
- **Arm U (DERIVED):** `--seg-form-unify-tau` + `--tau-anneal-shape geometric` + the KEEP list
  (floor 0.31, tail, LADDER incl. fitted per-pair σ_ij, Muon finisher, event-triggering).
- **Arm D (CONTROL):** the incumbent discrete-stage config (or the prior through-R discrete trace as the
  historical baseline, since run-1 produced NO trajectory).
- **Success for Arm U:** reach the CE-arc floor (≈0.00475–0.00544) with **NO loss-form transition bump**
  at the τ≈0.3 crossing (vs the incumbent 0.0056→0.020, 3.4×), descending monotonically to the τ\*-floor
  (≈0.004563) or below.
- **FALSIFICATION:** if Arm U is WORSE than Arm D at the τ\*-floor on through-R d_seg (vs baseline
  **CE 0.01045→0.005443, τ_softplus(0.3)→0.004563**), the CE/tau_softplus discretization was
  load-bearing (contra the math) ⇒ **REVERT the loss unification, KEEP only the geometric shape flip.**
- Measure both n600, byte-closed through R (AXIS-9). Pointer 0.19110 UNMOVED until a byte-closed exact row.

## NOT DONE (honest)
No launch (operator-gated; live run-1 + dashboard untouched). No exact-eval row. The A/B above is
pre-registered, not run. `[no-triality]` per charter.
