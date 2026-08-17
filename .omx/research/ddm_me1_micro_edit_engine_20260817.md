# ddm_me1 — THE MICRO-EDIT ENGINE: built, and pointed where it can actually decide

Date: 2026-08-17 · Arm: `ddm_me1_micro_edit_engine_20260817` · Authority: research + exact
local arithmetic + decode-identical code-length measurement · **Score claim: false** ·
**Pointer moved: false**

Operator binding: *"Seems like we could build and or train a tool to explore and optimize and
apply all micro edits that lower score."* Amendment 1: *"Seems like it's signal that could be
used to optimize the learned prior too."*

## Conclusion first

I built the engine and used it to answer one question the campaign could not otherwise answer
for free. The headline is a **negative with a proof attached**, plus a re-derivation that
changes where the campaign should aim.

1. **The stage-5 opening job was void, and I verified that myself.** The banked qs2 / re1
   edits are already inside the frontier row. Not relayed — read at source.
2. **The gap is 97.20% pose, but perfect pose no longer closes it.** At the rr4 base, zeroing
   d_pose leaves **0.00023867 S** — exactly **281.55 net Seg flips or 358.44 bytes**. At CP135
   that residual was 4,319 flips. It fell **15.3×** because the rate term fell. Micro-edits are
   now a *credible closer*, where before they were rounding error.
3. **The semantic micro-edit families are at their measured asymptote on this base.** Best ever
   realized: mc36 union at −2.068e-5, already shipped. The two attempts after it REFUSED
   (qs5 +2.52e-6, qs4 +2.44e-4). Both hit an identical **−17 net realized flip ceiling** from
   100 and 132 changed pixels; the seg model over-predicts **3.4×**.
4. **So I pointed the engine at the one axis it can decide alone: the coder.** A change to the
   probability law is decode-identical, so distortion is unchanged *by construction* and bytes
   are exactly measurable locally at zero cost. I built that instrument, **verified it against
   two independent shipped constants to 0.000000**, and raced four context architectures.
5. **All four lose.** Best is +359 B. And the reason is a theorem, not a mystery: our
   platform-exactness constraint permits only *averaging* rules, and **an average can never
   beat its best member.** That is the crux, and it names the exact next build.

**Own-vehicle frontier: `S = 0.15853325034789678 @ 181,161 B [contest-CUDA T4, n600]`, archive
sha `35ac2b9b…`. This arm did NOT move it.** No Modal spend; $1.38 headroom untouched.

## 1. Verifying the void opening job (STORES CONSULTED at source)

The charter's first job was to recompile the banked qs2/re1 edits against the rr4 coder. I was
told this was void and told to trust nothing relayed. I checked:

* `.omx/research/falsified_premise_registry.jsonl` → `qs2_re1_bank_union_is_held_and_unfired_20260817`,
  `verdict_scope: instance`, registered by ddm_bu1.
* Its falsification rows: the union *was* built 2026-08-14 as `ddm_mc35`, repaired as `ddm_mc36`
  Variant C, fired on T4, **PROMOTED at realized ΔS −2.068040e-5**. Subsumption proven at EVENT
  ID: mc36's parse-back recovers compensation pairs `[7, 96, 105, 176, 178, 517, 523]` = qs2's six
  minus the measured-harmful 532, union re1's 96 and 7.
* The qs2 and re1 memos both carry CONSUMED banners added 2026-08-17.

**CONFIRMED VOID.** Recompiling would double-count value the frontier already holds. The registry
also records that the hold was *arithmetically correct*: the naive union is −5.581792e-6, i.e.
55.8% of the naming bar — compute was never warranted.

One portable law from that consumption, which the engine's compose stage now encodes: mc36 beat
the naive leg-sum **3.705×** by re-solving compensation *jointly* over the composed object. Never
price a union as the sum of its legs.

## 2. The axis split, re-derived at the live base (not quoted)

`ScoreState.axis_split` recomputes this from the base rather than carrying a constant, because
the residual moves whenever the rate term moves. At d_seg 0.00029611, d_pose 6.88e-6, 181,161 B:

| quantity | value |
|---|---:|
| S (reconstructed from components) | 0.15853325034789677 (published: …678) |
| seg term | 0.02961100 |
| pose term | 0.00829457654133109 |
| rate term | 0.12062767380656568 |
| gap to 0.15 | **0.00853325034789677** |
| **pose share of the gap** | **97.2030%** |
| residual after PERFECT pose | **0.00023867380656568** |
| …in net Seg flips | **281.55** |
| …in bytes | **358.44** |

Independently reproduced by rv2 (pose→0 saves 0.00829458 vs gap 0.00853325; seg alone needs
−28.8%; rate alone needs −12,815 B → archive ceiling 168,346 B). Two derivations, same numbers.

The model also cross-validates against a third party: fed qs2's measured legs it returns
**−4.37491476e-6** where eu4 derived **−4.374914e-6** by a different route.

**Consequence for allocation.** Pose is still the single largest term, but it is no longer
*sufficient*: a perfect pose solve lands at 0.15023867, still above target. The campaign needs
pose **plus** ~282 flips or ~358 bytes. That closing 2.8% is exactly micro-edit scale — which is
what makes this engine worth having, and also what makes its current asymptote a real problem.

## 3. The engine

`src/tac/micro_edit/` + two experiment correctors. It unifies rather than rebuilds: it shells out
to the existing compile chain and drives the *real* shipped corrector, so the live-law control
cannot drift from what ships.

| module | role |
|---|---|
| `score_model.py` | exact `Decimal` contest arithmetic; base-independent marginals derived from `tac.contest_oracle.constants`; state-dependent pose marginal; union-gating law in code |
| `ledger.py` | append-only fcntl-locked edit-outcome corpus; refuses a realized byte-moving row with no retained payload |
| `candidate.py` | typed `EditCandidate` + generator protocol; fingerprint excludes estimates so plugins de-duplicate safely |
| `coder_replay.py` | **the decode-identical measurement surface**; full n600 in 20 s |
| `ddm_me1_spatial_context_corrector.py` | causal-spatial context as a joint factor |
| `ddm_me1_mixed_context_corrector.py` | context-family mixture over the shipped transport law |

Three design points earned by measurement rather than taste:

* **`compose_deltas_unverified` returns `realized=False` unconditionally** and refuses mixed
  bases. eu4's union law and qs4's +2.4e-4 cross-object refusal, enforced in code.
* **`delta_s` is the true nonlinear difference, never a linearisation.** A test proves the sqrt's
  concavity means linearising *under*-sells a pose reduction — the error runs the direction that
  would make us skip a good pose move. (My first draft of that test asserted the sign backwards;
  the test caught it.)
* **The ledger guard fired on me during this very arm**, refusing four ablation rows whose
  payloads I had not kept. I re-ran them and kept the bytes.

### Why the local instrument cannot admit a token-changing edit

MEASURED, and it reframed the whole arm: the local advisory n600 CPU chain reports
**d_seg 0.00042714 / d_pose 0.00014747** where the T4 reports **0.00029611 / 6.88e-6**. Pose is
**21× off**. Locally, only **bytes** are authoritative. Any family that changes tokens must buy a
T4 row to be admitted — at $1.38 remaining, roughly 8 rows for the whole campaign.

A change to the *probability law* is different in kind: the decoded token field is bit-identical,
so distortion is unchanged by construction and bytes are exact locally. That is the one axis the
engine can decide by itself, and it is where I spent the budget.

## 4. Positive controls — the instrument first

No verdict from this module is admissible unless both hold. Both were run on the full n600 field:

| control | target | measured | delta |
|---|---:|---:|---:|
| uncorrected HPAC cross-entropy | 112,109.57757858819 B | 112,109.57758 B | **0.000000 B** |
| shipped rr4 corrected code length | 884,090.2210952122 bits | 884,090.2211 bits | **0.000000 bits** |
| single-family mixture ≡ shipped law | 884,090.2210952122 bits | identical | **0.000000 bits** |

The third is the refactor control: my mixture generalisation collapses *bit-exactly* onto the law
it generalises, so any measured difference later is the new family, not my plumbing.

## 5. The coder race — four architectures, all negative

Our shipped rr4 law is **already opal-class**. Reading it at source:
`q = p_max·m/(p_max·m + one_minus)`, `P'(c) = (1−q)·P(c)/(1−P(argmax))` — a rank-one split into
the maximal projector and its complement, learning only the transport between the sectors and
preserving the inherited relative law inside the complement. That is PR 138's described mechanism.
The difference is not the transport form. It is that ours uses **one** joint context of 51,200
bins and theirs mixes **55 families**.

I tested that difference directly. All rows are full n600, decode-identical, exact:

| architecture | contexts | token bytes | Δ vs live |
|---|---:|---:|---:|
| **shipped rr4 (live control)** | 51,200 | **110,511.28** | — |
| + spatial as 6th joint factor | 256,000 | 110,578.40 | **+67.12** |
| mixture: shipped + temporal_spatial | 2 families | 110,870.75 | **+359.47** |
| mixture: shipped + spatial_boundary | 2 families | 111,072.49 | **+561.21** |
| mixture: shipped + spatial_surprise | 2 families | 111,174.10 | **+662.83** |
| mixture: shipped + surprise_only | 2 families | 111,200.39 | **+689.11** |
| 5-family, quality-weighted | 5 families | 111,427.61 | **+916.33** |
| 5-family, count-weighted | 5 families | 111,909.54 | **+1,398.26** |

Three findings, in the order I learned them:

**(a) The context is not diluted.** I predicted the joint context starved its bins. Measured on
the retained final state: only **0.0516%** of observations sit below `MIN_COUNT`; 90% live in 72
contexts; global hit rate 0.998104. Hypothesis refuted before I built on it.

**(b) Refining a joint context costs more than it informs.** 51,200 → 256,000 bins for one crude
spatial feature: +67 B. Conditioning is never free.

**(c) Count-weighting rewards crudeness.** `surprise_only` has 320 bins, so its per-bin counts run
to millions and it drowns the refined family. Adding a measured-quality term recovered 482 B
(+1,398 → +916) — the right direction, still a loss.

### The theorem that explains all of it

A weighted arithmetic mean is bounded by its members: `min(m_k) ≤ Σw_k m_k / Σw_k ≤ max(m_k)`.
So when one model is already best, **blending it with weaker ones can only move the estimate away
from the right answer.** Averaging cannot beat its best member. It is now a test.

Real context mixing does not average — it **sharpens**, combining models multiplicatively in
log-odds (PAQ-style logistic mixing with online-learned weights). That needs `log`/`exp`.

**And that is exactly what we forbade ourselves.** `ddm_rr4_free_corrector_v2` exists because one
`log2`/`exp2` round trip in v1 perturbed 50.09% of positions by a float32 ULP and desynchronised
the T4 decoder — the rr2 refusal at S 27.83. So:

> **The crux: our platform-exactness constraint admits only averaging rules, and averaging
> provably cannot beat the best single model. Opal-class gains need a sharpening mixer, which
> needs transcendentals, which broke us once already.**

This is a located crux, not a wall. The cure is a **fixed-point integer log-odds mixer**: a frozen
integer stretch/squash table plus integer arithmetic is deterministic *by construction* — more
exactly reproducible than the float path, not less. PR 138 shipping 49.4 MB of regenerated adaptive
state says the winning architecture is a large adaptive mixer, not a bigger context. That is a real,
sized target: their token stream fell 114,706 → 110,022 B (−4.08%); our corrector took −1.43%.

## 6. Per-family asymptote table (the charter's falsifier-honesty output)

| family | best realized | disposition |
|---|---:|---|
| qs2 coupled compensation | −4.375e-6 @ +34 B | **CONSUMED into mc36** |
| re1 realization flips | −1.207e-6 @ 0 B | **CONSUMED into mc36** |
| mc35/mc36 union | **−2.068e-5** | **SHIPPED — in the live row** |
| qs5 in-compile Schur | +2.520e-6 @ +26 B | REFUSED; −17 flips from 132 px |
| qs4 collateral suppression | +2.438e-4 @ +28 B | REFUSED; stale compensation |
| coder: joint-context refinement | +67.12 B | REFUSED (this arm) |
| coder: arithmetic-mean mixture | +359.47 B best | REFUSED (this arm) |

Semantic edits: two consecutive REFUSALS after the union, both pinned at −17 realized flips with
a 3.4× model-over-prediction. That is a family at its asymptote on this base.

**No candidate clears the 1e-5 naming bar. I am not sealing a fire-order and not manufacturing a
candidate.** The $1.38 / ~8 remaining T4 rows stay unspent, which is the correct call: nothing here
would survive one.

## 7. NEXT_IF_RESUMED, ranked by expected ΔS per unit effort

1. **Fixed-point integer logistic mixer.** Frozen integer stretch/squash tables + integer weight
   update. Platform-exact by construction. This is the direct cure for the located crux and the
   only architecture the evidence says can beat the shipped law. Target: −1,000…−3,000 B
   (−6.7e-4…−2.0e-3 S, 67–200× the naming bar). Fully decidable locally at $0 on the instrument
   this arm leaves behind — 40 s per architecture.
2. **Widen the family set only under the sharpening mixer.** Spatial families are not worthless;
   they are unusable under averaging. Re-test all four once (1) exists.
3. **Charter amendment 1 stage 7(b), counted-weight refit.** Now correctly ranked *below* (1):
   PR 138 shows the zero-counted-byte adaptive route already outperforms, and a counted refit must
   additionally pay its own weight cost.
4. **Semantic edits: stop, or fix the 3.4× realization gap first.** Two refusals at an identical
   −17 ceiling means the generator proposes flips the render→SegNet round trip does not realize.
   Model realization before proposing again.

**Row-count threshold for stage 7(a) (charter asks me to state it):** the score-equivalence map
needs ≥30 realized token-changing rows before it is honest, because the −17 ceiling reproduced
twice on n≈100–132 sites and a map built on fewer would inherit that unmodelled 3.4× realization
gap. I have 0 such rows and could not buy them at $1.38. The coder axis needs none, which is the
second reason it was the right target.

## STORES CONSULTED

`.omx/research/falsified_premise_registry.jsonl` (qs2/re1 void, verified at source) ·
`ddm_rr4_t4_verdict_pointer_move_20260817.md` (live base) ·
`ddm_rr4_cuda_prob_reencode_20260817.md` + `RESULT_build.json` (sections, token stream) ·
`ddm_eu4_fresh_eyes_fractal_composition_20260813.md` (marginal ladder, union law) ·
`ddm_qs2_r2_admitted_verdict_20260813.md` · `ddm_qs4_collateral_suppression_20260813.md` ·
`ddm_qs5_resolve_compensation_20260813.md` · `ddm_re1_round1_dual_axis_verdict_20260814.md` ·
`ddm_rv2_frontier_adversarial_review_r1_20260817.md` (routing arithmetic, Modal headroom) ·
`experiments/ddm_hm1_hpac_logit_replay.py` (logit→probability map, cross-entropy constant) ·
`experiments/ddm_rr4_free_corrector_v2.py` (the live law) ·
`upstream/evaluate.py` + `tac.contest_oracle.constants` (score definition) ·
public PR 138 `opal_v1` body (mechanism class, read not copied) ·
retained corrector state, token field, logits, boundary, group index (all sha-verified on disk).

## Artifacts

Payloads: `/Volumes/APDataStore/pact/ddm_me1/` — `ME1_CODER_RESULTS.json`, eight retained
per-frame bit vectors (one per architecture), `table_values.npy`.
Ledger: `.omx/research/micro_edit_outcome_ledger.jsonl` (7 realized rows, every one payload-backed).

## Landing blocker (open, NOT caused by this arm)

The memo and the ledger landed at `679506f4e3`. The ten engine `.py` files are on disk,
ruff-clean, 19/19 tests green (16 light + 3 heavy under `TAC_ME1_HEAVY=1`), and carry their two
review-tracker passes — but they are **NOT COMMITTED**. The pre-commit hook's CI-blind step
refuses, and the refusal is unrelated to this arm:

```
src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_execute_runs_training_archive_and_receiver_proof
Fatal Python error: Bus error
Fatal Python error: Segmentation fault
```

That is a pre-existing MLX/Metal test. Every file this arm adds is NEW, so nothing here can
change its behaviour; the hook selects it merely because `src/tac/**` is staged. It passes when
run in a small group and hard-crashes in the 31-node selection under concurrent MLX load (3-5
sibling pytest processes were live throughout). Signature matches the known
`concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806` genus.

I did NOT set `PREFLIGHT_SKIP_CI_BLIND_TESTS=1`. The gate is the only surface that runs these
modules, and an arm silencing it to land its own work is exactly the bypass that memory names.

**Handoff — run when the machine is quiet (no sibling MLX pytest):**

```bash
FILES=(src/tac/micro_edit/__init__.py src/tac/micro_edit/score_model.py \
  src/tac/micro_edit/ledger.py src/tac/micro_edit/candidate.py \
  src/tac/micro_edit/coder_replay.py src/tac/micro_edit/tests/__init__.py \
  src/tac/micro_edit/tests/test_score_model.py src/tac/micro_edit/tests/test_coder_replay.py \
  experiments/ddm_me1_spatial_context_corrector.py \
  experiments/ddm_me1_mixed_context_corrector.py)
ARGS=(); for f in "${FILES[@]}"; do S=$(shasum -a 256 "$f" | awk '{print $1}'); \
  ARGS+=(--expected-content-sha256 "$f=$S"); done
.venv/bin/python tools/subagent_commit_serializer.py --timeout-seconds 2400 \
  --message "ddm_me1 engine code: exact-arithmetic score model + payload-guarded ledger + typed candidates + decode-identical coder replay (controls exact to 0.000000) + two context-architecture correctors [no-triality] [p0-ledger-ok]" \
  --files "${FILES[@]}" "${ARGS[@]}"
```

If it refuses again with the same node, that MLX test is a standing repo-wide commit hazard for
anyone touching `src/tac/**` and deserves its own arm — not a skip flag.
