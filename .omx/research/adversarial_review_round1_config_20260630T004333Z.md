# Adversarial Review — ROUND 1, LENS B (CONFIG / CORRECTNESS) — v2 witness program

**UTC:** 20260630T004333Z · **Axis:** `[$0 CPU read-only design-audit / advisory]` · **Pointer UNMOVED: contest-CPU 0.19110.**
**MEANS, not ends.** This finds config/correctness bugs in the trainer/DSL/codec BEFORE the GPU run; it does
NOT move the exact score. No score claim. Operator: *"almost positive there are bugs/mistakes/things missing."*
Assumed there are; hunted hard. READ-ONLY — proposes fixes, edits none (fixes land after synthesis).

**Surfaces audited:** `experiments/train_levelset_witness_realized_through_R_mlx.py` (argparse L2017-2393 + schedule
fns L625-702 + loop L1770-1810); `src/tac/witness_dsl/{curriculum_dsl,campaign,gauge}.py`; the three DSL test
files; `src/tac/boundary_math/{lane_sdf_component,lever_b_levelset_generator}.py`. Cross-checks the FEED-kk
arbitrariness audit (`witness_config_arbitrariness_audit_20260629T224737Z.md`) — VERIFIED each flagged item +
found MORE + CORRECTED two over-flags.

---

## 0. TOP-LINE (read first)

The most severe issues are **measurement-validity** bugs in the A/B campaign engine, not crash bugs — the
trainer runs, but several warm-start arms would measure the WRONG d_seg delta (confounded), and one lever is
**misnamed so it tests a different mechanism than its name claims**. These corrupt the campaign's verdicts,
which is the whole point of the campaign.

| # | sev | one-line | file:line | conf |
|---|---|---|---|---|
| C1 | **HIGH** | DSL never threads `--anneal-epochs` → non-Muon warm-start arms RE-HEAT temp/LR over the extended `epochs` denominator (confounds the A/B). Schedule fns also don't clamp past `_ae`. | curriculum_dsl `flag_dict` L228-247 + trainer L672/L1807 (no clamp) | HIGH |
| C2 | **HIGH** | `BASELINE.resume_from=_CE_CKPT` (ep299) but `with_lever` adds to `base.epochs=1500` → bare `BASELINE.with_lever(X)` runs a ~1301-epoch FULL RE-RUN from CE, not the intended ~100ep warm-start from L7@1500. | curriculum_dsl L354 + L189 | HIGH |
| M1 | MED | `BASELINE` sets `--w-pose 1.0`, contradicting the trainer default `0.0` + the frontier doctrine ("witness's only job is d_seg") + the FEED-kk audit's own "default already w-pose=0 ✓". Every non-A5 arm runs pose competing for d_seg capacity. | curriculum_dsl L366 vs trainer L2074 | HIGH |
| M2 | MED | `DirectionalBasis()` lever is MISNAMED: it toggles `--lane-edge-weight` (class-1 lane HINGE), NOT the directional Fourier basis (`--self-orient`, already ON in BASELINE). The measured −48% basis is therefore NOT an A/B arm; the docstring conflates two mechanisms. | curriculum_dsl L401-410 | HIGH |
| M3 | MED | `--max-bank-freq` caps ONLY the curvelet bank (sub-Nyquist by default → no-op); the real over-Nyquist source (self-orient dir-feats, up to 1024 cyc/unit) is NOT capped by it. FEED-kk fix #1 ("cap at 64") is INEFFECTIVE. | trainer L791 vs L806-810; lever_b_levelset_generator L126-129 | HIGH |
| M4 | MED | `--l7-start-epoch` trainer default **800** ≠ DSL/completed-run **900** → bare-CLI `--curriculum` launch diverges from the baseline; non-reproducible. | trainer L2146 vs curriculum_dsl L345 | HIGH |
| L1 | LOW-MED | `--mod-dim 32` stale comment "RD-optimum ~122KB" contradicts FEED-fl/fq SVD-floor 21; `--hidden-dim 96` vs waterfilled ~120. DSL BASELINE keeps 96/32 → run off the solve (magnitude UNVERIFIED here). | trainer L2056-2060 + curriculum_dsl L357 | MED |
| L2 | LOW-MED | EON `_V_HORIZON=174` vs in-code comment "188 IPM-optimal"; latent now (lane-prior/structured-init default-OFF), but a 14px horizon error scales the WHOLE ground-plane depth map when those levers engage. | lane_sdf_component L70 | HIGH |
| L3 | LOW | SDF decision-band half-width is uncontrolled (no `--margin-band-width`); emergent from temp-end × eikonal × render — the binding Lane R-survival knob is not first-class. (confirms FEED-kk #5/#7) | (design gap) | MED |
| L4 | LOW | seed inconsistency (`_LUMA_FOURIER_SEED 7` vs `0` elsewhere) — determinism hygiene. (confirms FEED-kk #12) | lever_b_generator | MED |

**CORRECTIONS to FEED-kk (over-flags / wrong fix):** see §3.
**VERIFIED-SOUND (checked, not bugs):** see §4.

---

## 1. CRITICAL / HIGH findings (detail + fix)

### C1 — `--anneal-epochs` never threaded → warm-start arms re-heat temp/LR (measurement-validity)  [HIGH]
**Mechanism.** The cosine schedules use denominator `_ae = anneal_epochs or args.epochs`:
`_softmax_temp_for_epoch` (trainer L672-674), `_hosc_beta_for_epoch` (L651-655), LR cosine (L1807-1808). The
DSL `flag_dict()` (curriculum_dsl L228-247) emits `--epochs` but **never emits `--anneal-epochs`**; no lever
sets it; `expand_cycles` (campaign L46-72) doesn't set it. `with_lever` extends `epochs` (L189), so a warm-start
arm runs the schedule over the EXTENDED denominator.
**Concrete confound.** `expand_cycles` cycle "l7_refine" (campaign test L49-54: ep1500→1600, no Muon, no
tau-freeze) → `_ae=1600`. At ep1500: `prog=1499/1599=0.9375`, temp `=0.05+0.475·(1+cos(0.9375π))≈0.059` — i.e. the
"refine" window **re-heats** temp from the 0.05 the L7 run ended at and re-warms LR, the OPPOSITE of a refinement.
This silently confounds the d_seg Δ the campaign is built to measure. The trainer's OWN help (L668-670) +
loop comment (L1804-1806) EXPLICITLY mandate setting `--anneal-epochs` to the original schedule length for
warm-starts — the DSL ignores its own mandate.
**Second half:** the schedule fns DON'T clamp `prog` at 1.0, so for ANY `ep > _ae` the cosine re-heats above the
end value (even with the fix). The `--anneal-epochs < --epochs` WARN (L2406) claims "the tail runs at the
**clamped** end values" — but the code does NOT clamp; it re-heats. Doc-vs-code mismatch.
**Muon arms are immune** (they freeze temp via start=end=0.05 and the LR cosine is gated `not muon_switched`,
L1799). So this bites every NON-Muon warm-start arm (PoseDecouple, DirectionalBasis, SoftBoundary, FiLMFix,
LanePrior, StiefelW, CodeSpectralEntropy, l7-refine cycles).
**Fix.** (a) Thread `--anneal-epochs = <original schedule length, e.g. 1500>` into every warm-start program —
simplest: add it to `BASELINE.base` and have `with_lever`/`expand_cycles` preserve it; (b) clamp
`prog = min(prog, 1.0)` in `_softmax_temp_for_epoch` / `_hosc_beta_for_epoch` / the LR cosine so a continuation
past `_ae` HOLDS the end value (matching the WARN's "clamped" claim) instead of re-heating; (c) add a DSL
`validate()` clause: if `resume_from` is set and `epochs > anneal_epochs` and the program is not temp-frozen,
require `--anneal-epochs`.

### C2 — `BASELINE.resume_from=_CE_CKPT(ep299)` + `with_lever` epochs math → bare composition is a full re-run, not a warm-start  [HIGH]
**Mechanism.** `BASELINE.resume_from = _CE_CKPT` = `levelset_resume_stageCE_ep299.npz` (curriculum_dsl L331-332,
L354) and `BASELINE.epochs = 1500`. `with_lever` computes `new_epochs = self.epochs + epochs_delta` (L189) and
`resume_from` defaults to `_INHERIT` (keeps CE@299). So `BASELINE.with_lever(Muon(1500,100))` → epochs **1600**,
resume **CE@299** → `range(299,1600)` = **1301** gradient epochs that RE-RUN the whole CE→tau→l7 curriculum from
ep299 and only THEN apply the finisher — NOT the ~100-epoch warm-start window the lever docstrings describe
("warm-start window ... when resumed at end-of-run", e.g. TauFrozen L416-418, Muon L387). The `with_lever`
epochs arithmetic (`base.epochs + delta`) is self-consistent ONLY when `base.epochs == resume_epoch`, i.e. when
resuming from L7@1500 (where `_L7_CKPT` = `…stageL7_ep1500.npz`, L333-334). It is INCONSISTENT with the default
CE@299.
**Why it matters.** The DM1Minimal docstring example `BASELINE.with_lever(*DM1Minimal())` (L512) and the
curriculum_dsl tests (L95/L101/L150…) use the bare path → they'd launch 1301-epoch re-runs (≈13× the intended
compute) AND, combined with C1, with a stretched anneal. The **campaign** path is safe ONLY because the campaign
tests pass `resume_from=_L7` explicitly (test_witness_campaign L130-132; expand_cycles passes it L66). So the
footgun is in the DEFAULT + the documented bare usage, not the explicit campaign launch.
**Fix.** Set `BASELINE.resume_from = _L7_CKPT` (ep1500) so `base.epochs(1500)==resume_epoch(1500)` makes
`with_lever` epochs math correct by construction; OR change `with_lever` to compute `epochs = resume_epoch +
epochs_delta` by reading the ckpt epoch; OR add a `validate()` clause that flags `epochs - resume_epoch` ≫
`sum(epochs_delta)` (run length far exceeds the declared window). Pick one; document the CE-warm-start vs
L7-warm-start intent explicitly. CONF HIGH that the arithmetic is inconsistent with the default; MED on which
fix the operator wants (depends whether arms should warm-start from CE@299 with short windows + `--anneal-epochs
1500`, or from L7@1500).

---

## 2. MEDIUM findings (detail + fix)

### M1 — `BASELINE --w-pose 1.0` contradicts the documented-optimal `0.0`  [conf HIGH it contradicts]
curriculum_dsl L366 sets `--w-pose: 1.0`. Trainer default L2074 is `0.0`, with the comment (L2071-2073): *"DROP
pose-from-texture (the COLLAPSED amortized carrier) … the witness's ONLY binding job is d_seg. w_pose=0 by
default."* The FEED-kk audit §2 synergy #7 also asserts "Default already w-pose=0 ✓ (the synergy is realized)" —
**false for the DSL BASELINE.** Consequence: A0 + every non-A5 arm trains with pose loss competing for the
shared-decoder capacity (the +0.70 seg-pose coupling FEED-gi), masking d_seg gains; only A5 PoseDecouple sets it
back to 0. The doctrine wants w-pose=0 as the FLOOR and "add pose" as the (optional) lever — the BASELINE has it
backwards.
**Caveat (calibration):** BASELINE claims to "reproduce the launched config" (L329); if the completed run truly
used w-pose=1.0 then BASELINE is faithful and the completed run itself was off-doctrine. Either way: reconcile —
set `BASELINE --w-pose 0.0` (and make pose-loss the opt-in lever), OR document why the completed run used 1.0.

### M2 — `DirectionalBasis()` lever is misnamed; tests the lane-edge hinge, not the directional basis  [conf HIGH]
curriculum_dsl L401-410: `DirectionalBasis()` overrides `{"--lane-edge-weight": …, "--lane-edge-start-epoch": …}`
— i.e. the **class-1 lane-edge margin HINGE** (a loss term, ~19% of flips). The actual measured −48% lever is the
**oriented Fourier BASIS** = `--self-orient` + curvelet orientation (FEED-kk 1E), which is ALREADY ON in BASELINE
(`--self-orient: True`, L360). The lever docstring (L403-405) conflates them: *"Turn the lane-edge directional
term ON … The all-class directional/tangent basis measured −48%."* Two distinct mechanisms, one name. Result:
(a) an operator running the "DirectionalBasis" arm believes they're A/B-ing the −48% basis but are A/B-ing the
weaker class-1 hinge → mis-attributed verdict; (b) the −48% basis has NO A/B arm at all (no lever toggles
`--self-orient`; it's baked into every arm). **Fix.** Rename the lever to `LaneEdgeHinge` (it IS LEVER-3); fix
the docstring; if a real basis A/B is wanted, add a separate lever toggling `--self-orient` (and confirm the
ordering-synergy: capacity-routing only on top of self-orient ON — which BASELINE satisfies).

### M3 — `--max-bank-freq` doesn't reach the over-Nyquist source; FEED-kk fix #1 is ineffective  [conf HIGH]
`--max-bank-freq` is consumed ONLY at trainer L791 `curvelet_directional_B(bank, max_freq=args.max_bank_freq)` —
it caps the curvelet bank `B`. The bank's own docstring (lever_b_levelset_generator L120-129) states the DEFAULT
bank max is 16 cyc/unit (4× BELOW Nyquist=64) so capping it is a **no-op**, and that the real over-Nyquist waste
is the **self-orient directional feats** (`freq_across=32, n_dir_freqs=6 → 32·2^5 = 1024 cyc/unit, 16× over`),
built separately at L806-810 / L1236-1238 from `--freq-across`/`--n-dir-freqs` and NOT touched by `max_freq`. So
FEED-kk §3 fix #1 ("Cap `--max-bank-freq` at 64 … one-line set") FIXES NOTHING for the actual aliasing source.
The real cap is `--n-dir-freqs ≤ 2` (at freq_across=32) or `--freq-across 8, --n-dir-freqs 4`. **Note:** the DSL
BASELINE is already Nyquist-safe (`--n-dir-freqs 2 --freq-across 32 --self-orient True`, L360-361), so the
PLANNED run is fine — the bug is latent: any arm that bumps `--n-dir-freqs` "for finer angular coverage" while
trusting `--max-bank-freq 64` to cap it will alias under R (d_seg killer). **Fix.** Either (a) extend the cap to
the self-orient feats (apply the Nyquist drop in `self_orientation_directional_feats`), or (b) add a `validate()`/
preflight that refuses `freq_across·2^(n_dir_freqs-1) > stem_nyquist`, and correct the FEED-kk fix-#1 guidance.

### M4 — `--l7-start-epoch` default 800 ≠ DSL/baseline 900  [conf HIGH]
Trainer default L2146 = **800**; DSL BASELINE Stage L345 = **900** (and the preserved completed-run ckpt is
`…stageL7_ep1500` with tau→l7 @ 900). Both satisfy the curriculum guard (L2414 `0<tau<l7≤epochs`), so neither
crashes — but a bare-CLI `--curriculum` launch WITHOUT `--l7-start-epoch` runs l7@800 (tau gets 500ep) while the
DSL/baseline runs l7@900 (tau gets 600ep) → different curriculum, non-reproducible. **Fix.** Set the trainer
default to 900 to match the canonical completed run + DSL (FEED-gn's "tau dead after ~675" argues for ≤900 but
that's a separate sweep; first make the default CONSISTENT). CONF HIGH on the inconsistency; the "right" value
(800 vs 900 vs FEED-gn early-l7) is a SWEPT question, not this fix.

---

## 3. CORRECTIONS to the FEED-kk audit (verified over-flags / wrong fix)

1. **`_CY` "missing/ARBITRARY" (FEED-kk 1I, item #2 in my brief) → NOT a bug.** lane_sdf_component L66-70 uses a
   small-angle **flat-ground IPM** parameterized by `_V_HORIZON` (the vanishing-point row), `forward = H·fy/(v −
   v_h)`, `lateral = −(u−cx)·forward/fx`. This formulation has NO principal-point-y term — `cy` is structurally
   absorbed into `v_horizon`. `_CX` IS present and used (lateral). So "cy unmodeled = ARBITRARY" is a false alarm;
   the modeling choice is sound. (The `_V_HORIZON 174 vs 188` discrepancy IS real — see L2.)
2. **capacity-routing + Muon "OFF by default" (FEED-kk top-line #3) → NOT a bug.** `--margin-saliency-weight 0.0`
   / `--hardness-oversample 0.0` / `--muon-start-epoch None` are CORRECTLY default-OFF: they make the A0 baseline
   arm byte-identical, and the campaign turns them on as A/B arms (LEVER-4/5, Muon). "The default config doesn't
   run the levers" is true but it's the A/B design, not a misconfiguration. The real risk is upstream (C1/C2
   confounding the arms that DO turn them on), not the default-OFF.
3. **FEED-kk §3 fix #1 ("cap `--max-bank-freq` at 64", "one-line set") → INEFFECTIVE** (see M3). Correct the
   guidance: the over-Nyquist source is `--n-dir-freqs`/`--freq-across`, not the bank.
4. **FEED-kk §2 synergy #7 "Default already w-pose=0 ✓" → FALSE for the DSL BASELINE** (w-pose=1.0; see M1). True
   only for the trainer default.

---

## 4. VERIFIED-SOUND (checked; not bugs — recorded so Round 2 needn't re-walk)

- **DSL never-invent-flags guard** (`real_trainer_flags` + `validate` L250-257) is real + tested — emitted flags
  are checked against the trainer's actual argparse. No invented flags found in BASELINE/levers.
- **C1 dead-arm guard** (curriculum_dsl L266-282) keys `("epoch","__epoch","__resume_epoch")` MATCH the ckpt save
  keys (`__epoch` L194, `__resume_epoch` L223) — the guard is NOT a silent no-op. (But it only fires when the
  resume ckpt exists at plan time; chained `expand_cycles` cycles whose resume ckpt doesn't exist yet bypass it —
  benign because the window math keeps `epochs > resume_epoch`.)
- **store_true C2 guard** (`real_store_true_flags` + L259-264; tested L173-176) — a `False` on a store_true flag
  is refused before launch (would compile to a crashing `--no-X`).
- **curriculum boundary guard** (trainer L2413-2420) fail-closed on `0<tau<l7≤epochs`.
- **gauge NO-FAKE invariant** (`GaugeCost.__post_init__` L169-184) — a PENDING cell may not carry fabricated
  numbers; `GaugeChoice.validate` rejects non-compliant/non-deterministic charts by construction.
- **Muon temp/LR freeze** (L1771-1799) is deterministic in `muon_start_epoch` (resume-into-finisher reproduces),
  and immune to C1.

---

## 5. RANKED FIX LIST (highest measurement-validity leverage first)

1. **C1** — thread `--anneal-epochs` (original schedule len) into all warm-start programs + clamp `prog≤1` in the
   3 schedule fns. (Unblocks trustworthy non-Muon A/B deltas — the campaign's purpose.)
2. **C2** — make `BASELINE.resume_from` / `with_lever` epochs math self-consistent (set resume=`_L7_CKPT` OR
   compute epochs from the ckpt epoch) + a `validate()` window-sanity clause.
3. **M2** — rename `DirectionalBasis`→`LaneEdgeHinge`, fix the conflated docstring; (optional) add a real
   `--self-orient` A/B lever.
4. **M1** — reconcile `BASELINE --w-pose` to 0.0 (or document the completed-run 1.0).
5. **M4** — set trainer `--l7-start-epoch` default = 900 (match DSL/completed run).
6. **M3** — extend the Nyquist cap to the self-orient dir-feats (or add a preflight) + correct FEED-kk fix #1.
7. **L1** — reconcile `--mod-dim`/`--hidden-dim` defaults + the stale "122KB" comment to the FEED-fq waterfill
   (verify the 21/120 magnitudes on the actual artifact first — UNVERIFIED here).
8. **L2/L3/L4** — `_V_HORIZON`→188 (or sweep) when warp levers engage; first-class `--margin-band-width`; unify
   seeds.

---

## 6. DAG-FEED SUMMARY

**ADVERSARIAL REVIEW R1 / LENS B (config+correctness) — DONE ($0 CPU read-only; pointer 0.19110 UNMOVED, MEANS≠ends.)**
Audited the v2 witness trainer (116 args) + the witness_dsl (curriculum/campaign/gauge) + boundary_math EON/bank
constants + the 3 DSL test files. **2 HIGH measurement-validity bugs in the A/B campaign engine:** **(C1)** the DSL
never threads `--anneal-epochs`, so every NON-Muon warm-start arm re-heats temp/LR over the extended `epochs`
denominator (and the schedule fns don't clamp past `_ae` despite the WARN claiming "clamped") → confounds the d_seg
Δ the campaign exists to measure; trainer L668-670/L1804-1806 mandate setting it and the DSL ignores its own
mandate. **(C2)** `BASELINE.resume_from=_CE_CKPT(ep299)` + `with_lever` adding to `base.epochs=1500` → a bare
`BASELINE.with_lever(X)` (the DM1Minimal docstring + curriculum_dsl tests) runs a ~1301-epoch FULL re-run from CE,
not the intended ~100ep warm-start from L7@1500 (the campaign path is safe only because it passes resume=`_L7`
explicitly). **4 MED:** **(M1)** BASELINE `--w-pose 1.0` contradicts trainer default 0.0 + frontier "witness's only
job is d_seg" + the FEED-kk audit's own claim — every non-A5 arm trains pose competing for d_seg capacity;
**(M2)** the `DirectionalBasis` lever is MISNAMED — it toggles `--lane-edge-weight` (class-1 hinge), NOT the
directional Fourier basis (`--self-orient`, already ON), and its docstring conflates the two → the measured −48%
basis is not an A/B arm; **(M3)** `--max-bank-freq` caps only the (sub-Nyquist) curvelet bank, NOT the self-orient
dir-feats (up to 1024 cyc/unit) where the over-Nyquist aliasing lives → FEED-kk fix#1 is INEFFECTIVE (real cap =
`--n-dir-freqs`/`--freq-across`; BASELINE already safe via n-dir-freqs=2); **(M4)** `--l7-start-epoch` default
800≠DSL/completed-run 900 (bare-CLI launch non-reproducible). **CORRECTED 2 FEED-kk over-flags:** `_CY` is NOT a
bug (small-angle flat-ground IPM is v_horizon-parameterized, cy absorbed; cx present); capacity-routing/Muon
default-OFF is the intended A/B baseline, not a misconfig. **VERIFIED-SOUND:** never-invent-flags guard, C1
dead-arm guard keys match ckpt, store_true guard, curriculum boundary guard, gauge NO-FAKE invariant. Memo
`.omx/research/adversarial_review_round1_config_20260630T004333Z.md`. Cross-refs: FEED-kk
(witness_config_arbitrariness_audit_20260629T224737Z). Round-2 should re-test C1/C2 after fix + a 3rd-pass on the
reorient/self-orient feat path. pointer 0.19110.
