# From-0 deployment — full-stack adversarial review (2026-06-13)

**Operator:** *"Adversarial reviews first"* + *"Should probably also review the full stack
deployment."* This is the DESIGN-level adversarial review gating the #116 from-0 decisive A/B
launch (levers 2+3+5 v2 distortion arm vs vendored levers-OFF control). It covers the whole
deployment DAG, not just the v2 wire-in. Findings: 🔴 blocker, 🟡 must-resolve, 🟢 clean.
Authority for every in-loop number it produces: `[contest-CPU advisory]` NON-PROMOTABLE until a
byte-closed archive runs through `upstream/evaluate.py`.

## The deployment DAG (what actually runs)
```
base_ch20 vendored HNeRV  ─┐
levers into curriculum:    │   shared epoch-0 init (same seed; RNG-neutral FiLM build)
  L2 soft_cosine @ fast-   │        ┌── arm A (distortion): L2+L3v2+L5 on
     cool 1.0→0.3 hold     ├──►A/B ─┤
  L3 v2 pose-FiLM (rgb_0)  │        └── arm B (control):   levers OFF (vendored CE, no FiLM)
  L5 margin-weight τ=0.3   │
                           │   train (split-by-head: SegNet grad MPS / PoseNet grad CPU,
                           │          OR CPU-only) → EMA shadow → per-epoch exact CPU d_seg/d_pose
                           └─► byte-close (vendored 3-section + additive pose section)
                                 → inflate_film_decoder_v2 round-trip → exact d_seg/d_pose/rate
                                 → verdict: does the lever stack beat control on the SCORE?
```

## A. v2 wire-in (driver) — 🔴→resolved
- **🔴 FiLM params route to Muon, bypassing the LR cap.** `partition_params_for_muon` sends
  "2-D+ weights not in stem/not in rgb" to Muon. v2's `pose_mlp.fc1/fc2` (Linear),
  `film_resid.proj` (Conv2d), `film_resid.gamma_head/beta_head` (Linear) ALL match → in a Muon
  stage they'd train under Muon at `muon_lr`, and in AdamW stages they sit in the decoder group
  at full `adamw_lr` — both violate the #118-SEALED requirement (FiLM LR ≤ 1e-3 to kill the
  transient overshoot). A naive prefix-swap wire-in would re-deploy the v1 instability v2 exists
  to fix. **FIX:** exclude FiLM params (by name) from the Muon partition AND the decoder AdamW
  group; give them a dedicated AdamW group at `min(adamw_lr, _FILM_LR_CAP=1e-3)`. Added ONLY when
  `pose_film_enabled` → basin byte-identity preserved.
- **🟢 reused seams are version-agnostic:** `_FiLMEvalDecoder.forward` calls `wrapper(z, idx)`
  generically; `wrapper_sd_to_archive_decoder_sd` keeps any non-`decoder.` key verbatim (so
  `pose_mlp.*`/`film_resid.*` ship in the blob). Both work on the v2 wrapper unchanged.
- **Plan (5 version-aware touchpoints):** (1) `pose_film_version:int=1` ∈{1,2} + `_FILM_LR_CAP`;
  (2) `_new_decoder` v2 branch (keep RNG-neutral snapshot/restore); (3) `_build_stage_runtime`
  FiLM-group separation (THE fix); (4) build-archive/inflate version-aware import+prefix+rebuild
  +`inflate_film_decoder_v2`; (5) default-OFF byte-identity (test).

## B. Do the distortion levers FIGHT? (operator's concern) — 🟢 with v2
- **L3↔L2 (pose vs seg): the ONE pathological fight v2 REMOVES.** v1 injected FiLM on the shared
  6×8 stem → fed BOTH heads → pose perturbed `f1` (the SegNet frame) → d_pose coupled into d_seg.
  v2 puts the residual FiLM on `rgb_0` ONLY; `f1` renders from the clean trunk → **physically
  invariant to pose** (proved by the pose-invariance test). The fight is gone by construction.
- **L2↔L5 (seg surrogate × margin-weight): same objective, synergistic.** L5 multiplies L2's
  per-pixel gradient by `exp(−margin/τ)`; with τ slaved to T at the resonance it amplifies exactly
  the small-margin flips L2 can move. Not a fight — a reweighting (anneal memo §7).
- **L3↔L5: orthogonal.** L5 touches only the seg loss; L3 only rgb_0/pose.
- **Residual coupling (honest):** L2(+5) and L3 still share the decoder TRUNK `x` (rgb_1 and rgb_0
  both read it). That is ordinary multi-task coupling, NOT the v1 pathology — the FiLM carries the
  pose-specific correction off-trunk, but the trunk serves both heads and the optimizer balances.
  **The from-0 A/B is precisely the measurement of the NET** (does the combined arm beat control
  on BOTH d_seg and d_pose). Verdict: no pathological fight after v2; the shared-trunk coupling is
  the thing the decisive run exists to quantify.

## C. The from-0 A/B launcher — 🔴 DOES NOT EXIST in the right shape (the real scope)
- `launch_pose_film_basin.py` is **forkpoint-RESUME**, **v1-only**, and configures **no L2/L5**
  (it relies on the default curriculum's CE seg loss). `launch_split_by_head_basin.py` is the
  from-0 **single-arm control** (levers off). Neither is the #116 from-0 **A/B with the v2 lever
  stack**. **This launcher must be BUILT.** Cleanest shape: ONE launcher `--arm {distortion,
  control}` run twice with the SAME `seed` (shared epoch-0 init via the RNG-neutral FiLM build):
  - arm=distortion → `pose_film_enabled=True, pose_film_version=2` + curriculum StageSpecs set to
    `seg_surrogate="soft_cosine", seg_temperature=1.0, seg_temperature_end=0.3,
    seg_temperature_hold_frac=0.3, margin_weight_tau=0.3` (the just-confirmed fast-cool anneal +
    L5).
  - arm=control → vendored defaults (`pose_film_enabled=False`, CE seg, no margin-weight).
- **🟡 lever→curriculum injection:** the levers live on `StageSpec` fields; the launcher must
  build/patch the curriculum's specs with the lever config (the driver's `build_curriculum`
  yields vendored specs; the launcher overlays the lever fields per stage). Confirm the overlay
  path (or add a `cfg`-level lever block the driver applies to every spec).

## D. Byte-close → inflate → exact eval — 🟡 needs a v2 round-trip test
- The additive pose-section grammar + `inflate_film_decoder_v2` are built; the split keys
  (`pose_mlp.`/`film_resid.`) match the blob. **🟡 add a byte-close round-trip parity test on a v2
  archive** (train tiny → build archive → `inflate_film_decoder_v2` → assert rendered pairs match
  the in-memory wrapper render, and the additive pose section parses to `(n_pairs,6)`), mirroring
  the v1 round-trip test. The exact d_seg/d_pose that pick BEST run on the **CPU authority** (no
  MPS) — unchanged.

## E. Deployment mechanics — 🔴 daemon, 🟡 split-device
- **🔴 launch as a detached `nohup` daemon, NOT `run_in_background`** (the SIGURG-144 at ~3 min
  killed the anneal run twice today). Reparent to init, `< /dev/null`, DONE-marker on exit; the
  session is a reader. Durable per-epoch checkpoints already give resume.
- **🟡 split-by-head + v2 FiLM device placement:** the basin trains split (SegNet grad MPS /
  PoseNet grad CPU). v2's FiLM is on the rgb_0/pose path → under split-by-head the FiLM forward
  must run on the correct device for the pose cotangent. **Verify the v2 wrapper composes with
  `split_by_head`** before a split launch; if unverified, **default the from-0 run to CPU-only**
  (slower but fully trusted + no device-placement risk) and treat split-by-head as an opt-in
  speedup after a parity check.
- **🟢 authority:** `__post_init__` refuses MPS as the authority device; exact metric is CPU.

## Open decisions (defaults chosen; override if desired)
1. **Device:** default **CPU-only** for correctness (split-by-head opt-in after the v2 parity
   check). 2. **Epoch budget:** a bounded proportional budget for the first decisive read (not the
   full 29,650) — enough to see the d_seg/d_pose delta vs control, resumable to extend. 3.
   **Control seg-loss:** vendored CE (levers fully OFF). 4. **Seed:** shared across arms.

## Verdict + gated build sequence
Design is **clean after the §A Muon fix**; the lever stack does not pathologically fight (§B); the
real scope is the **from-0 A/B launcher (§C) + the v2 round-trip test (§D) + daemon mechanics
(§E)**, not a one-line launch. Build order, each gated:
1. v2 driver wire-in (§A) + byte-identity + EMA-split + round-trip tests (§D).
2. from-0 A/B launcher (§C) with the fast-cool anneal + L5 overlaid on the curriculum.
3. **Independent code-reviewer pass** (reviewer-vs-author) on the wire-in + launcher.
4. recursive review rounds → 3 clean → SEAL (#100).
5. detached-daemon launch (§E), CPU authority, resumable; harvest the A/B verdict.
