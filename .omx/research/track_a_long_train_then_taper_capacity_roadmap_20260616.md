# Track-A roadmap: long small-learned train (optimal config) → THEN taper → THEN capacity

**Operator 2026-06-16:** "building to a long track A training with the small learned, after
which we can then explore taper and capacity." This note pins the optimal config + the phased
sequence so the next unit inherits it. All `[contest-CPU advisory]` until byte-closed exact eval.

## The optimal small-learned config (synthesis of the whole session)
The base_ch=20 small learned basis (HNeRV decoder, 83,356 params), trained with:
- **Loss = SHARP soft_cosine** (the disambiguator + iso verdict): `T 0.3→0.05` anneal (hold_frac
  0.5) + `margin_weight_renorm=True` + tight margin `τ=0.5`. **seg_weight = 1.0** — the iso run
  proved the win is SHARPNESS, not the seg_weight crank (sw1.0 d_seg 0.002790 ≈ sw1.5 0.002786,
  and 1.5× only ADDED pose drift). DROP the 1.5× crank.
- **Pose = FiLM-v2 DECOUPLING** (operator pose-decoupling point): rgb_0-only residual FiLM, rgb_1
  (d_seg) FiLM-CLEAN → `∂d_seg/∂pose = 0` exactly. Pose carried by 6 stored scalars (~1KB,
  Wyner-Ziv), so the sharp seg crank has NO pose cost. + **pose-grad throttle** (k=4, resume_thr
  0.001) once d_pose is at floor. FiLM-warm-start (SEALED) loads the converged basin into the
  wrapper's inner decoder (identity FiLM).
- **EMA warmup** on (tracks early, converges to faithful 0.999 by ~750ep).
- **Budget = LONG** (ride the power-law: d_seg ≈ 0.0367·ep^-0.35, still descending at 800ep —
  the 0.00359 'floor' was UNDER-TRAINING, not capacity; see
  [[dseg-floor-is-loss-movable-not-capacity-bound-oomph-wins]]).

## The one real decision: 96-pair vs 600-pair for the long run
- **96-pair** (warm-start control/best, fast ~19s/ep): a 3000ep run ~16h. RISK: long oomph on 96
  pairs can OVERFIT the 96-pair latents → the advisory d_seg understates vs 600-pair. Cheap
  research milestone, but NOT a real candidate without 600-pair validation.
- **600-pair** (FROM-0 — control latents are (96,28), can't warm-start 600): the REAL operating
  point, no overfit risk, the path to a byte-closed exact-eval candidate. ~6× slower/ep.
- **Recommendation:** validate the config on the running 96-pair decoupled A/B FIRST (confirm
  pose held at floor + d_seg drops vs coupled). THEN do the long run at 600-pair from-0 (the real
  milestone) — OR a 96-pair long run first if a fast read is wanted, but gate any exact claim on
  600-pair. No new code needed: the launcher already supports --pose-film-v2 / --oomph-seg-weight-mult
  1.0 / --epochs N / (600-pair from-0 uses launch_from0 + an oomph overlay — small add).

## Phased sequence (gated)
1. **NOW (running):** coupled vs decoupled 600ep@96 A/B → confirms FiLM removes the residual pose
   drift (the decisive proof of the decoupling). ETA ~5.6h.
2. **Long Track-A train (gated on #1 confirming):** small basis + the optimal config above, long
   budget, resumable. The small-learned floor under the optimal config.
3. **Taper (#121, orthogonal, sealed):** d_seg-aware capacity REALLOCATION (early→mid-late, gate-2
   sensitivity band), byte-matched. A/B vs the long-train baseline.
4. **Capacity:** scale base_ch (20→36→…) and pair count toward 600; the capacity-vs-rate RD trade.
5. **Frontier:** byte-close → dual CPU/CUDA exact eval → pointer (the END; current frontier 0.191,
   unmoved — all of the above is means until an exact row crosses).

## Sealed building blocks (this session)
- warm_start_dir + ema_warmup driver hooks (11-round SEAL).
- configurable-taper carrier (parity-gated) + launch_taper_ab (#121).
- oomph disambiguator + FiLM-warm-start decoupling (3-pass SEAL).
HEAD at writing: b7fcd5de9.
