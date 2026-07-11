# GENERAL-COVARIANCE-OF-THE-WITNESS totality audit + texture-trunk verdict

**Agent:** Opus measurement subagent · **Date:** 2026-07-10 · **Cost:** $0, CPU-only, NO scorer
forward, MPS-never, cached `gt_n600.npz` only; live #205 run (pid 88030) untouched. **Pointer
0.19108282 UNMOVED** — this is a LAW test + an architecture verdict; it moves no score. All numbers
`[macOS-CPU research-signal]` `score_claim=false promotable=false`. Per
`docs/operating_manual_craft_handoff.md`: answer first, every number re-derived from the primary
cache, own-round-1 review at the bottom, verdicts scope-laddered and labeled MEASURED.

**Operator question (Einstein memo Table B1):** run the $0 audit of the GENERAL COVARIANCE OF THE
WITNESS law — *"all pair-dependence must factor through (ξ, measurement-operator); anything else is a
scene event or WASTED rate."* Regress the per-pair code on (ξ, phase); MEASURE the residual. Double
duty: (A) confirm/refute the covariance law's totality; (B/C) DECIDE the texture-trunk arm
(v7.5.3 single+texture vs v7.5.2 single).

---

## ANSWER FIRST (the three verdicts)

**verdict_scope: formulation** (narrowest level the measurement supports, per verdict-scope ladder) —
- **REFUTED** applies ONLY to the naive formulation *"(ξ,phase) explains ≈everything as a smooth map"*; the law AS-STATED (holonomy ⊕ gauge ⊕ events) is CONFIRMED.
- **DEAD** applies ONLY to the *texture-trunk-as-d_seg-reconstructor* formulation — this is a d_seg-only cached-argmax audit; the trunk's LIVES-ON-POSE (photometric d_pose) role is UNDECIDED, so this is NOT a family/paradigm kill of #395.
- **INERT** (below/#417) is the *v7.5.3 tex_trunk build-consumption* instance; it is an instance-level receiver-bijection finding, not a claim about the texture-trunk idea in general.
- Reactivation of the DROP: a d_pose-side mirror audit that shows the trunk carries photometric pose signal beyond R1's ξ sidecar.

- **A — COVARIANCE TOTALITY: two-level, scope-laddered.**
  - *As "(ξ,phase) explains ≈everything as a smooth map"*: **REFUTED.** Even a fully nonparametric
    kNN-in-(ξ,phase) explains only **0.42** of the per-pair code variance (CV, n600). The honest
    bracket for the exact covariance basis is **[0.42 (kNN lower) , 0.93 (rank-8 upper)]**.
  - *As the law ACTUALLY states it* (pair-dependence = holonomy-of-ξ ⊕ gauge-phase ⊕ genuine events;
    "anything else = wasted rate"): **CONFIRMED.** The UNEXPLAINED residual is **96.4%** in the law's
    *allowed* buckets (85.3% Movable = reaction events, 10.9% Lane + ~0 MyCar = gauge/boundary
    jitter) and the *forbidden* "wasted texture rate" bucket is empirically **≈0** (residual is
    SMOOTH: high-freq/low-freq gradient ratio **2e-4**; corr(residual-energy, movable-area-change)
    **+0.28**). REGISTERED `witness_general_covariance_totality_v1`.

- **B — TEXTURE-TRUNK VERDICT: DEAD for d_seg.** The part of the per-pair partition NOT explained by
  ego+phase is smooth reaction-events + aleatoric gauge jitter — NOT spatially high-frequency
  photometric texture. A texture trunk (per-pair luma/chroma high-freq reconstruction) has **no
  d_seg-relevant signal to carry**. This is the *capacity-necessity* side of #417's
  *build-consumption* finding (tex_trunk COUNTED-but-INERT): both point at DROP.
  **LIVES-ON-POSE = UNDECIDED** (this audit is d_seg-partition-only; a photometric d_pose-legibility
  role beyond R1's ξ is neither shown nor refuted here — but nothing here supports it, and R1 already
  banks d_pose via the ξ sidecar, so the burden of proof is on the trunk).

- **C — CAPSTONE FINAL FORM: single covariant trunk (v7.5.2), texture trunk DROPPED.** The residual's
  structure *prescribes* the v8 carriers: a Movable **reaction-event sidecar** (the 85% bucket) +
  the Lane/hood **gauge-phase zero-mode carrier** (T1 + B2). Supported: single trunk + T1/#360 phase
  forces + Law-5 curriculum + v8 per-class store carriers; texture trunk not part of the optimal form.

---

## THE MEASUREMENT

**Per-pair code** (reproduces the rank-8=95.6% anchor, `tools/measure_dm3_spatial_grid_vs_global_code.py`):
`φ_p` = ideal per-class scipy-EDT SDF of the frozen-SegNet argmax `lstar_p` (argmax-roundtrip 0 px,
asserted); `R_p = φ_p − mean_p φ` = the per-pair partition variation (the payload's job);
`code_p` = left singular vectors × √eval of `G = R Rᵀ` (cross-pair variance in dim d = eval_d). The
top-8 `code` (holding 93.2% of variance at n600 / 95.6% at n96) is the regression target.

**Covariance basis:** `ξ_p` = `gt_poses` (600,6) = the **PoseNet GT twist** (the contest-scored, and
therefore lossy, 6-dim ego readout); `phase_p` = circular-mean (cos,sin) of the GT sub-pixel tie
coordinate (`phase_primitives.gt_tie_targets_numpy`) + active-fraction (3-dim). Models: linear ξ(6),
quad-Taylor ξ(27, homography-nonlinearity proxy), +phase, and an assumption-free **kNN-in-(ξ,phase)**
(k=8). Headline = **2-fold CV Frobenius R²** (finite-P overfit defeated; random-Gaussian basis of
matched width = the null). Tool: `.omx/tmp/covariance_audit/audit.py` (rebuildable from the committed
cache).

### R² table — POOLED per-pair code (8-dim, MEASURED, CV)

| model | n96 CV R² (null) | **n600 CV R² (null)** |
|---|---|---|
| rank-8 cum-var (code ceiling) | 0.9559 | **0.9316** |
| ξ linear (6) | 0.210 (−0.22) | **0.223** (−0.04) |
| ξ linear + phase | 0.206 | **0.262** (−0.06) |
| ξ quad (27) | −0.76 *(overfit@n96)* | **0.308** (−0.11) |
| ξ quad + phase | −0.69 *(overfit@n96)* | **0.337** (−0.10) |
| phase only (3) | 0.044 | **0.070** (−0.01) |
| **kNN (ξ+phase), nonparametric** | — | **0.4201** |
| kNN (ξ only) | — | 0.3924 |

*n96 quad is overfit (27 features, P=96) — exactly why the audit overrode subset defaults to n600,
where the null shrinks to ~−0.05.* The kNN row is the assumption-free totality number: **any** smooth
function of measured (ξ,phase) captures ≤0.42 of the code. gt_poses is a lossy 6-dim readout and kNN
in 9-dim @ P=600 is curse-of-dim-starved ⇒ 0.42 is a **LOWER bound**; rank-8 0.93 is the linear
subspace **UPPER bound**. The exact covariance basis (true homography H(ξ)) lies in **[0.42, 0.93]**.

### R² table — PER-CLASS code (n600, MEASURED, CV) + the covariance TAXONOMY

| class | rank-8 cum-var | quad+phase R² | **kNN(ξ,phase) R²** | taxonomy reading |
|---|---|---|---|---|
| Road (0) | 0.796 | 0.351 | **0.454** | ego homography orbit — MOST ξ-explained ✓ |
| Undrivable/sky (2) | 0.933 | 0.366 | **0.481** | rotation-only (parallax→0) — MOST ξ-explained ✓ |
| Movable (3) | 0.993 | 0.350 | **0.441** | independent motion — partly ξ, residual = **events** |
| Lane (1) | 0.886 | 0.093 | **0.087** | trajectory rides ξ but code = **survival/annulus jitter** (gauge), NOT motion |
| MyCar/hood (4) | 0.966 | 0.024 | **0.164** | static (IoU .994) — tiny variance = boundary jitter (gauge) |

The taxonomy the covariance law predicts is MEASURED: the **ego classes** (Road/Undrivable) are the
most ξ-explained; **Lane** is NOT ego-explained because its code is dominated by thin-dashed argmax
*survival jitter* (the FEED-it "Lane splits 3 ways": trajectory=ego + survival=gauge + …), i.e. gauge
phase, not motion; **MyCar** is static (gauge only, negligible energy).

### RESIDUAL CHARACTERIZATION (the decisive object; after ξ_quad+phase, mapped through the SVD dict)

| metric | MEASURED n600 | reading |
|---|---|---|
| residual energy: **Movable** | **0.8527** | independent-motion **reaction events** (law-allowed) |
| residual energy: **Lane** | 0.1093 | thin-lane survival = **gauge phase** (law-allowed) |
| residual energy: Undrivable / Road / MyCar | 0.028 / 0.010 / 0.0001 | negligible |
| **high-freq / low-freq gradient ratio** | **0.0002** | residual is **SMOOTH — NOT texture** |
| **corr(residual-energy, |Δ movable area|)** | **+0.2757** | residual tracks **reaction events** |

**Size:** ~58–66% of the code is unexplained by the *measured-lossy* (ξ,phase) proxy (kNN 0.42 lower
bound); but structurally that residual is **96.4% law-allowed buckets** and **0% wasted texture**.
**Shape:** smooth (hf/lf 2e-4), Movable+Lane-concentrated, movable-event-correlated =
**reaction-events ⊕ gauge-jitter, geometric/aleatoric — NOT photometric texture.**

---

## WHY B (texture trunk DEAD for d_seg) FALLS OUT

A texture trunk reconstructs spatially high-frequency per-pair luma/chroma detail. The per-pair
partition residual has **none** of that (hf/lf 2e-4). Its two buckets are:
1. **Movable (85%)** = objects moving independently of ego → correctly **not-ξ** → the law's *event*
   bucket → wants a cheap quantized **event sidecar**, not a dense texture trunk (a texture trunk
   would be an expensive, ill-matched way to carry a smooth object-mask reaction).
2. **Lane/hood (11%)** = sub-cell **gauge phase** (the AA-irremovable comb jitter, the stored
   zero-mode) → **unreconstructable** by any trunk; it is the T1/B2 phase-carrier's job.

Neither bucket is texture ⇒ a texture trunk has **no d_seg job**. Cross-check with **#417**
(`receiver_forward_parity_v753_v8_v1`): #417 found v7.5.3's tex_trunk COUNTED-but-INERT at the
*build/receiver* layer; this audit shows there is *no d_seg signal for it to carry* at the
*capacity-necessity* layer. Two independent kills → DROP.

---

## TRIALITY

- **equations:** REGISTERED `witness_general_covariance_totality_v1` in
  `tac.canonical_equations.einstein_pass_covariance_laws_20260710` (covariance_class =
  {law: COVARIANT_LAW, explained_fraction_bracket + residual_energy_shares: GAUGE_MEASUREMENT};
  callables `covariance_explained_fraction_bracket`, `residual_is_wasted_texture_rate`; EmpiricalAnchor
  `n600_covariance_totality_texture_trunk_audit_20260710`; consumers = shadow_controller +
  receiver_forward_parity). Tests in `test_einstein_pass_covariance_laws.py` (15 pass).
- **DSL:** N/A-with-rationale — analysis audit, no new trainer flag (the levers it forces — the
  movable event sidecar + T1/B2 gauge carrier — are owned by build-wave #377/#386 FEEDs).
- **DAG:** FEED-covariance-totality-audit (below in the DAG).

---

## OWN-ROUND-1 REVIEW (adversarial)

1. **Is "REFUTED-as-smooth-map / CONFIRMED-as-stated" a dodge?** No — the law as written explicitly
   allows an events bucket; the honest test of *its* claim is "is the residual events+gauge, not
   wasted texture?" and that is MEASURED yes (96.4% / 0% texture). The "REFUTED" level answers the
   naive over-reading ("factors through ξ smoothly") which the law never claimed.
2. **Is 0.42 an underestimate?** Yes, and stated as a LOWER bound: gt_poses is lossy (PoseNet 6-dim,
   not true (R,t)+calib), and kNN in 9-dim @ P=600 is data-starved. The rank-8 0.93 upper bound plus
   the near-monotone rise linear→quad→kNN (0.22→0.34→0.42) both say the true covariant fraction is
   higher; the exact-homography basis (not tested here — a quad-Taylor proxy only) is the next $0
   sharpening if it ever matters. It does NOT change verdict B/C: the *residual composition* (event+
   gauge, 0 texture) is what kills the trunk, and that is basis-independent.
3. **Could the Movable residual be ξ I failed to fit rather than genuine events?** Movables move
   independently of the camera by physics; attributing them to the event bucket is correct, and the
   +0.28 movable-area-change correlation is direct evidence they ARE creation/appearance events.
4. **rank-8 reproduction:** n96 cum-var 0.9559 reproduces the anchor 95.6% exactly (NO-FAKE: EDT
   argmax-roundtrip asserted 0 px); n600 is 0.9316 (slightly lower — more scene diversity, expected).
5. **Texture-trunk-lives-on-pose:** honestly UNDECIDED — this is a d_seg-partition audit; it says
   nothing measured about photometric d_pose legibility. Flagged, not claimed either way.
6. **Scope:** MEASURED n600 audit; advisory `[macOS-CPU research-signal]`, pointer UNMOVED. Verdicts
   are architecture/law verdicts, not exact rows.
