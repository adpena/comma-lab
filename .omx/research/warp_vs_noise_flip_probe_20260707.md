# Equal-L² warp-vs-noise flip-rate probe (Mallat GIS row 5) — deformation vs additive flip production — 2026-07-07

**Axis:** `[macOS-CPU advisory]` NON-PROMOTABLE. **Pointer 0.19110 UNMOVED** — means (a $0
mechanism measurement informing the #268 S_R reachability-weighting design), not an exact row.
All measurements n600 (ALL 600 pairs), frozen CPU-torch SegNet, never MPS. Live #205 run
untouched (this probe reads only the GT cache — no witness render at all).

**Question (FEED-08f row 5, operator GO):** Mallat's Group-Invariant-Scattering bound
(CPAM 2012; ‖Sx_τ − Sx‖ ≤ C·(‖∇τ‖_∞ + …)·‖x‖ — deformation-gradient load-bearing) predicts
that for a scattering-like cascade (SegNet's stride-2 stem + deep conv is in scope as an
approximate-scattering heuristic, NOT a theorem), **smooth diffeomorphic perturbations flip
the argmax near the separatrix far more efficiently per unit input energy than additive iid
noise** (which the cascade attenuates). If confirmed, the #268 through-R reachability surface
S_R should be parameterized by DEFORMATION sensitivity, not additive-perturbation sensitivity
— and UNIWARD-style additive cost maps are the wrong chart (consistent with the measured
msal_uni INERT verdict, `[[msal_uni_texture_proxy_inert_build_exact_sR_reachability_weight]]`).

## PRE-REGISTRATION (written BEFORE measurement)

### Channel (the exact detector leg)

Perturb the CAMERA-RES GT frame1 (`gt_n600.npz: gt_f1`, 874×1164×3 uint8 — the frame SegNet
scores; d_seg is last-frame-only) in FLOAT, then round/clip → uint8 (the honest contest
channel), then the contest scorer preprocess (bilinear → 512×384, normalize) → frozen SegNet
→ argmax. Flip = pixel whose argmax differs from the UNPERTURBED baseline argmax (recomputed
same-path; positive control: baseline must reproduce the cached `lstars`, and recomputed
margins must match the cached `margins` — mismatch rates reported).

### Conditions (10 SegNet forwards per pair: base + 4 warps + 4 matched noises + nz_x4)

| cond | perturbation | amplitude |
|---|---|---|
| base | none | — |
| wego_d05 / wego_d15 | **ego-consistent screw warp**: ground-plane diffeomorphism induced by the forward-translation twist ξ = d·e_z (tac.lie screw formalism; flat-ground IPM with the in-tree constants cam_h=1.2, fy=399.5·(874/384), v_h=174·(874/384)): output row v at forward Z_o samples source (v_s, u_s) with v_s−v_h = cam_h·fy_cam/(Z_o+d), u_s−c_x = (u−c_x)·Z_o/(Z_o+d); identity above horizon (displacement →0 continuously at the horizon) | d ∈ {0.05, 0.15} m (5%/15% of a real ~1 m inter-frame ego step; sub-px mid-field, a few px near-field — smooth + realistic) |
| wsm_a05 / wsm_a15 | **generic smooth warp**: Gaussian random displacement field (white noise smoothed σ=48 px, camera res), seeded per pair (`default_rng(1000+pair)`), joint-RMS-normalized | RMS ∈ {0.5, 1.5} px |
| nz_(each warp cond) | **iid Gaussian noise**, seeded (`default_rng(2000+10·pair+cond)`), scaled per pair per condition so the FLOAT L² equals that warp condition's ‖ΔI‖₂ exactly | matched |

### Metrics

Per pair per condition: flips per margin bin — bins on the recomputed baseline margin
(top1−top2 logit), edges **[0, 0.25, 0.5, 1, 2, 4, ∞)**; lane-involved flips (baseline or
perturbed argmax == class 1); realized float L² and post-uint8 L² (recorded — the uint8
round can suppress small perturbations differently per class; if the pooled post-uint8
energy ratio warp/matched-noise falls outside [0.8, 1.25] the primary verdict carries a
flag and a per-realized-energy-normalized secondary reading is reported).

### Instrument-validity controls (binding, per the 4db610af2 adversarial review — a metric
without measured dynamic range is not a finding)

- **Determinism floor:** re-forward the unperturbed baseline for pair 0 twice; must be
  bit-identical (CPU torch) — establishes flip noise floor = 0, so every measured flip is
  signal.
- **Perfect-endpoint control:** baseline argmax must reproduce the cached `lstars`
  (mismatch rate reported; expected ~0) and recomputed margins must match cached `margins`
  (max |Δ| reported) — verifies the instrument IS the one that produced the GT reference.
- **Known-worse control (dynamic range):** one extra condition `nz_x4` = iid noise at 4× the
  L² of wego_d15 — must produce substantially more flips than nz_wego_d15 (monotone).
- **Monotonicity gate:** pooled flips(wego_d15) > flips(wego_d05) and flips(wsm_a15) >
  flips(wsm_a05); pooled flips(nz_x4) > flips(nz matched to wego_d15).
- **Amplitude floor:** every condition must pool ≥ 10,000 flipped pixels across n600 (else
  the amplitude is below the instrument's discrimination scale). ANY gate failing ⇒ the
  affected comparison is INDETERMINATE-at-this-resolution, not a finding.

**Authority note:** all numbers use one instrument — the frozen CPU-torch SegNet forward
(`load_real_segnet("cpu")` + `preprocess_input`), the same path that produced the cached
`lstars`/`margins`; no trainer-verdict or probe-render numbers are mixed in.

### Pre-registered verdict thresholds

Primary statistic per warp condition c: pooled (micro-avg over all pairs) flip-count ratio

    ρ_c = flips(warp_c, margin<1) / flips(noise matched to c, margin<1)

(margin<1 = bins 0–2, the separatrix-adjacent population; 93% of real flips live at
margin<1 per the 4-lens memo).

- **DEFORMATION-DOMINATES**: ρ_c ≥ 2 for ALL FOUR warp conditions ⇒ S_R (#268) weighting
  should be deformation-sensitivity-based; UNIWARD-style additive cost maps are the wrong
  chart; grounds the replicate-vs-downweight split in the flicker lever (predictable
  ego-jitter = diffeomorphic = the dominant flip producer).
- **NOISE-COMPARABLE**: ρ_c ≤ 1.2 for all four ⇒ the GIS-stability reading does NOT
  discriminate at this scale; additive sensitivity is an adequate S_R chart.
- **MIXED**: otherwise — reported per warp class (ego-screw vs generic-smooth may separate:
  the ego warp concentrates on the road plane where the lane separatrix lives).

Secondary pre-registered readouts: full margin-binned flip-rate curves per condition
(the GIS prediction is margin-graded: deformation advantage should CONCENTRATE in the
lowest bins); lane-involved flip share per condition (does deformation preferentially
produce LANE flips — the flicker-decomposition connection).

### Triality routing (pre-committed)

Measured result → `update_equation_with_empirical_anchor` on
`margin_saliency_reachability_replaces_texture_proxy_v1` (the S_R line this probe informs);
a NEW equation is registered only if the result earns a law-shaped statement beyond that
anchor. DAG FEED + council-draft addendum + memory line per the standing triality discipline.

### Discipline

Chunked resumable foreground (atomic tmp+replace state); free-RAM ≥ 20 GiB gate per
invocation; VBATCH ≤ 6 pairs; peak RSS target ≤ 10 GiB; NO MPS; deterministic seeds recorded.
Tool: `tools/warp_vs_noise_flip_probe_n600.py`.

---

## RESULTS

(to be appended after measurement — every number from the probe's n600 JSON)
