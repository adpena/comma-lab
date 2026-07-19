# v10 RATE-CRUSH — Phase-1 lossless close-out + pivot to the description axis (2026-07-19)

Lane `lane_v10_ratecrush_20260719` (respawned agent; predecessor died mid-run — its code
and n24 receipts were recovered from `.omx/tmp/ratecrush_recovery_20260719/` and
`<SSD>/evidence/v10_ratecrush_20260719/`, then continued, not restarted).
`research_only=true` · axis `[macOS-CPU local rate measurement] NON-PROMOTABLE` ·
`score_claim=false` · pointer `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**.
Operating manual honored: every number below is MEASURED or labeled DERIVED/EXTRAPOLATED;
verdicts carry scope (`docs/operating_manual_craft_handoff.md`).

Mid-session the operator re-scoped the charter (binding): the exact-plane lossless family
is structurally dead as a frontier path; stop the n600 lossless scale-up, keep the coder
rankings + decode-exactness harness as donors, and pivot to bytes(DESCRIPTION of ŷ).
This memo records both halves honestly.

## 1. Phase-1 rung table (lossless crush of the frozen C1 scorer planes)

Byte identity to the officially-scored C1 planes (Y0 `5e86e419…`, Y1 `6a731946…`) pins
distortion to the official capstone report (`d_seg 0.00015196`, `d_pose 0.00010184`,
distortion sum **0.047108**); implied S = 0.047108 + 25·TOTAL/37,545,489. TOTAL values
below the official row are n24/n48-extrapolated local coder measurements, not archives.

| rung (codec → exact bytes) | TOTAL bytes | implied S | shippable? |
|---|---:|---:|---|
| predictor-residual-u8.v1 — OFFICIAL n600 archive | 409,526,925 | **272.73 (official row)** | YES (the measured C1 spine) |
| JXL lossless e9, both planes | 253,302,698 [n24-extrap] | 168.71 | YES mechanically — codec wired end-to-end (see §2), decode-exact, pip `imagecodecs` decodes cjxl streams byte-exactly 8/8 at 20.7 ms/plane (~25 s/1200 planes, inside the 30-min budget; external decoder FREE per upstream/README.md:118) — but **pointless vs the budget box** |
| FFV1 level3 | 265,711,124 [n24-extrap] | 176.97 | dominated by JXL |
| WebP lossless z9 | 266,316,200 [n24-extrap] | 177.38 | dominated |
| MED(LOCO-I)+brotli-q11 planar (best fully self-contained) | 275,734,040 [n24-extrap] | 183.65 | dominated |
| exact-residual conditional floor (#541 n48, excl frame0) | 200,469,250 [n48-extrap] | 133.53 | lower BOUND of the family, not a payload (frame0 uncharged) |
| #541 complete production floor (incl frame0 bootstrap) | 398,401,225 [n48-extrap] | 265.33 | the honest charged floor |

Pass-3 n24 completions (receipt `rank_streams_n24_pass3.json`, exact coder bytes):
- **R2 (min-coder)**: brotli-q11 wins on MED residuals everywhere — med+lzma 246.1K,
  med+zstd 246.0K vs med+brotli 230.4K B/plane-pair. No coder swap pays.
- **R1 (temporal)**: temporal∘MED 254,455 vs intra MED 227,029 B over the same 23 chained
  pairs — temporal prediction LOSES at every composition tried (mod256+br, smooth+br,
  ∘MED+br, ∘MED+lz), consistent with inter-coded x264/x265 losing to intra JXL in the
  donor ranking. `verdict_scope: formulation` — mod-256 temporal differencing on frozen
  scorer planes, n24.

### Family verdict

**RATE-DEAD vs the frontier budget. `verdict_scope: family` — exact-plane storage under
ANY lossless entropy stage.** Grounds: (a) the budget box — beating S=0.19108 at
exact-realization distortion (0.000193) requires TOTAL ≤ **286,682 B (477.8 B/pair)**;
at capstone distortion it is 264,000 B (440 B/pair); (b) the measured conditional floor
of the exact-residual family is ~334 KB/pair (#541 n48: 16,037,540 B/48) — **~700× over
the box**, before charging the ~330 KB/pair frame-0 bootstrap; (c) the donor sweep spans
five independent codec families (brotli/zstd/lzma stream ranks, JXL, WebP, FFV1,
x264/x265 lossless) within a 1.9× spread — the information content of near-exact planes
is megabytes/pair. This is the entropy argument, measured, not asserted. Consistent with
the 07-19 ledger row (`seg_and_pose_solved_exact_lattice_realization_one_rd_axis_20260719`):
exact-residual = RATE-DEAD; the one open axis is bytes(generator + band-slack).

### Donors KEPT (committed this landing)

- `src/tac/codec/v10_jxl_plane_codec.py` — sha-custodied fail-closed JXL two-plane codec;
  decode backends: pip `imagecodecs` (contest-runtime path, byte-exact proven) → `djxl`.
  Prices/carries any future compact payload section; the per-plane SHA custody is the
  decode-exactness harness for ANY future compact receiver.
- Additive wire-in: `v10_production_receiver` (Y_CODEC_IDS + build/decode branches) and
  `v10_two_plane_timing_receiver` (`_validate_packet` + the single expansion dispatch) —
  12-pair round-trip exact through build→parse→two-plane split; predictor-residual spine
  regression PASS; all 71 existing v10 receiver/codec tests pass.
- `experiments/v10_ratecrush_rank_streams.py` (+pass-3), `…_rank_donor_coders.py`,
  `…_build_jxl_archive.py` (n600 builder+verifier — NOT run at n600 per re-scope).
- Receipts: `<SSD>/evidence/v10_ratecrush_20260719/{rank_streams_n24{,_pass2,_pass3}.json,
  rank_donor_coders_n24.json, r0a_jxl_n24_verify_e9.json, r0b_pip_decoder_compliance.json}`.

## 2. Description axis — first-hand state + the R-D table (REVISED PRIMARY)

Budget box (DERIVED from the score law + official/advisory distortion rows):

| distortion anchor | TOTAL budget for S<0.19108 | per-pair |
|---|---:|---:|
| capstone official (d_seg 1.52e-4, pose exact) | 264,000 B | 440 B |
| exact-lattice realization (d_seg 9.66e-7 n600 advisory, d_pose 9.3e-10) | 286,682 B | 477.8 B |
| feasibility condition | at 100 KB TOTAL need d_seg < 1.24e-3 · at 236 KB need d_seg < 3.39e-4 · box CLOSES at ~287 KB | |

Measured/derived description-axis points (all consumed first-hand this session):

| point | TOTAL bytes | d_seg through exact solve + hard oracle | status |
|---|---:|---|---|
| exact-lattice realization ŷ=y (n600) | ~200–400 MB (exact family floor) | **9.66e-7 MEASURED** (114/117,964,800 px; fp32 ULP-tie class) | realized, RATE-DEAD |
| #541 rung-E production archive (n48) | 31,873,460 B /48 → ~398 MB eq. | **1.234e-4 MEASURED** (fp32 admissibility boundary) | realized, RATE-DEAD |
| #549 secant curve, 9 lossy points (n24) | ~2.22 MB/pair range-residual → ~1.33 GB eq. | 1.86e-5 … 2.14e-2 MEASURED; all adjacent secants pay the 150.182 B/1e-6 break-even; #536 knee = margin_m0p3↔precision_drop1, marginal gap 4.14e-12 score/byte | realized, locally KKT-paying, stranded ~4,600× above the box |
| PDW1 power-diagram target coefficients (#543) | **306 B Brotli MEASURED** (prefix 0..194/600) | **NONE** — extraction BLOCKED at frame 195 (fp32-Torch vs float64 label disagreement at one near-boundary pixel); only feature-pullback mismatch 2.35% exists, explicitly NON-EQUIVALENT to d_seg | inside box by 3 orders; UNREALIZED |
| MDL MS-complex contour (+ξ) | 228,764 B (+7,195) = 235,974 B — **DERIVED HEURISTIC ESTIMATE** (post-Brotli /2 shared-edge), not an emitted payload | **NONE** — no contour→plane receiver grammar exists | inside box; UNREALIZED, bytes themselves unverified |
| necessity-by-inversion anchors | camera-support 1.66%; bytes→edges Road-Lane 61% (#449/necessity ledgers) | — | design input: the counted statistic concentrates on inter-class edges |

### Verdict (the charter question, answered plainly)

**No measured point yields S < 0.19108. The 100–300 KB box contains ZERO
(total-bytes, d_seg)-measured points — every realized description is ≥200 MB-class, and
every ≤300 KB description is unrealized.** Nothing measured this session or found in the
stores crosses the box; the best realized implied S anywhere is ~133.5 (a floor bound,
not a payload).

### Dominating-term diagnosis (the next crux, named exactly)

- In every REALIZED family the bytes are dominated by two terms of equal magnitude:
  **the dense per-pixel signed residual (~334 KB/pair)** and **the frame-0 bootstrap
  (~330 KB/pair — 50% of the #541 complete floor; frame0 is charged, never free)**.
  No entropy stage moves either by more than ~1.9×.
- For the descriptions already inside the box (PDW1 306 B; MS-contour ~236 KB derived),
  **bytes are NOT the binding term — REALIZATION is**: (a) the #543 fp32-vs-fp64
  boundary-label authority blocker (frame-195 instance) stops the power-diagram
  extraction before any d_seg can exist; (b) no receiver grammar expands PDW1/contour
  coefficients into a uint8 camera preimage, so the exact solve + hard oracle have
  nothing to measure. The box-feasibility condition is quantified: a ~236 KB description
  must realize d_seg < 3.39e-4; the only distortion-proxy we have for the current PDW1
  fit (feature-pullback 2.35%) is ~2 orders above that if it translated 1:1 — so after
  the realization blocker falls, the second crux is the fit's realized distortion, and
  the necessity anchors (support 1.66%, Road-Lane edges 61%) say the repair budget
  should be spent on inter-class edges.
- **Next unit, directly aimed**: declare the PDW1 receiver arithmetic to be the frozen
  CPU-Torch fp32 forward (the same move that made the lattice realization exact), close
  the #543 frame-195 instance under that single declared arithmetic, then measure
  (bytes, d_seg) of the realized power-diagram ŷ̂ at n24 through the existing hard
  oracle — the first point that can actually land inside the box.

## Stores consulted

`v10_capstone_first_byteclosed_row_20260719.md` · `seg_secant_rd_curve_20260719_codex.md`
+ n24 v2 JSON (sha `2894096590…`) · `v10_lattice_rate_verdict_and_composition_20260719.md`
(incl. n600 replay 9.66e-7) · `constructive_solver_541_20260719_codex.md` ·
`v10_power_diagram_byteclose_findings_20260718.md` · `mdl_ms_complex_K_lower_bound_20260718.md`
· `necessity_dseg_calibration_20260715.md` · `p0_449_frozen_segnet_necessity_close_verdict_20260716.md`
· upstream/README.md:114-118 · predecessor receipts (SSD evidence dir).

Six-hook wire-in: sensitivity-map N/A (no new per-axis weights — rate-only rungs recorded
here); Pareto constraint ACTIVE (budget box + family RATE-DEAD verdict constrains the
selector); bit-allocator N/A (no per-tensor change); cathedral dispatch N/A (research_only);
continual-learning ACTIVE (this memo + receipts are the posterior rows); probe-disambiguator
N/A (single interpretation; blockers named). # FORMALIZATION_PENDING:budget-box constants are
direct arithmetic on the registered score law + existing seg_rate_breakeven_v1; no new equation.
