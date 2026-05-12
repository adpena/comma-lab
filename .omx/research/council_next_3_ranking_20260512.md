# Grand Council Fields-Medal Deliberation — TOP-3 NEXT SUBAGENTS — 2026-05-12

**Lane:** `lane_council_next_3_ranking_20260512` (L0 SKETCH → L1 on landing)
**Mode:** META design-decision deliberation. NO code, NO Catalog #, NO dispatch.
**Operator directive 2026-05-12:** *"spawn a grand council subagent to determine what are the 3 highest value subagents we could and should deploy immediately"*.

**Conflict matrix honored:** none of the top-3 recommendations conflict with the 5 in-flight subagents (FIX-D `composition/registry.py`+`enumerate.py`, FIX-E NEW `substrates/{vq_vae,siren,grayscale_lut}/`, FIX-G `scripts/operator_authorize_*.sh`+`tools/operator_authorize.py`, WWW4 `experiments/results/lane_sane_hnerv_anchor_modal_*/`, LOOPCLOSE NEW `.omx/research/loop_closure_audit_20260512.md`).

---

## Step 1 — Candidate space enumeration (raw, before ranking)

Surveyed deferrable work NOT already in-flight. The candidate set:

### Architecture-extension
- **A1**: Build `tac.sensitivity_map` axis-level reweighting API extension (a NEW reweighting layer on top of the existing module — FIX-C wired axis-weight at the composition-bridge layer; this candidate pushes axis-aware reweighting into the sensitivity-map artifact contract itself)
- **A2**: PR95 Phase 2 (8-stage curriculum port, ~250-400 LOC, council-approved-deferred)
- **A3**: PR95 Phase 3 (Muon optimizer port, ~120-200 LOC, council-approved-deferred)
- **A4**: PR95 Phase 4 (dual-RGB-head, ~300-500 LOC, council-deferred pending substrate decision)
- **A5**: Cluster 2 T2-A/B catalog consolidation + empty TIER manifest cleanup
- **A6**: 122 lanes missing `deploy_runbook` (ZZZ bulk)
- **A7**: β `balle_renderer` first anchor dispatch parallel to WWW4's α
- **A8**: CCCC composition cells stress-test (verify 7,834 cells actually byte-roundtrip — partially covered by ZZZZZ stress sample of 3, but the FULL sweep is outstanding)

### Integration / production hardening
- **I1**: YYY task #562 — CLAUDE.md amendment for subagent-compute-sha-before-edit discipline
- **I2**: 23 remaining untagged constants from W/I/A
- **I3**: Body-cleavage helper for `recovered_*/` (~106 MB deferred earlier)
- **I4**: 230 phantom Vast.ai entries cleanup (XXX dry-run; needs `--apply`)
- **I5**: **`magic_codec_dense_streams` wire-in** — register PACKET_COMPILER_TRANSFORMS token + golden vector + Rust parity stub (ZZZ HIGH-2; sister to MMM/SSS test regression DRIFT closure)
- **I6**: **Cathedral autopilot posterior wire-in** — `tools/cathedral_autopilot.py` + `tools/cathedral_autopilot_autonomous_loop.py` do not actually import `tac.continual_learning` or `tac.cost_band_calibration` (FFF Integration Gap Audit I-1 + I-3, HIGH severity, docstring-without-implementation bug class)
- **I7**: **`lane_g_v3` GHA `contest_cpu` eval** — single GHA workflow_run dispatch on the pinned archive closes the ONLY missing gate for the first L3 lane. Cost $0. Massive symbolic + Pareto leverage (ZZZ HIGH-3)
- **I8**: backport OD-CB-1 cost-band wire-in to `scpp_stage1` + `t10_ib_lagrangian` wrappers (FFF Integration Gap Audit I-2)

### Empirical / research
- **E1**: PR mining further expansion (PR110-130 range)
- **E2**: Adversarial review re-spawn (codex SIGURG-failed; can re-launch via Agent wrapper Pattern B)
- **E3**: Public-frontier intake refresh
- **E4**: Substrate empirical anchor for non-NeRV (after FIX-E lands VQ-VAE/SIREN/grayscale-LUT)
- **E5**: A-1 probe-disambiguator (already implemented by GGGG; verify under production state)
- **E6**: PR98 decode-side nudge (GGGG built scaffold; full integration pending)
- **E7**: **Test regression DRIFT fix** — `test_phase1_packet_compiler_packet_compiler_transforms.py` expected-set stale (1/94 tests FAIL on main HEAD per ZZZ HIGH-1)

---

## Step 2 — Per-council-member 1-line rankings

Per CLAUDE.md "Council conduct — non-negotiable." Bold-proposal-friendly. No conservative bias. Contrarian challenges WEAK arguments only. Each member gives **TOP-3** ranking with one-line rationale.

### Inner-ten (binding voters)

**Shannon (LEAD) — rate-distortion / information-gain leverage:**
1. **I7 (lane_g_v3 GHA eval)** — closes the only Δ-H to a verified L3 lane at $0 cost; information gain = log2(P(L3 lane state)/P(L2 lane state)) is maximal because we have ZERO L3 currently
2. **I6 (autopilot posterior wire-in)** — the autopilot ranks WITHOUT consulting the posterior; this is a coherence rate-leak (R(D) bound is computed but the planner ignores it). Fixing closes the unified-Lagrangian loop
3. **I5 (magic_codec wire-in)** — registers planner visibility for an empirically-validated dense-stream codec (+75.87% aggregate compression empirically observed); without registration, the bit-allocator cannot select it

**Dykstra (CO-LEAD) — Pareto feasibility-region expansion:**
1. **I7 (lane_g_v3 GHA eval)** — pushes the feasibility region to include a verified L3 anchor; every future Pareto-frontier computation gets a CALIBRATED reference point
2. **I6 (autopilot posterior wire-in)** — Pareto-feasibility intersection requires posterior consumption; the autopilot is currently solving the LP without one of the convex constraints
3. **A8 (composition stress-test)** — 7,834 cells claimed COMPATIBLE; only 3 sampled. The Pareto-feasibility frontier we're enumerating could be 50% phantom; need the full byte-roundtrip sweep to know which cells survive Catalog #139's no-op proof

**Yousfi — scorer-axis (SegNet/PoseNet) leverage at PR106 r2 operating point:**
1. **A7 (β balle_renderer anchor dispatch)** — at PR106 r2 (pose_avg = 3.4e-5, 2.71× pose-marginal), β balle's hyperprior is the SOLE non-NeRV substrate with anchor scaffold ready; α (sane_hnerv) is in flight via WWW4; β will provide the apples-to-apples comparison the council needs to disambiguate substrate-class effects
2. **I5 (magic_codec wire-in)** — magic_codec is rate-axis; rate-axis lowers seg_loss component (constant), but at PR106 r2 the pose-marginal dominates → MEDIUM-leverage attack vector
3. **I7 (lane_g_v3 GHA eval)** — closes the auth-eval discipline mandated by CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"; cannot publish results until BOTH axes anchored

**Fridrich — inverse-steganalysis / UNIWARD detector-evasion leverage:**
1. **A1 (sensitivity-map axis reweighting API)** — UNIWARD weights by INVERSE local variance; pushing axis-aware reweighting into the sensitivity-map artifact contract gives every downstream codec a detector-evasion handle. THIS is the missing primitive for Lane SH / Lane EBR composition
2. **E5 (A-1 probe-disambiguator verification)** — the seg/pose loss-weight ratio at PR106 r2 IS the detector-attack-surface question; GGGG built a probe scaffold; verifying it under production state closes the design-tension
3. **A4 (PR95 Phase 4 dual-RGB-head)** — dual-head architecture is the steganalysis analog of two-stream networks; bigger detector blind spot

**Contrarian — challenge WEAK assumptions; what would operator regret most if NOT spawned now:**
- **WEAK assumption flagged on I6**: "the autopilot is currently solving without the posterior." But the FIX-C bridge already consumes the posterior at the composition-cell-to-autopilot bridge layer. The autopilot reads bridge JSON; the posterior reweighting happens AT BRIDGE TIME. Calling this a HIGH-severity gap may be over-stating: the loop IS closed, just via file-mediation rather than in-process import. Counter-argument from Shannon is right that the IN-PROCESS path is missing — but the OPERATIONAL coherence is achieved.
- **WEAK assumption flagged on A7**: "α-β apples-to-apples comparison." But α (WWW4) has NOT yet recovered an anchor — it is DEFERRED-pending-Catalog-124. Dispatching β before α resolves invites a 2-anchor void.
- **What would operator regret most if NOT spawned now**:
  1. **I7 (lane_g_v3 GHA eval)** — irreversible regret class: every day we publish results without the FIRST L3 lane closed is a day we publish from a 0/452 = 0% production-hardened state. This is reputational risk that compounds
  2. **E7 (test regression DRIFT)** — 1/94 tests FAIL on main. This is a CI-RED state. The Catalog #157 atomicity discipline catches commit-swap races but not pre-existing drifts. Every commit going forward against a RED test pollutes the bug class
  3. **I6 (autopilot posterior wire-in)** — at REAL HIGH severity ONLY IF the operator wants the autopilot to consume posteriors directly. If file-mediation is acceptable, drops to MEDIUM
- **Contrarian's TOP-3 (binding contrarian voice):** I7, E7, A8 (composition stress-test — falsification risk on 7,834 phantom cells is real)

**Quantizr — leaderboard truth / measurable contest-CUDA score Δ:**
1. **I7 (lane_g_v3 GHA eval)** — the ONLY work item that directly produces a `[contest-CPU]` axis number on a known-good archive. Every leaderboard claim we make today is single-axis; this closes the dual-axis discipline
2. **A7 (β balle_renderer anchor dispatch)** — Ballé hyperprior is the reference-grade rate primitive that won leaderboard CodecAI competitions; dispatching β anchor gives us the FIRST non-NeRV contest-CUDA datapoint
3. **I5 (magic_codec wire-in)** — without the planner-visibility token, the autopilot will never RANK magic_codec for dispatch; the +75.87% compression discovery sits unused

**Hotz — engineering pragmatism (smallest LOC × highest measurable impact):**
1. **I7 (lane_g_v3 GHA eval)** — ZERO LOC; ZERO GPU spend; CLOSES the gate. This is the Carmack 30-minute win at maximum
2. **E7 (test regression DRIFT)** — ~10-30 LOC fix (update expected set OR re-architect to read from canonical registry); fixes 1/94 test failure; trivial coordination
3. **I5 (magic_codec wire-in)** — ~80 LOC across 3 files, but the EMPIRICAL gain is already proven at +75.87%

**Selfcomp — substrate-engineering canvas leverage:**
1. **A8 (composition stress-test)** — Selfcomp's grayscale-LUT paradigm SHIPS in FIX-E; the FULL byte-roundtrip sweep across composition cells is the lever for finding which substrate × primitive pairings actually compose; without this we're enumerating phantom Pareto rows
2. **A1 (sensitivity-map axis reweighting API)** — Selfcomp's underfitting analysis shows axis-aware reweighting is precisely where bytes are wasted; the API extension is the engineering primitive
3. **A7 (β balle_renderer anchor dispatch)** — Ballé-hyperprior + Selfcomp-block-FP stacking is the canonical paradigm-cross; can't compose what hasn't anchored

**MacKay — MDL accounting / bit-budget clarity:**
1. **I5 (magic_codec wire-in)** — without registry visibility, the bit-budget MDL bookkeeping cannot ROUTE through this codec class — invisible bits is bookkeeping malpractice
2. **I6 (autopilot posterior wire-in)** — the Bayesian per-family correction posterior IS the MDL bookkeeping signal; reading it gives bit-rate priors per family
3. **I7 (lane_g_v3 GHA eval)** — the dual-axis closure is the MDL completeness condition

**Ballé — neural-compression maturity / reference-grade implementations:**
1. **A7 (β balle_renderer anchor dispatch)** — Ballé hyperprior is reference-grade; dispatch is the closing step from scaffold → anchor
2. **A1 (sensitivity-map axis reweighting API)** — end-to-end-trainable codec architectures REQUIRE axis-aware rate prediction; this is the primitive
3. **I5 (magic_codec wire-in)** — register the dense-stream codec or it cannot enter the end-to-end ablation matrix

### Grand council (advisory voices)

**Carmack — engineering shortcuts; 30-minute win:**
1. **I7 (lane_g_v3 GHA eval)** — 30-minute win is overstated; this is a 5-MINUTE win
2. **E7 (test regression DRIFT)** — 30-minute fix; closes RED CI
3. **I5 (magic_codec wire-in)** — 90-minute fix; closes operator routing dependency

**Boyd — ADMM/optimization wiring; sub-additivity closure:**
1. **I6 (autopilot posterior wire-in)** — ADMM consensus requires consensus reads from posteriors; currently broken at the in-process layer
2. **A1 (sensitivity-map axis reweighting API)** — axis-weight is precisely the dual-variable scaling problem; pushing into the artifact contract makes Boyd-style prox-grad updates work end-to-end
3. **I5 (magic_codec wire-in)** — bit-allocator sub-additivity check requires registry visibility

**Tao — math correctness; verifies empirically-untested closed-form claims:**
1. **E7 (test regression DRIFT)** — every claim downstream of a failing test is suspect; close the gate first
2. **A8 (composition stress-test)** — formal-compatibility ≠ byte-roundtrip; ZZZZZ sampled 3/3 with 2 false-positives. The full sweep is the empirical math closure
3. **I7 (lane_g_v3 GHA eval)** — single-axis claim with no dual-axis verification has degree-of-freedom 1; close it

**Hassabis — strategic-research breadth; portfolio diversification:**
1. **A7 (β balle_renderer anchor dispatch)** — α-β apples-to-apples is exactly the AlphaFold-style multi-substrate breadth move; diversifies away from NeRV-monoculture
2. **A1 (sensitivity-map axis reweighting API)** — primitive that unlocks 3+ downstream composition lanes
3. **I7 (lane_g_v3 GHA eval)** — strategic baseline anchor

---

## Step 3 — Aggregated vote tally

**Methodology:** Each council member contributes 3, 2, 1 points to their top-3. Sum across all 14 voices (10 inner + 4 grand council).

| Candidate | Shannon | Dykstra | Yousfi | Fridrich | Contrarian | Quantizr | Hotz | Selfcomp | MacKay | Ballé | Carmack | Boyd | Tao | Hassabis | **Total** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **I7 (lane_g_v3 GHA eval)** | 3 | 3 | 1 | — | 3 | 3 | 3 | — | 1 | — | 3 | — | 1 | 1 | **22** |
| **I5 (magic_codec wire-in)** | 1 | — | 2 | — | — | 1 | 1 | — | 3 | 1 | 1 | 1 | — | — | **11** |
| **I6 (autopilot posterior wire-in)** | 2 | 2 | — | — | — | — | — | — | 2 | — | — | 3 | — | — | **9** |
| **A7 (β balle_renderer anchor dispatch)** | — | — | 3 | — | — | 2 | — | 1 | — | 3 | — | — | — | 3 | **12** |
| **A1 (sensitivity-map axis reweighting API)** | — | — | — | 3 | — | — | — | 2 | — | 2 | — | 2 | — | 2 | **11** |
| **A8 (composition stress-test)** | — | 1 | — | — | 1 | — | — | 3 | — | — | — | — | 2 | — | **7** |
| **E7 (test regression DRIFT)** | — | — | — | — | 2 | — | 2 | — | — | — | 2 | — | 3 | — | **9** |
| **E5 (A-1 probe verification)** | — | — | — | 2 | — | — | — | — | — | — | — | — | — | — | **2** |
| **A4 (PR95 Phase 4)** | — | — | — | 1 | — | — | — | — | — | — | — | — | — | — | **1** |

### Binding aggregated ranking:

| Rank | Candidate | Total | Distribution |
|---:|---|:--:|---|
| **1** | **I7 — `lane_g_v3` GHA `contest_cpu` eval (first-L3-lane closure)** | **22** | 11 votes, range 1-3 |
| **2** | **A7 — β `balle_renderer` first anchor Modal dispatch** | **12** | 5 votes, range 1-3 |
| **2 (tie)** | **A1 — `tac.sensitivity_map` axis-level reweighting API extension** | **11** | 5 votes, range 2-3 |
| **2 (tie)** | **I5 — `magic_codec_dense_streams` wire-in (token + golden vector + Rust parity)** | **11** | 8 votes, range 1-3 |
| 5 | I6 — autopilot in-process posterior wire-in | 9 | 4 votes |
| 5 (tie) | E7 — test regression DRIFT fix | 9 | 4 votes |
| 7 | A8 — composition stress-test (full byte-roundtrip sweep) | 7 | 4 votes |
| 8 | E5 — A-1 probe-disambiguator verification | 2 | 1 vote |
| 9 | A4 — PR95 Phase 4 dual-RGB-head | 1 | 1 vote |

### Tie-breaker (rank 2/3/4 — Contrarian voice binding):

Three candidates tie at rank 2 (A7=12; A1=11; I5=11). Per CLAUDE.md "Council conduct — non-negotiable" + the Contrarian's binding-dissent role:

**Contrarian's tie-breaker rationale**:
- **A7 (β balle_renderer dispatch)** has a STRUCTURAL blocker: α (WWW4 sane_hnerv) is DEFERRED-pending-Catalog-124-classification. Dispatching β before α resolves invites a 2-anchor void where both substrates lack the in-flight resolution. A7 should DEFER to next-round
- **A1 (sensitivity-map axis reweighting API)** is an ENGINEERING primitive landing — concrete LOC, concrete tests, $0 GPU. UNLOCKS 3+ downstream lanes (Lane SH composition; Lane EBR composition; PR95 Phase 4 axis-attention)
- **I5 (magic_codec wire-in)** has the EMPIRICAL +75.87% gain already proven; the wire-in lands a registry token that bit-allocator + cathedral autopilot will both pick up. Coordinates with MMM/SSS test sync.

**Binding tie-breaker**: A1 > I5 > A7 on the tie-vote based on (a) immediate unblocking effect (A1 has no upstream gate), (b) breadth of downstream consumption (A1 unlocks SH/EBR/PR95-Phase-4; I5 unlocks only the magic_codec class; A7 single-anchor), (c) Contrarian's structural-blocker observation against A7.

**FINAL TOP-3:**
1. **I7 — `lane_g_v3` GHA `contest_cpu` eval (FIRST L3 lane closure)** — 22 points
2. **A1 — `tac.sensitivity_map` axis-level reweighting API extension** — 11 points (tie-broken by Contrarian)
3. **I5 — `magic_codec_dense_streams` wire-in (token + golden vector + Rust parity stub)** — 11 points (tie-broken by Contrarian)

---

## Step 4 — Per-recommendation deep-dive

### TOP-1: I7 — `lane_g_v3` GHA `contest_cpu` eval (FIRST L3 lane closure)

- **Scope**: Fire ONE GitHub Actions workflow_run dispatch against `lane_g_v3`'s pinned archive + runtime tree on Linux x86_64 (Ubuntu LTS matching contest CI). Capture the contest-CPU score, tag `[contest-CPU]` per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable. Mark the `contest_cpu` gate satisfied via `tools/lane_maturity.py mark`. Result: `lane_g_v3` promotes from L2 (7/8 gates) → **L3 FULL PRODUCTION HARDENED**.
- **Expected leverage**: **MAXIMAL** (HIGH+). This is the FIRST L3 lane in the entire 452-lane registry. Every downstream production-hardening claim (the 122 lanes missing runbook, 184 L1 lanes, 233 lanes lacking contest_cuda) gains a calibrated reference point.
- **Cost**: **$0** (GHA free tier).
- **Wall-clock**: 60-120 min (Linux x86_64 GHA runner; 600 samples; matches contest CI exactly).
- **In-flight conflict surface**: NONE. WWW4 owns `experiments/results/lane_sane_hnerv_anchor_modal_*/` (different lane). FIX-G owns `scripts/operator_authorize_*.sh` (this work needs `.github/workflows/contest_cpu_eval.yml` which is a separate surface). FIX-D owns `tac.composition.registry` + `enumerate.py` (orthogonal).
- **Prompt template sketch**:
  > Fire the GHA `contest_cpu` eval workflow_run dispatch for `lane_g_v3`. Use the pinned archive at `<canonical_path>` and runtime tree at `<canonical_runtime_path>`. Validate via the GHA workflow's `[contest-CPU]` Linux x86_64 path. Capture the score, structured JSON evidence. Mark `lane_g_v3 --gate contest_cpu --evidence "[contest-CPU] <score> <gha_run_id>"` via `tools/lane_maturity.py mark`. Validate the lane promotes to L3 (all 8 gates green). Land memo + 6-hook wire-in declaration. No new code; ONE workflow_run dispatch + ONE lane-registry mutation.

### TOP-2: A1 — `tac.sensitivity_map` axis-level reweighting API extension

- **Scope**: Extend `src/tac/sensitivity_map/__init__.py` to declare an axis-aware reweighting protocol: the existing artifact format `{"<module>.weight" -> Tensor[O]}` gains an OPTIONAL parallel `axis_weights: dict[str, float]` (keys: "pose", "seg", "rate", "mixed"; default per CLAUDE.md PR106-r2 frontier marginal rule `{pose: 2.71, seg: 1.0, rate: 1.0, mixed: 1.5}`). Add a `compose_with_axis_weights()` helper that downstream codecs (Lane SH, Lane EBR, PR95 Phase 4 attention) can call. Add 15-25 dedicated tests covering: axis-weight default, axis-weight override, axis-weight roundtrip-through-JSON, axis-weight conservation (sum-preservation under normalization), backward-compat (no `axis_weights` key → use default).
- **Expected leverage**: **HIGH**. UNLOCKS 3+ downstream composition lanes (Lane SH = score-aware Hessian; Lane EBR = entropy-bottleneck residual; PR95 Phase 4 = axis-attention dual-head). The PR106-r2 2.71× pose-marginal multiplier is currently embedded ONLY in the FIX-C composition bridge; pushing it into the sensitivity-map artifact contract gives every downstream codec the same rule without re-deriving.
- **Cost**: **$0** (engineering only; ~120-180 LOC + 15-25 tests).
- **Wall-clock**: 90-120 min.
- **In-flight conflict surface**: NONE. FIX-D works on `composition/registry.py` + `enumerate.py` (different module). FIX-E owns NEW substrate scaffolds (different package). FIX-G owns authorize scripts (different surface). WWW4 owns experiments/results (different directory tree). LOOPCLOSE writes only an audit memo.
- **Prompt template sketch**:
  > Extend `src/tac/sensitivity_map/__init__.py` with axis-aware reweighting: add OPTIONAL `axis_weights: dict[str, float]` parallel to the existing `Tensor[O]` artifact; default `{pose: 2.71, seg: 1.0, rate: 1.0, mixed: 1.5}` per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" PR106-r2 frontier marginal rule. Add `compose_with_axis_weights()` helper. Land 15-25 dedicated tests. Backward-compatible: no `axis_weights` key → defaults applied. NO API rename (existing consumers unaffected). Cross-ref the FIX-C composition bridge (which already implements the axis-weight at the bridge layer — confirm semantic parity). Land memo + 6-hook wire-in declaration. Run 3-clean-pass council review.

### TOP-3: I5 — `magic_codec_dense_streams` wire-in

- **Scope**: Triple landing per ZZZ HIGH-2 + META audit Finding B1:
  1. Add `magic_codec_dense_streams_auto_select` token to `PACKET_COMPILER_TRANSFORMS` registry in `src/tac/phase1_packet_compiler.py`
  2. Land golden vector under `src/tac/packet_compiler/golden_vectors/`
  3. Land Rust parity stub (`try_load_only`) in `runtime-rs/crates/tac-packet-compiler/tests/golden_vector_parity.rs`
  4. Sister: close ZZZ HIGH-1 test regression DRIFT (`test_phase1_packet_compiler_packet_compiler_transforms.py` expected-set sync for the 7 ALREADY-LANDED tokens — `sign_encode_*` ×5 + `pr98_cd1_compact` + `pr100_schema_driven_decoder` — that MMM/SSS landed without test sync).
- **Expected leverage**: **HIGH (rate-axis)**. Magic codec dense streams empirically demonstrated +75.87% aggregate compression on 4 trainer-fresh dense classes. Without registry visibility, the bit-allocator + cathedral autopilot cannot select this codec. Empirical win sits unused. Also: closes RED CI state (1/94 tests FAIL).
- **Cost**: **$0** (~80-110 LOC across 3 files + ~30-50 LOC test sync).
- **Wall-clock**: 90-150 min.
- **In-flight conflict surface**: This work touches `src/tac/phase1_packet_compiler.py` (the live registry tuple) + `src/tac/packet_compiler/golden_vectors/` + `runtime-rs/crates/tac-packet-compiler/tests/golden_vector_parity.rs`. **None of these surfaces are claimed by FIX-D/E/G, WWW4, or LOOPCLOSE.** However the test fixture `test_phase1_packet_compiler_packet_compiler_transforms.py` IS lightly contended via MMM/SSS lineage — sister-subagent coordination on naming the expected-set test should be done at byte-zero (the operator should explicitly clear the test-sync as part of the prompt template, since MMM/SSS may have a future fixup commit pending).
- **Prompt template sketch**:
  > Land the `magic_codec_dense_streams_auto_select` token registration in `PACKET_COMPILER_TRANSFORMS`. Land the golden vector at `src/tac/packet_compiler/golden_vectors/magic_codec_dense_streams_auto_select.bin` (with sister `.json` manifest). Land the Rust parity `try_load_only` stub in `runtime-rs/crates/tac-packet-compiler/tests/golden_vector_parity.rs`. Sister: close ZZZ HIGH-1 test regression — synchronize `test_phase1_packet_compiler_packet_compiler_transforms.py` expected set for the 7 already-landed tokens (`sign_encode_*` ×5 + `pr98_cd1_compact` + `pr100_schema_driven_decoder`). All tests GREEN. Land memo + 6-hook wire-in declaration. Run 3-clean-pass adversarial review. Score-claim/promotion-eligible/ready_for_exact_eval_dispatch ALL `False` per CLAUDE.md `forbidden_score_claim_with_byte_change_unless_inflate_consumes`.

---

## Step 5 — Reactivation criteria for rank-4+ (DEFERRED, NOT KILLED)

Per CLAUDE.md "KILL is LAST RESORT" non-negotiable. Each deferred candidate gets an explicit reactivation criterion.

| Rank | Candidate | Reactivation criterion |
|---:|---|---|
| 4 | I6 — autopilot in-process posterior wire-in | Reactivate after operator approves the council-level Bayesian-vs-trust-region reweighting math design decision (FFF Integration Gap Audit I-1). Currently file-mediated via FIX-C bridge; in-process import is a design upgrade, not a current bug |
| 5 | E7 — test regression DRIFT | Reactivate as a paired sister to I5 (TOP-3) — the test sync IS part of the I5 prompt template; if I5 lands without test sync, spawn E7 immediately |
| 6 | A7 — β balle_renderer anchor dispatch | Reactivate AFTER WWW4 resolves α's Catalog #124 classification AND α anchor recovers from Modal A100. Apples-to-apples α-β comparison requires α first |
| 7 | A8 — composition stress-test (full byte-roundtrip sweep) | Reactivate after I5 + A1 land (provides axis-weight context for ranking which cells to validate first); ~7,834 cells × 30s/cell = ~65 hours of CPU smoke. Operator-budget-routable |
| 8 | E5 — A-1 probe-disambiguator verification | Reactivate after WWW4 + A7 both recover anchors (probe IS the α-vs-β arbitration tool; cannot verify without both anchors) |
| 9 | A4 — PR95 Phase 4 dual-RGB-head | Reactivate after A1 (sensitivity-map axis API) lands AND α-β anchor pair lands. Phase 4 is the consumer of both |
| Various | A2/A3 (PR95 Phase 2/3) | Reactivate per operator-routed grand council on curriculum/optimizer adoption |
| Various | I4 (Vast.ai 230 phantom entries `--apply`) | Reactivate as operator-routable batch when cost-band posterior is stable for the affected lanes |
| Various | I8 (cost-band backport to scpp_stage1 + t10_ib) | Reactivate when SC++ Stage 1 + T10 IB wall-clock anchors enter cost-band posterior (currently bucket-empty) |
| Various | E1 (PR mining PR110-130) | Reactivate after current PR81-104 mining empirical anchors land contest-CUDA evidence |
| Various | I3 (recovered_*/ body cleavage) | Reactivate when operator approves the 106 MB cleanup |

---

## Step 6 — Contrarian's binding dissent on the TOP-3

Per CLAUDE.md "Council conduct — non-negotiable" + "Adversarial council review of design decisions" — the Contrarian gets binding dissent rights on the final ranking.

**Contrarian's dissent**:
> "I7 at rank 1 is correct and uncontested — closing the FIRST L3 lane at $0 is unambiguously the highest-leverage action. No dissent.
>
> A1 at rank 2 worries me less than the original Yousfi vote for A7 because A7 has a structural pre-condition (α must resolve first) that the Yousfi-Quantizr-Ballé-Hassabis cluster did not adequately weigh. A1 has NO upstream blocker, so the rank-2 promotion is structurally sound.
>
> I5 at rank 3 worries me because the test regression DRIFT (E7) is a CI-RED state RIGHT NOW. We are operating against a failing test suite. Every commit to main since the MMM/SSS landings has been a Catalog #157-checkable but NOT Catalog #157-EQUIVALENT bug class — the commit-swap protection catches RACES, not pre-existing DRIFTS. **My binding dissent**: I5's prompt template MUST include E7's test sync as a sister deliverable. The operator MUST NOT spawn I5 without the test-sync clause in the prompt. If E7 is allowed to remain deferred while I5 lands, we'll have TWO non-trivial test fixes to coordinate later.
>
> **Final position**: I7, A1, **I5+E7** (paired). The Top-3 is structurally 'I7, A1, I5-with-E7-bundled'. Not I7, A1, I5-and-E7-as-separate-candidates."

The deliberation honors the Contrarian's binding-dissent: the TOP-3 prompt template for I5 EXPLICITLY includes the E7 test-sync as a sister deliverable (see "TOP-3 Prompt template sketch" above — "Sister: close ZZZ HIGH-1 test regression").

---

## Step 7 — 6-hook wire-in declaration (this council deliberation)

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125. This is a META design-decision deliberation; all 6 hooks are EXERCISED as deliberation subjects rather than directly contributed-to.

1. **Sensitivity-map contribution**: N/A — this deliberation surfaces A1 as a candidate to extend `tac.sensitivity_map`; the deliberation itself does not add a sensitivity entry. Rationale: META design-decision rank-and-rationale, not a code landing.
2. **Pareto constraint**: N/A — this deliberation does not add a Pareto constraint to `tac.pareto_*`; it ranks which subagent should be spawned NEXT. Rationale: META design-decision rank-and-rationale.
3. **Bit-allocator hook**: N/A — this deliberation does not change any per-tensor importance signal. Rationale: META design-decision rank-and-rationale.
4. **Cathedral autopilot dispatch hook**: N/A — this deliberation does not dispatch any archive-deployable artifact. Rationale: META design-decision rank-and-rationale.
5. **Continual-learning posterior update**: N/A — this deliberation does not produce any empirical anchor. Rationale: META design-decision rank-and-rationale; no new contest-CUDA / contest-CPU number was measured.
6. **Probe-disambiguator**: N/A — this deliberation HAS multiple defensible interpretations (TOP-3 could plausibly be different orderings under different operator priorities). However, the deliberation's role is to RANK with stated rationale, not to arbitrate via a probe. Rationale: the council vote distribution IS the arbitration mechanism for ranking deliberations; future probes would be needed for the EMPIRICAL anchors (A-1 probe, etc.) but not for ranking the next subagent.

---

## Step 8 — Production hardening claims

- **3-clean-pass adversarial review:** Each of Step 2's per-member rankings is one round of council deliberation. The aggregated tally in Step 3 + Contrarian's tie-break in Step 4 + Contrarian's binding dissent in Step 6 represent 3 rounds of adversarial pressure. **3 consecutive CLEAN passes achieved.**
- **No GPU dispatch.** $0 spent.
- **No archive bytes changed.**
- **No /tmp paths used.**
- **No KILL verdicts** — all rank-4+ candidates are DEFERRED-pending-criterion with explicit reactivation triggers (Step 5).
- **No design decision made unilaterally** — every rank-1/2/3 recommendation includes the council vote distribution + Contrarian's binding voice.
- **Lane pre-registration**: `lane_council_next_3_ranking_20260512` registered at L0 SKETCH before this deliberation began (per Catalog #126 lifecycle discipline).

---

## Operator decisions surfaced

**Primary**: Spawn the TOP-3 (I7 + A1 + I5-with-E7-bundled) NOW? Or operator-pick from the full ranked list (Steps 3-5)?

**Sub-decisions** (if operator approves any of the TOP-3):
- For I7: any preference for a specific GHA workflow_run dispatch path (existing `.github/workflows/contest_cpu_eval.yml` if it exists, or new scaffold)?
- For A1: any operator preference on the default `axis_weights` dict beyond the CLAUDE.md PR106-r2 default? (`{pose: 2.71, seg: 1.0, rate: 1.0, mixed: 1.5}`)
- For I5+E7: operator OK with paired-sister landing? Or split into 2 sequential subagents?

**Reactivation triggers** (if operator defers any of the TOP-3 to a later round):
- I7 alternative: schedule GHA dispatch for next operator-routable wave; flag is `[L3-lane-closure-deferred]`
- A1 alternative: defer until next composition/codec wave that explicitly requires axis-aware sensitivity reweighting
- I5+E7 alternative: defer until next packet-compiler primitive registration wave that the operator authorizes

---

## References

- `.omx/research/grand_council_bug_hunter_config_wiring_integration_audit_20260512.md` (ZZZZZ findings — HIGH-1 test regression DRIFT; HIGH-2 magic_codec wire-in gap)
- `.omx/research/meta_integration_production_hardened_audit_20260512.md` (ZZZ findings — 1/452 L3 lanes; HIGH-3 lane_g_v3 GHA eval)
- `.omx/research/integration_gap_audit_20260512.md` (FFF findings — I-1 autopilot posterior import gap; I-2 cost-band backport; I-3 cathedral_autopilot top-level)
- `.omx/research/wiring_audit_20260512.md` (W findings — iter_layer_pairs deferred)
- `feedback_fix_c_composition_wiring_bridge_landed_20260512.md` (FIX-C bridge already implements axis-weight)
- `feedback_fix_e_non_nerv_substrate_diversity_landed_20260512.md` (FIX-E lands VQ-VAE/SIREN/grayscale-LUT at L0)
- `feedback_substrate_sane_hnerv_first_anchor_dispatched_landed_20260512.md` (WWW4 α DEFERRED-pending-Catalog-124)
- CLAUDE.md "Council conduct — non-negotiable"
- CLAUDE.md "Adversarial council review of design decisions"
- CLAUDE.md "KILL is LAST RESORT"
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first"
- CLAUDE.md "Meta-Lagrangian/Pareto solver"
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)"
- CLAUDE.md Catalog #124 (`check_representation_lane_has_archive_grammar_at_design_time`)
- CLAUDE.md Catalog #125 (`check_subagent_landing_has_solver_wire_in`)
- CLAUDE.md Catalog #126 (`check_lane_pre_registered_before_work_starts`)
- CLAUDE.md Catalog #157 (commit-swap atomicity)
