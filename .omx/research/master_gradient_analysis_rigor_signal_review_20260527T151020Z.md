# MASTER-GRADIENT ANALYSIS — ADVERSARIAL RIGOR + SIGNAL REVIEW (2026-05-27)

- timestamp_utc: 2026-05-27T15:10:20Z
- agent: claude (`mg_rigor_review_20260527`) — adversarial RIGOR+SIGNAL review per operator concern *"need to review master gradient analysis for rigor and signal"*
- lane_id: `lane_master_gradient_analysis_rigor_signal_review_20260527`
- git_head_at_review: `5c3456d02`
- scope: $0 — MLX-local / numpy / CPU only; NO paid GPU dispatch. READ-MOSTLY adversarial review. The only writes are THIS verdict memo. I did NOT edit `src/tac/master_gradient_mlx_extractor.py`, the cathedral consumer package, the canonical_equations_registry, or any `*pact_nerv*` file (sister-owned per Catalog #314/#340).
- authority: planning + observability ONLY. Every numeric claim below is `[macOS-MLX research-signal]` / `score_claim=false` / `promotable=false` per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#127/#323. NOT a contest score, NOT a frontier claim.
- discipline anchors: Catalog #229 (read all 5 artifacts + 2 authority npy in full before any verdict) + #192/#127/#323 (canonical Provenance non-promotable) + #307 (paradigm-vs-implementation classification) + #287 (every claim has a NUMBER, not an assertion) + #314/#340 (sister-checkpoint guard; READ-ONLY on sister scope).
- mission_contribution (Catalog #300): `frontier_protecting` — establishes the TRUE epistemic status of the synergy≈0 verdict so strategy is NOT routed on a misattributed mechanism.

---

## TL;DR VERDICT

The synergy≈0 / "single-op Pareto-optimal" conclusion is **CORRECT IN DIRECTION but its STATED MECHANISM is a PROJECTION + PROBLEM-SPEC ARTIFACT, not an empirical property of the contest score surface discovered from per-pair data.**

- **The per-pair leverage MAP is genuine signal** — 64 distinct pair columns, byte-identically the first 64 columns of the 600-pair artifact, deterministic, real-video GT.
- **The "near-collinear operator gradients → synergy ≈ 0" finding is NOT discovered from the per-pair data.** It is a **rank-1 mathematical tautology** of `_build_multiop_problem_spec`: every operator gradient is constructed as `(seg_aggregate, pose_aggregate, rate) × leverage_i` — a scalar multiple of ONE shared base direction. I measured the operator-gradient matrix RANK = **1**, max cosine-distance = **0.0** across all 5 productive operators. A rank-1 operator-gradient set has zero synergy-axis polytope volume FOR ANY input artifact, per-pair or aggregate, MLX or PyTorch-authority.
- **Running the PyTorch-autograd authority would NOT settle the synergy question.** The authority artifacts have the SAME ~28-tensor-uniform byte projection (26 distinct byte-rows; zero within-tensor variation), and the synergy collapse lives DOWNSTREAM in the problem-spec builder, which is identical for both paths.

Per Catalog #307: the synergy≈0 conclusion is an **IMPLEMENTATION-LEVEL fact about the sweep's problem-spec construction**, NOT a **PARADIGM-LEVEL fact about multi-op composition on the contest score surface**. The verdict memo `paradox_half2_5d_canvas_repoint_empirical_verdict_20260527T142332Z.md` is unusually honest (it explicitly labels MULTIOP_BEATS_V14V2 a label-artifact and keeps everything non-promotable) BUT it attributes the synergy collapse to "near-collinear operator gradients **at the frontier**" — implying an empirical score-surface property — when the collinearity is structurally guaranteed by the spec builder before any frontier geometry enters.

---

## CRUX 1 — Uniform-mantissa-projection collinearity artifact: **CONFIRMED ARTIFACT (byte axis), but it is SHARED with the authority and is NOT the synergy driver**

**Quantified (`master_gradient_fec6_frontier_mlx_per_pair_64pair_20260527.npy`, shape `(178517, 64, 3)`):**

| metric | value |
|---|---|
| nonzero SEG byte-rows | 177968 / 178517 |
| **EXACT distinct nonzero SEG byte-rows** | **25** |
| matrix rank of distinct SEG rows (over pair axis) | 25 (npairs=64, ntensors=28) |
| EXACT distinct nonzero POSE byte-rows | 27 |
| rate-axis max-abs | 0.0 |

The 178,517-byte axis is effectively **only 25–27-distinct-row rank** — exactly the ~28-decoder-tensor structure (3 tensors zeroed on degenerate RMS → collapse to 25/27). This is the direct consequence of `project_per_tensor_sensitivity_to_per_byte` (`master_gradient_mlx_extractor.py:652-655`): every byte in a tensor span gets `per_byte_seg = (seg_per_pair × scale_mag) / numel` **broadcast uniformly**. So the artifact carries 28-tensor-rank in the byte axis, NOT 178k-byte-rank. **This crux's factual claim is TRUE.**

**BUT — this is NOT MLX-specific and NOT the synergy driver.** The PyTorch-autograd AUTHORITY artifacts have the SAME collapse:

| authority artifact | shape | nonzero seg rows | **distinct seg byte-rows** | byte-axis row-value CHANGES | within-block variation |
|---|---|---|---|---|---|
| `master_gradient_a1_headered_diagnostic_8pair_per_pair_20260518.npy` | `(178162, 8, 3)` | 161779 | **26** | 25 | **0.000e+00** |
| `master_gradient_pr101_lc_v2_diagnostic_8pair_per_pair_20260518.npy` | `(178158, 8, 3)` | 161779 | **26** | 25 | **0.000e+00** |

The authority tool (`tools/extract_master_gradient.py:1799-1809`) lands in the compressed-region `seg_mass / n_comp` uniform-spreading path → per-tensor-uniform blocks with EXACTLY zero within-block variation. So the byte-axis rank collapse is a **shared methodology property of BOTH tools**, not an MLX heuristic flaw.

**Do the 6 productive operators map to distinct tensors or collapse?** They do NOT map to distinct tensors at all — the operators consume the per-pair canvas cells (sum-over-byte-axis = a tensor-weighted scalar per pair), not individual tensors. The collinearity that matters is at the OPERATOR-gradient level (Crux 3), where it is rank-1 by construction.

**Crux 1 verdict:** The byte-axis is a genuine 28-tensor-rank projection artifact (TRUE), but it is shared with the authority and is upstream of — not the cause of — the synergy≈0 result.

---

## CRUX 2 — Heuristic-prior ranking vs PyTorch-autograd authority: **UNTESTABLE with existing artifacts; the memo's "agrees in RANKING" claim is UNVERIFIED**

**A direct apples-to-apples per-byte/per-pair Spearman is NOT possible with the artifacts on disk:**

- The MLX heuristic artifact is on the **fec6 frontier** archive (`6bae0201…`, `(178517, 64, 3)`).
- The only PyTorch-autograd authority artifacts are on **DIFFERENT archives** (`a1_headered` + `pr101_lc_v2`) with **8 pairs**, not 64.
- Different archive bytes → different byte addresses; different pair counts → unalignable pair axis. **No valid per-byte or per-pair rank-correlation exists.**

The producer memo's claim that the MLX-FD ranking "agrees in RANKING with PyTorch-autograd within FD discretization" (line 86) is therefore an **UNVERIFIED assertion**, not a measured number. It is plausible (the MLX scorer port is parity-validated per MLX-ARCH-5) but it has NOT been tested apples-to-apples.

**What IS measurable:** the two authority artifacts' per-pair leverage rank-correlate at **Spearman ρ = 1.0** for both SEG and POSE (n=8). This is a weak signal (n=8, near-identical HNeRV substrates with shared latent/frame ordering) but indicates per-pair leverage RANKING is substrate-stable at the authority level — supporting the heuristic prior's *ranking* utility while saying nothing about its *magnitudes*.

**Crux 2 verdict:** The "agrees with authority" claim is currently UNTESTED. To actually verify it requires running `tools/extract_master_gradient.py` on the **same fec6 archive** for the same pair set and computing the Spearman. Until then the heuristic-prior ranking is plausible-but-unconfirmed.

---

## CRUX 3 — Verdict robustness / "class-shift is the ONLY lever": **the synergy≈0 EVIDENCE is a rank-1 tautology, NOT empirical; the conclusion is DIRECTIONALLY right but ROUTES on the wrong mechanism**

**The decisive measurement.** I instrumented `_build_multiop_problem_spec` on the live 64-pair canvas (192 cells, 5 productive operators):

```
operator gradient norms (n_ops x 3 axes):
[[2.924e-06  1.931e-05  0.0]
 [1.020e-06  6.740e-06  0.0]
 [1.705e-06  1.126e-05  0.0]
 [1.270e-06  8.388e-06  0.0]
 [2.041e-06  1.348e-05  0.0]]
RANK of operator-gradient matrix: 1
max cosine-distance from unit row0 across operators: 0.0
```

**Every operator gradient is `(seg_aggregate, pose_aggregate, 0) × leverage_i`** (sweep tool lines 244-246: `norms.append((seg * leverage, pose * leverage, rate * leverage))` where `seg/pose/rate` are the SINGLE shared `_axis_cell_value` aggregate and `leverage = |best_predicted_delta_score|`). The matrix is **RANK 1**. A rank-1 operator-gradient set has a feasible polytope with **zero synergy-axis volume by construction** — the Dykstra solver's "near-collinear → synergy ~0" output (118 iters, residual 5.43e-14, synergy `-7.61e-13`) is the **arithmetic image of a rank-1 input**, not a discovery about the frontier score surface.

This holds identically for the archive-aggregate path (it produced the same conclusion at `-2.55e-19`) — confirming the per-pair re-point did NOT change the mechanism. The per-pair structure changed the per-operator `leverage` scalars (and thus `best_single_op_delta` from −0.0154 → see Crux 4) but NOT the shared per-axis direction that forces rank-1.

**Per Catalog #307 classification:** the synergy≈0 result is an **IMPLEMENTATION-LEVEL fact** about the sweep's problem-spec construction (a rank-1 operator basis cannot express synergy), NOT a **PARADIGM-LEVEL fact** that "multi-op DISTORTION composition cannot beat single-op at the contest frontier." The latter is a much stronger claim that this sweep architecture is structurally INCAPABLE of falsifying — because it never gives the operators distinct gradient DIRECTIONS, only distinct magnitudes along one shared axis.

**Crux 3 verdict:** The "single-op Pareto-optimal / class-shift is the ONLY lever" conclusion should be DOWNGRADED from "empirically VINDICATED" to "DIRECTIONALLY plausible but evidentially circular." It is NOT rigorous enough to ROUTE strategy on AS-STATED. It needs either (a) a problem-spec builder that gives operators genuinely distinct per-axis gradient directions (from the per-pair canvas, not a shared aggregate × scalar) OR (b) explicit acknowledgment that the synergy term is structurally zero by construction and therefore conveys no information about real multi-op composability.

---

## CRUX 4 — 64-pair vs full-600 consistency: **CONSISTENT (honest pose-accumulation), no instability**

**Quantified:**

| metric | 64-pair | full-600 |
|---|---|---|
| first-64 per-pair coords match | — | **max-abs diff = 0.0** (64-pair IS byte-identically the first 64 columns of the 600) |
| operating-point d_seg | 0.001112 | 0.001222 |
| operating-point d_pose | 0.001386 | 0.001716 |
| leverage sign dist SEG (n_neg/n_pos) | 45/19 (70% neg) | 294/306 (balanced) |
| leverage sign dist POSE (n_neg/n_pos) | 36/28 | 314/286 (balanced) |
| sweep best_single_op ΔS | −0.01543 | −0.03597 |
| **sweep synergy `predicted_multiop_extra_delta`** | **−7.61e-13** | **−1.46e-12** |
| Dykstra feasible / residual / iters | True / 5.43e-14 / 118 | True / 1.25e-13 / 159 |
| V14-V2 threshold | −7.66e-6 | −7.66e-6 |

The 64-pair is the deterministic first-64-column slice of the 600 (max-diff exactly 0.0). The d_seg/d_pose upward drift is honest pose-accumulation (more pairs → fuller distortion). One REAL caveat the verdict memo did not surface: **the 64-pair sample is biased** — its leverage is 70% negative-sign (45/19), while the full 600 is balanced (294/306). So the 64-pair was a non-representative early-pairs subset for the LEVERAGE SIGN distribution, even though the synergy CONCLUSION is unchanged. The synergy term roughly doubled (−7.6e-13 → −1.5e-12) but stayed ~6 orders below threshold; the verdict holds across both pair counts.

**Crux 4 verdict:** CONSISTENT — no instability. Honest pose-accumulation. Minor caveat: 64-pair leverage-sign distribution is a biased subset of the full population (the verdict memo's "representative-but-truncated" framing is right for the conclusion, slightly optimistic for the sign distribution).

---

## OVERALL GRADES

### RIGOR grade: **B− as a HEURISTIC PRIOR / F as AUTHORITY / the analysis is honest about the latter but mis-attributes the synergy mechanism**

- As a **heuristic prior for probe RANKING**: trustworthy. The module + memo are exemplary in false-authority discipline (Codex hardening flips it to `tensor_fd_uniform_decompressed_projection_heuristic_v1`, refuses `master_gradient_anchors.jsonl` authority via 3 explicit blockers, marks everything `macOS-MLX research-signal` non-promotable). The per-pair leverage map is deterministic, real-video-GT, reproducible.
- As **AUTHORITY**: correctly NOT claimed (the blockers fire). The 28-tensor byte-projection is shared with the PyTorch tool, so "authority" would require a genuine per-byte/per-param projector neither tool has.
- The **one rigor gap**: the paradox-closer attributes synergy≈0 to "near-collinear operator gradients at the frontier" — an empirical-sounding claim — when the collinearity is rank-1 BY CONSTRUCTION in `_build_multiop_problem_spec`. This is a misattribution of mechanism, not a false conclusion.

### SIGNAL grade: **REAL per-pair leverage signal in the ARTIFACT; ZERO synergy signal in the SWEEP (the synergy term is a structural constant, not a measurement)**

- The `(178517, N, 3)` artifact's per-pair axis IS genuine signal (64/600 distinct pair leverage coordinates from real-video distortion + central FD).
- The byte axis is 28-tensor-rank projection-collapsed (artifact, shared with authority).
- The synergy term `predicted_multiop_extra_delta` is **NOT a measurement** — it is the arithmetic image of a rank-1 operator basis, structurally pinned near zero for any input. It conveys no information about real multi-op composability.

---

## ROUTING RECOMMENDATION

1. **The class-shift routing conclusion STANDS as a PRIOR but must be RE-LABELED.** "Single-op Pareto-optimal; class-shift is the only lever" is a reasonable strategic prior (the leverage map shows single-op `replace_many` dominates), but it is NOT empirically established by the synergy≈0 sweep — that sweep is structurally incapable of finding synergy. Route on the leverage map's single-op ranking, NOT on the (circular) synergy verdict. Per Catalog #307, downgrade the synergy≈0 line in the paradox-closer verdict from `HARD-EARNED-EMPIRICALLY-VERIFIED` to `IMPLEMENTATION-LEVEL-RANK-1-TAUTOLOGY` (the conclusion may still be true; the EVIDENCE cited does not establish it).

2. **The synergy≈0 verdict does NOT need the PyTorch-authority run to "confirm" it** — that run would reproduce the SAME rank-1 tautology because the collinearity is in the spec builder, not the artifact. Confirming with the authority is a non-test.

3. **Operator-routable next step that WOULD actually settle multi-op synergy** ($0, design + code): fix `_build_multiop_problem_spec` to give each operator a genuinely DISTINCT per-axis gradient DIRECTION derived from that operator's own per-pair cell footprint (e.g. `replace_many` vs `temporal_coherence` should weight seg-vs-pose differently because they touch different pair subsets), then re-run. If the rank->1 operator basis STILL yields synergy ~0, that is a genuine empirical finding about the score surface. Until the spec builder is rank-deficient, the sweep cannot falsify multi-op synergy and the verdict is uninformative on that question.

4. **Lower-priority $0 fidelity item:** run `tools/extract_master_gradient.py` (PyTorch authority) on the SAME fec6 archive + same 64-pair set, compute the per-pair leverage Spearman vs the MLX heuristic, and record the actual number — this would convert the producer memo's UNVERIFIED "agrees in ranking" claim into a measured one (Crux 2). It does NOT bear on the synergy question (#2 above).

**No paid FIRE-phase should be gated on the synergy≈0 verdict as currently stated** — the verdict's mechanism is circular. A single-op `replace_many` / drop-one candidate remains the highest-EV paid candidate per the leverage map (independent of the synergy artifact), and Track-A class-shift remains a legitimate parallel design lane — but justify those on the leverage map + frontier-gap arithmetic, NOT on a rank-1 synergy tautology.

---

## CANONICAL PROVENANCE (Catalog #323)

Every numeric claim in this memo is `evidence_grade=macOS-MLX research-signal` / `axis_tag=[macOS-MLX research-signal]` / `score_claim=false` / `promotion_eligible=false` / `ready_for_exact_eval_dispatch=false` / `hardware_substrate=darwin_arm64_m5_max_macos_mlx_advisory`. Sources: direct numpy inspection of the on-disk artifacts (shas in their sidecars) + two live sweep runs (`/tmp/sweep64.json`, `/tmp/sweep600.json`) + direct instrumentation of `_build_multiop_problem_spec`. NOT a contest score; contest-CUDA/CPU exact-eval per Catalog #246 remains the only path to any score/frontier/PR claim.

## SISTER COORDINATION (Catalog #314/#340)

READ-ONLY on all sister scope. I did NOT edit `src/tac/master_gradient_mlx_extractor.py` (automation/wire-in sister), the cathedral consumer package, `.omx/state/canonical_equations_registry.jsonl`, or any `*pact_nerv*` file (PACT-NeRV sister). My only write is THIS memo. I did NOT run the PyTorch authority tool (it would write a disjoint `.omx/state/` artifact; I deferred it as the Crux-2 operator-routable since it does not change any verdict).

## DISCIPLINE CLOSURE

- Catalog #229 premise verification: read `master_gradient_mlx_extractor.py` + 2 memos + populator (lines 695-854) + sweep tool (lines 218-400) + both authority npy in full BEFORE any verdict.
- Catalog #307 paradigm-vs-implementation: synergy≈0 classified IMPLEMENTATION-LEVEL (rank-1 spec-builder tautology), not PARADIGM-LEVEL (multi-op-can't-beat-single-op).
- Catalog #287 no-overstatement: every claim carries a NUMBER (rank 1, cosine-dist 0.0, 25/27 distinct rows, −7.61e-13 / −1.46e-12 synergy, max-diff 0.0, ρ=1.0).
- Catalog #192/#127/#323 canonical Provenance: all non-promotable macOS-MLX.
- Catalog #206 checkpoint discipline: 3 in-progress checkpoints via `tools/subagent_checkpoint.py`.
- Catalog #110/#113 APPEND-ONLY: NEW memo; ZERO mutation of the producer memo, the paradox-closer verdict, or any sister artifact.
