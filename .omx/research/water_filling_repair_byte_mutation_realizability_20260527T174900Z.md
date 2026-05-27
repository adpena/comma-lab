# Water-Filling REPAIR — Byte-Mutation REALIZABILITY Micro-Probe (Catalog #139) — 2026-05-27T17:49:00Z

**Evidence grade:** `[macOS-MLX research-signal]` — NON-PROMOTABLE per Catalog #192/#127/#323.
**Score claim:** `false`. **Promotion eligible:** `false`. **Ready for exact-eval dispatch:** `false`.
**Subagent:** `water_filling_repair_realizability_20260527`.
**Lane:** `lane_water_filling_repair_byte_mutation_realizability_20260527`.
**Cost:** $0 — MLX-local on M5 Max GPU; NO paid GPU dispatch.
**Tool (NEW, owned by this lane):** `tools/probe_repair_byte_mutation_realizability.py`.
**Result artifact:** `.omx/state/water_filling_repair_byte_mutation_realizability_20260527T174111Z.json`.

This memo CLOSES the `INDETERMINATE` verdict from the proxy probe
(`.omx/research/water_filling_repair_probe_plus_op_palette_audit_20260527T164825Z.md`,
verdict `INDETERMINATE_UPPER_BOUND_PROXY_NOT_REALIZABLE`) with a REALIZED
measurement, not another proxy.

---

## TL;DR — the realized verdict

**NULL — `NULL_REALIZED_REDUCTION_DOMINATED_BY_RATE`.**

The mutation is **NOT a no-op** (every step genuinely moved the MLX scorer
output — `is_no_op=false` at all 9 sweep cells). The realizability assertion the
operator asked for is satisfied: the repair-direction mutation DOES propagate to
the decoded frames + scorer output.

BUT the **REALIZED** distortion-reduction is **NEGATIVE at every (K, step)** — moving
the highest-sensitivity decoder weights along the gradient-indicated descent
direction makes the contest distortion *worse*, not better. The proxy's
"repair marginal beats 25/N by ~6098×" is **empirically FALSIFIED**: not only does
the realized marginal fail to beat the rate cost, it has the wrong sign.

Mechanism (the honest explanation): the FEC6 frontier decoder weights are
**trained near a distortion minimum**. The per-tensor finite-difference gradient
gives only the *local linear slope*; at a finite repair step the *curvature*
dominates and the move overshoots into higher distortion. A trained frontier has
no free local descent to harvest — exactly why the proxy (which summed raw FD
magnitudes as if each were a realizable independent reduction, blowing past the
total 0.2532 distortion budget) was a pure upper-bound artifact.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog
#307, this is an **IMPLEMENTATION-level NULL at this operating point**, recorded as
a Catalog #313 **DEFER** (research-deferral with reactivation criteria), NOT a
paradigm KILL. Per the standing paradox-Half-2 verdict + the PACT-NeRV class-shift
directive, **class-shift / original-substrate training remains the dominant lever
regardless** — REPAIR joins `full_drop` in the falsified-at-this-operating-point set.

---

## The measurement (REALIZED, not proxy)

Exactly the operator-specified 6-step protocol, executed on the ACTUAL archive:

1. **Rank** archive bytes by master-gradient seg+pose sensitivity (L1 over 600
   pairs) from `.omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy`.
   Pick top-K, K ∈ {16, 64, 256}.
2. **Map** top-K archive bytes → the decoder tensors whose mantissa-byte spans
   own them (the master-gradient is a per-TENSOR FD projected per-BYTE, so a
   top-K byte set names a set of decoder tensors).
3. **Mutate** the implicated tensors' weights along the **repair direction** =
   the negative sign of a *freshly-recomputed signed* per-tensor central FD of
   the contest distortion (the producer's stored gradient took `abs`, so the
   signed descent direction is re-derived here). Step = `step_mult · fd_rel_eps ·
   RMS(tensor)`, step_mult ∈ {0.5, 1.0, 2.0}.
4. **Re-run** the SAME parity-validated MLX SegNet+PoseNet scorer oracle the
   gradient producer uses (`tac.master_gradient_mlx_extractor` decode + score
   pipeline — REUSED, not reimplemented) on the reconstruction from the mutated
   `state_dict`.
5. **Measure** REALIZED Δd_seg + Δd_pose vs the unmutated baseline.
6. **Compare** realized distortion-reduction to the rate cost `25·K/37545489`.

**Apples-to-apples confirmed:** `archive_matches_gradient_subject=true` — the
probe verified the gradient artifact's `archive_sha256` (`6bae0201fb082457…`)
equals the archive it decoded + scored. Same archive, same oracle, same operating
point.

**Realizability assertion (the operator's step 6):** the mutation MUST change the
scorer output. It does — `is_no_op=false` everywhere (abs Δd_seg / Δd_pose are
O(1e-5..1e-4), far above the 1e-9 no-op threshold). This is the answer to "do
repair bytes propagate?": **yes, they propagate — and they make it worse.**

### Per-K realized Δseg/Δpose/Δscore vs rate-cost table

Baseline operating point (unmutated, 64-pair): `d_seg=0.001112`, `d_pose=0.001386`,
`S_distortion = 100·d_seg + √(10·d_pose) = 0.228962`.

| K | step×eps | tensors moved | d_seg→ | d_pose→ | realized reduction | rate cost 25K/N | beats rate? | no-op? |
|---:|---:|:--:|---:|---:|---:|---:|:--:|:--:|
| 16 | 0.5 | 4/4 | 0.001111 | 0.001403 | **−5.948e-04** | 1.065e-05 | **NO** | no |
| 16 | 1.0 | 4/4 | 0.001113 | 0.001407 | **−9.621e-04** | 1.065e-05 | **NO** | no |
| 16 | 2.0 | 4/4 | 0.001120 | 0.001393 | **−1.046e-03** | 1.065e-05 | **NO** | no |
| 64 | 0.5 | 6/6 | 0.001116 | 0.001407 | **−1.216e-03** | 4.261e-05 | **NO** | no |
| 64 | 1.0 | 6/6 | 0.001123 | 0.001439 | **−3.278e-03** | 4.261e-05 | **NO** | no |
| 64 | 2.0 | 6/6 | 0.001137 | 0.001430 | **−4.372e-03** | 4.261e-05 | **NO** | no |
| 256 | 0.5 | 10/10 | 0.001125 | 0.001408 | **−2.225e-03** | 1.705e-04 | **NO** | no |
| 256 | 1.0 | 10/10 | 0.001139 | 0.001402 | **−3.324e-03** | 1.705e-04 | **NO** | no |
| 256 | 2.0 | 10/10 | 0.001188 | 0.001429 | **−9.357e-03** | 1.705e-04 | **NO** | no |

"realized reduction" = `−(S_distortion_mutated − S_distortion_base)`; **negative =
distortion got WORSE**. Every cell is negative → the repair-direction move is
*anti-helpful* on the trained frontier. The realized reduction grows MORE negative
with larger K and larger step (more weights moved + bigger overshoot), the exact
opposite of what a genuine sub-frontier marginal would show.

**Best realized cell:** K=16, step=0.5 → realized reduction **−5.948e-04** vs rate
cost +1.065e-05. The proxy claimed the top byte beats rate by ~6098×; the realized
measurement shows the move *loses* distortion-budget and adds rate — dominated on
both axes.

### Apples-to-apples verdict math

- **Proxy claim (FALSIFIED):** Σ-marginal beats 25/N by ~6098× (top-byte marginal
  4.06e-3 vs 6.66e-7). The realized top move is −5.948e-04 (wrong sign).
- **Physical-impossibility flag (confirmed root cause):** the proxy summed raw FD
  magnitudes across thousands of bytes → 0.26..2.06, exceeding the entire 0.2532
  distortion budget. That is the upper-bound artifact this realized probe replaces.
- **Wrong-derivative flag (confirmed):** the gradient measures `∂score/∂(byte_value)`
  (local slope), not `∂score/∂(repair byte)`. The realized finite-step move
  reveals the slope does not survive curvature — descent overshoots.
- **Rate-axis flag (unchanged):** the gradient has no rate dimension; the 25/N
  cost is fully external. The realized move pays rate AND loses distortion.

### Why the existing-weight descent move is the OPTIMISTIC bound

Mutating EXISTING decoder weights freely along the descent direction is a strictly
EASIER test than the literal canvas REPAIR op (which ADDS new sidecar bytes
carrying a residual correction at the same 25/N cost but with a weaker handle than
a free weight move). If even the free-weight descent cannot reduce realized
distortion below the rate cost — and here it cannot even reduce it *at all* — then
the literal byte-addition REPAIR is dominated *a fortiori*. The optimistic bound
being NULL closes the operator's question completely.

---

## The disposition (operator-facing)

- **NO FIRE candidate** is named. Per the proxy memo's gating logic + this
  realized falsification, declaring a REPAIR FIRE candidate would have repeated the
  unbounded-upper-bound error class. The realizable measurement says the move is
  not a frontier move; it is a frontier *regression*.
- **No numpy-portable inflate path is proposed** — there is no realized gain to
  ship. (Had the probe shown realized > rate, the FIRE path would have been a
  single-pass `SMOOTHED_RESIDUAL` receiver runtime, ≤200 LOC + ≤2 deps per HNeRV
  parity L4; that path is moot.)
- **REPAIR is DEFERRED (not killed)** at this operating point. Reactivation
  criterion: a class-shift substrate (PACT-NeRV / original-substrate training)
  that moves the operating point AWAY from the trained-frontier minimum — where a
  descent move would have real headroom — re-opens the repair question. On the
  current trained FEC6 frontier, there is no local distortion to repair.
- **Class-shift dominates, unamended.** Per the canonical paradox-Half-2 verdict
  (`paradox_half2_5d_canvas_repoint_empirical_verdict_20260527…`) + the standing
  PACT-NeRV / long-original-substrate-training class-shift directive
  (`feedback_pact_nerv_long_substrate_class_paradigm_shift_top_priority_20260527.md`),
  the highest-EV direction remains original-substrate class-shift, NOT another
  distortion-injection / repair operator.

---

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map contribution** — ACTIVE: the probe consumes the canonical
   per-pair master-gradient sensitivity map; the realizability finding (FD-value
   sensitivity ≠ realized recoverability on a trained frontier) is a reusable
   guard for any allocator/repair consumer of that map.
2. **Pareto constraint** — ACTIVE (resolving): the rate↔distortion facet is now
   measured, not proxied — the realized move pays rate AND loses distortion → it
   is strictly Pareto-DOMINATED. The probe records the facet is closed for REPAIR
   at this operating point.
3. **Bit-allocator hook** — ACTIVE: confirms `allocate_per_pair_bits`-style
   ranking identifies sensitive bytes correctly, but the realized cap is a hard
   guard: ranked-sensitive ≠ repair-recoverable on a trained frontier.
4. **Cathedral autopilot dispatch hook** — ACTIVE (NULL signal): do NOT fire a
   REPAIR budget; the realized probe is the gate and it returns NULL.
5. **Continual-learning posterior update** — ACTIVE: Catalog #313 probe-outcome
   DEFER row appended (`probe_id=water_filling_repair_byte_mutation_realizability_20260527`),
   reactivation criterion = class-shift operating-point move.
6. **Probe-disambiguator** — ACTIVE: this probe IS the disambiguator between
   "proxy-marginal-beats-rate (phantom FIRE)" and "realized-marginal-beats-rate
   (real FIRE)"; it resolves the prior INDETERMINATE to a measured NULL.

## Discipline anchors

Catalog #229 (premise verification — read prior probe + producer memo + extractor
before measuring; confirmed `archive_matches_gradient_subject`) + #139/#220/#272
(byte-mutation realizability lineage — THIS probe is that micro-probe) +
#192/#127/#323 (canonical Provenance; macOS-MLX non-promotable; all markers
false) + #296 (no predicted band without realizability check — this IS the
realized check) + #307 (the NULL is IMPLEMENTATION-level at this operating point,
NOT a paradigm-level kill) + #313 (probe-outcomes DEFER row LANDED) + #246 (any
future score claim requires paired contest-CUDA/CPU exact eval) + #157/#174
(canonical serializer + POST-EDIT `--expected-content-sha256`) + #206 (checkpoints
every 10 tool uses) + #314/#340 (sister-checkpoint guard honored).
