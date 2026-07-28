# ddm_tb1 — THE renderer build: SPEC_tr1 trained partition→pixel renderer (T0+T1 landed; A2 race live)

**Date:** 2026-07-28 · **Arm:** `ddm_tb1_20260728` · **Charter:** tb1 (fork-adjudicated GO; fd2 verdict
`SEG_REALIZATION_GAP_AT_UINT8_DOMINANT` merged `e4bacb5d39`).
**Evidence axis:** `[macOS-CPU/MLX advisory]` throughout — `score_claim=false · promotion_eligible=false`.

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Nothing in this arm is a score; every
realized d_seg row below is the frozen CPU-torch scorer on macOS (advisory). The competitive bar is
`effective_frontier 0.172` (PR130); the GOAL bar is `min(0.15, 0.172)`. The pointer moves ONLY through a
byte-closed `upstream/evaluate.py` row (§R6 chain, not reached in this arm).

## STORES CONSULTED (recall-first, multi-pass)

CLAUDE.md + AGENTS.md (NO-FAKE #3/#6/#7/#8; rule-118; eval_roundtrip/EMA/QAT; MPS-train-vs-authority
split; #402 liveness; #205 verdict-chunk law; DSL-as-SoT) · tb1 charter (scratchpad) ·
`SPEC_tr1_trained_partition_renderer_20260728.md` (all six sections; §S1–S4 realized here) ·
`ddm_fd2_posenull_gn_disambiguation_20260728.md` (A1 lesson; the 36-pair instrument geometry; canary
discipline) · `ddm_eu1_sol_ultra_eureka_hunt_20260728.md` (P3.0 quotient/fiber; §P3.1 G1-LOTTO +
EU1-G1-LOTTO-N600 gate spec; rule-118 selector-accounting flag) · fd1 memo (Rung-2 routing; fd1r wall-clock
law; #383 pose-terminal) · pp1 band lemma (`ddm_pp1_correction_stream_position_band_v1`: ρ_c=5.02e-4) ·
fc1 budgets (187,727 B @0.172 / 154,522 B @0.15) · sn1 asymmetry
(`codex_findings_ddm_sn1_segnet_telemetry_asymmetry_20260723_codex.md` + SSD
`ddm_sn1_segnet_telemetry_asymmetry_n600_20260723/` — verified present before consuming) ·
`ema_decay_run_geometry_v1` LawRef (evaluator consumed live) · witness substrate
(`experiments/train_witness_realized_through_R_mlx.py`: `make_loss_fn`/`_apply_R`/
`_torch_R_to_camera_uint8`/`cpu_verdict_d_seg_argmax_batch`; the MLX-GPU render-precision
PORT-FIDELITY lesson at `_torch_R_to_camera_uint8`'s docstring) · `mlx_scorer_adapters.py` ·
`power_diagram_witness.open_stored_npy_memmap` · gt cache `mlx_fleet_gt_cache/gt_n600.npz` (MAIN) ·
MEMORY.md current-state (box-retired/warp-closed endgame; pose=terminal-6-eq; pools law; constants-are-
poison; shared-venv hijack; reaper-kill class; no-old-lineage ban).

## What landed (commits on this worktree, off main `e4bacb5d39`)

- `db105b6f51` — **T0 scaffold**: `experiments/train_tr1_partition_renderer_mlx.py` (trainer, both
  variants) + `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` (DSL Lever factories + fail-closed
  validate + sealed tickets) + `src/tac/tests/test_ddm_tb1_tr1_renderer.py` (18 tests, green).
- `ea9a982fe8` — **T0.1 forces wiring**: day-one confound alarms (frozen_epoch liveness, gnorm-hijack),
  F2 event-fallback knee, #685 param_delta_rms race-fairness telemetry, DUTY_TO_MEASURE extensions.
- `58835f6f29` — **T0.2 differentiable topology**: per-class Betti-0 + GT-components-ERASED +
  smallest-surviving-component telemetry in every A1 gate (closes the equal-flip-counts-hide-wrong-
  components aliasing gap); lane-pool topology-loss race DUTY entry (pools law).
- `17166ee9c4` — **T0.3 A1-instrument fix — the #85 EMA shadow-lag confound CAUGHT LIVE at the first
  n600 window**: at derived decay 0.99867 the ep9 gate rendered a ~41%-zero-init-seed shadow that scored
  0.842 (WORSE than gray init 0.507) and fired a FALSE A1 alarm chain that would have killed the window
  at ep14 with a wrong "inherited gap" verdict. Fix is structural and from the law itself: gate/confirm
  basis = LIVE params before the registered warmup boundary W=2/(1−d), EMA shadow after; basis recorded
  on every gate row; basis change rebases the A1 comparison + flip baseline; resume ⇒ warm shadow.
  Aborted window preserved (`t2_n600_plain_aborted_emawarmup_gate_artifact/`, named reason). This is the
  confound immune system working end-to-end: alarm LOUD → cause disambiguated (shadow-lag, not training
  gap: smooth was descending and the T1 precedent + preflight 1-epoch live-gate 0.334 both contradicted a
  real regression) → same-turn structural fix + custody.
- SSD receipts: `/Volumes/VertigoDataTier/pact/ddm_tb1_20260728/` (micro-smokes, T1/T2 run dirs with
  sealed tickets, telemetry.jsonl, per-stage EMA-shadow checkpoints, window receipts).

## The object (as built; COMPOSE-not-duplicate)

Token-grid latent field (P=frames, lattice 384/D × 512/D, c channels, uint8-STE-quantized description)
→ small conv renderer (conv0 + one conv per ×2 upsample + head; GELU; sigmoid×255) → **the canonical
witness loss** `make_loss_fn` via its `render_fn` hook (`compute_pose=False`, `w_pose=0`): render →
`_apply_R` (contest-exact MLX R: bicubic↑384→874 → uint8-STE @ camera → bilinear↓) → MLX SegNet →
measured seg forms (ce → tau_softplus at the knee event; margin_hinge raced). Frame_1-only (SegNet reads
the last frame; frame_0 is seg-free). Pose TERMINAL (#383) — no PoseNet in the trunk, by design.

**A1 (the fd2 binding transfer lesson) is structural**: every `gate_every` epochs, the EMA shadow renders
the gate set **fp32 on the MLX CPU stream** (the witness PORT-FIDELITY lesson: the MLX-GPU forward is
reduced-precision; the verdict render must not be) → `_torch_R_to_camera_uint8` → frozen CPU-torch SegNet
argmax → realized d_seg vs GT + flip counts vs the previous gate. Typed classifications
{FIRST_GATE, COUPLED_DESCENT, FLAT, A1_REALIZATION_GAP_ALARM}; 2 consecutive alarms ⇒
`a1_stage_exit_refuse` (stop, REROUTE — never scale a broken loop). Gate set: all pairs below n600; at
n600 the fd2 instrument geometry (block 447–450 + 32 rng(0) off-block).

## Design decisions (each with reason)

1. **MLX gradient / CPU-torch authority split** — the measured canonical split (104× scorer speedup;
   MLX NEVER a score). The A1 gate + full confirms run the frozen CPU-torch scorer only.
2. **Compose the witness loss via `render_fn`** rather than re-implement seg forms: inherits the
   MEASURED form arsenal (ce/tau_softplus/l7/unify_tau/margin_hinge + margin weighting) op-for-op; my
   surface area is the renderer + gate + ledger only.
3. **Zero-init token fields + shared_base temporal mode** (SYMMETRY force): counted token capacity is
   loss-driven only — ker(A) directions receive no gradient and stay at the zero lattice point, so no
   counted bytes land on the gauge orbit by construction. `independent` mode kept as the A/B control.
4. **D=12 EXCLUDED from the SPEC S1.2 race set** — 512/12 is a non-integer lattice; raced set {8,16}
   (deviation from SPEC, reason recorded here).
5. **EMA decay DERIVED from run geometry** (`ema_decay_run_geometry_v1`, `decay_from_warmup_fraction`
   φ=0.5) — never the borrowed 0.997. T1 n24: d=0.9778; clamped [0.9, 0.9995] for tiny smoke windows.
6. **Event-driven schedule** (never a PR95 stage skeleton): CE→tau_softplus at a measured knee event,
   with the F2 midpoint fallback so a non-firing event never strands the stage. A1 smooth-loss basis
   REBASES across a form switch (review pass 2 catch: the form change rescales the smooth loss and
   would otherwise fire a FALSE realization-gap alarm).
7. **Byte ledger = real compressor on real quantized payloads** (zlib-9, temporal-delta tokens,
   int8+fp16-scale weights, packbits masks) — labeled **COUNTED-ESTIMATE** until the E4/WS1 exporter
   grammar wires in at T3 (named boundary, not a claim of archive bytes).
8. **Resume re-anchors Adam moments fresh** (#517/#518 warm-start re-anchor): bounded windows restart
   moment estimation at the resume geometry; the β₂-derived window law binds the LONG burn (MAIN).

## The topology claim (steer #3 insight, receipt-verified; DERIVED, stated explicitly)

On the smooth-INR vehicle, island/component BIRTH is a saddle-node bifurcation with MEASURED HYSTERESIS
(`island_birth_saddle_node_hysteresis_measurement_20260715.md`, verified in-tree): gradient flow cannot
nucleate a component from zero — the pre-fold gradient vanishes — which is why islands-unborn plateaued
and #323's dilation/homotopy ladder existed. **On the token-grid + renderer vehicle, component birth is a
DISCRETE token change — no bifurcation, no hysteresis, no homotopy barrier. Topology change is
combinatorial, not dynamical.** This converges with the realization-wall physics: token lattice states
are HARD states, the native currency where v19-style realized acceptance operates (and where eu1's
discrete-search class lives). One coherent story: discrete birth × hard acceptance × token grid.

MEASURED on real GT (T0.2 micro-smoke: init erases ALL 134 GT Lane components on the 4-frame gate; plus a
post-hoc topology readout of the T1-plain final EMA checkpoint, which also REPRODUCED the recorded final
gate d_seg 0.02601 bit-for-bit from disk — checkpoint custody proven):

| class | GT Betti-0 (24 gate frames) | realized Betti-0 | GT components ERASED | smallest surviving (px) |
|---|---:|---:|---:|---:|
| Road | 26 | 52 | **0** | 1 |
| **Lane** | **756** | 68 | **742** | 11 |
| Undrivable | 25 | 79 | 0 | 1 |
| Movable | 77 | 69 | 51 | 46 |
| MyCar | 24 | 25 | 0 | 50,039 |

**The honest sharpening:** the token grid dissolves the topology barrier STRUCTURALLY (a token change CAN
birth a component with no hysteresis), but at T1 capacity/epochs the loss has not yet DRIVEN those token
changes for the Lane dashes — at d_seg 0.026 the bulk classes are component-complete while **742/756 Lane
dash components remain unborn** (each ~10–100 px; small-area gradient). The 1/persistence erasure law
holds on this vehicle too, and it names exactly which pool the next levers must draw from (the Lane-pool
race — class_weight_lane, clDice RE-RACE, margin forms — COMPETING per the pools law, never stacked).
DMTz separatrix edit-sidecar stays DOMINATED (#372) — not resurrected; its Morse-Smale reading survives
as the interpretation of the token grid.

## A2 — the G1-LOTTO rule-118 selector accounting (ADJUDICATED, per eu1's flag)

Per eu1 P3.0 (quotient/fiber factorization): if a decoder-visible architecture, seed, width/depth, mask
density, or program choice is selected using this video, its complete reproducible selector/config
belongs in counted `c`; changing free code to embody that selection is the hide-data-in-code fake.

**Adjudication (implemented in `selector_ledger_blob`, tested):**
- **COUNTED (lotto arm):** supermask bits (packbits+zlib) · per-out-channel modulations γ + biases
  (fp16) · token stream · the **selector/config ledger** {arch id, D, c, width, quant levels, STE mode,
  dither seed, temporal mode, **lotto PRNG seed**, mask density} as canonical JSON bytes.
- **FREE (generic):** the PRNG *expansion* of the fixed signed-constant conv bank from the counted seed;
  the renderer forward code; the token-grid interpreter; the dither-field expansion from the counted
  dither seed. All deterministic generic algorithms per rule-118.
- **NOT counted (reproducibility receipt):** encoder-only training hyperparameters (lr, epochs, form
  schedule) — runtime never consumes them.
- **Sharpening beyond eu1's minimum:** the selector ledger is counted for **BOTH** variants (the plain
  arm's D/c/width/STE/temporal-mode selections are equally video-selected). ~174 B at T1 — cheap, and
  the accounting is now symmetric so the race is fair at the ledger line too.

## op1 CONSUMPTION (openpilot/geometry review, merged to MAIN post-branch; verified from `.omx/research/ddm_op1_openpilot_physics_geometry_review_20260728.md` this session)

Four binding items, each adjudicated include-or-exclude-with-reason (op1's numbers verified against its
committed text: #609-v2 Road 39.0226 / Lane 47.1192 px p50 + hood D0 PASS; v_h=174 two-horizon-roles;
lane-carrier floor 0.002144; 98.806% image-stationary flip mass; d(v)=488.3/(v−192); CLADE ~39% SPADE
param overhead; the four P3 candidates with falsifiers):

1. **Token grid STAYS IN THE IMAGE PLANE — CONFIRMED-EXCLUDED BEV.** tr1's grid was already image-plane;
   op1's #609-v2 receipt makes that RECEIPT-BACKED, not just default: BEV is NON-static in the exact G1
   chart (probe KILL, exact-chart scope), d_seg is pixel-uniform so the image lattice is already the
   perspective-optimal allocation, and Lane is sub-pixel beyond ~60 m. NO BEV variant enters the D race.
   ξ-advection guidance survives ONLY as the SE(3)/terminal-pose path (untouched here) — the shared_base
   token mode remains identity-ξ IN THE IMAGE CHART, now doubly supported by the 98.806% image-stationary
   flip-mass row (the image chart is where the partition is approximately static).
2. **Settled geometry custody adopted for design use (never re-derived):** scorer-plane fx=400.27 /
   fy=399.82 / c=(256,192), cam 1.22 m; v_h=174 MEASURED-optimal lane-IPM horizon + 192 zero-pitch pose
   horizon (two-horizon-roles law — never forced to one); smooth-curve lane-carrier floor 0.002144
   (openpilot polys = positional PRIOR recovering ~64% of lane d_seg, never the carrier — tr1's renderer
   must learn the ragged ±1 px lane boundary, exactly SPEC G1's job).
3. **FREE openpilot roles registered:** decode-side CLADE-ICPE geometric conditioning features
   ((v−174), d(v)=488.3/(v−192), dist-to-boundary) cost ZERO counted bytes (rule-118 generic expansion) —
   registered as the `renderer_conditioning` raced lever (DUTY_TO_MEASURE), NOT hot-wired into the
   in-flight T2 race (mid-window feature changes would confound the A2 arms). Compress-time
   supercombo/poly init is $0-CPU confirmed — a warm-start candidate for the long burn, not T2.
4. **P3 raceable candidates adjudicated:**
   - **CLADE+geo-ICPE vs mini-SPADE** — REGISTERED raced lever (equal counted bytes; falsifier: transfer
     dead if mini-SPADE wins >10%). tr1's T0 conv ladder is currently NEITHER (plain conv on token
     inputs); the conditioning-block race is the S1.3 refinement round.
   - **Row-anisotropic D foveation (NO BEV)** — the $0 adoption gate was RUN THIS SESSION from the
     gt_n600 margins memmap: **72.1% of flip-prone mass in rows 160–240** (stable across margin
     thresholds 0.05/0.1/0.25; best 81-row band 166–246 @ 72.7%) ≥ the pre-registered 50% criterion ⇒
     **GATE PASSED — enters the S1.2 grid race as one variant lane** (receipt
     `op1_row_foveation_gate.json` on SSD). Raced, not unconditionally adopted.
   - **Boundary-gated token code width** — registered with its ≥15% token-stream-saving falsifier
     ($0 H(cell|neighbors) gate owed at the token-coder design round; feeds G4).
   - **OASIS per-pixel class balancing** — folded into the LANE-POOL race entry: `class_weight_lane`
     (already a wired lever) IS this family's simplest member; OASIS inverse-frequency balancing races
     IN THE SAME POOL as clDice/σ_cc′/persistence/sn1 sided weights — pools law, never stacked.

## FORCES / DYNAMICS / INTERACTIONS — the 15-item include-or-exclude table (operator completeness directive 2026-07-28; silent omission = the orphan bug)

| # | item | verdict at T0/T1 | reason |
|---|---|---|---|
| 1 | #360 four in-trunk forces (screw-consistency · MarginBandSatisficing #459 · tie-locus · R-phase) | **EXCLUDE from T1, registered DUTY_TO_MEASURE** | witness-vehicle-derived; adding 4 forces before the base-loop A/B lands would confound the A2 race. MarginBandSatisficing is the first T2+ candidate (min-S-over-solution-SET: stop over-deepening margins). |
| 2 | #382 per-class-pair σ_cc′ surface tension | **N/A-BY-CONSTRUCTION, registered** | tr1's loss has NO scalar length/MCF term, so the Γ-limit Lane-erasure mechanism is absent; σ_cc′ binds only if a curvature/length regularizer is added. Lane preservation rides the data term (`--class-weight-lane`, sn1 lever). |
| 3 | Triggers-forces P0 (F2/F4/F6) | **F2 WIRED** (midpoint event-fallback for the knee); F4/F6 N/A (no crest-fired lever, no hosc β, no in-trunk w_pose — pose terminal) | |
| 4 | Eikonal + DE-derived adaptive-ε (#318/#320) | **N/A** — no SDF/eikonal field in tr1 (direct RGB paint); the ep110 CFL re-entry class cannot occur | |
| 5 | #318 config-as-coupled-DE (lr/ε from von Neumann) | **PARTIAL: EMA derived; lr PROVISIONAL** | lr=2e-3 is a raced default, labeled provisional; the DE derivation is owed before the LONG burn (T3 ticket item). |
| 6 | Event-driven schedule (#686/#688) | **WIRED** — knee event + F2 fallback + continuation semantics on resume (events re-derive at resume geometry) | |
| 7 | Pathology alarms (#85 EMA lag / #304 freeze / critical slowing / #216 saddle / #475 grokking) | **liveness `frozen_epoch` + `gnorm_hijack` WIRED**; **#85 EMA shadow-lag CAUGHT LIVE at the first n600 window and structurally fixed (T0.3: gate basis from the law's warmup boundary W=2/(1−d))**; slowing/saddle/grokking = telemetry-visible via ep_loss + gate curves, adjudicated at check-ins (no auto-kill — bounded windows already cap loss) | |
| 8 | #518 resume/warm-start geometry | **PARTIAL** — Adam re-anchored fresh + events re-derived (conservative re-anchor); β₂-derived window lengths + LR-rewarmup owed at the T3 long-burn ticket | |
| 9 | #685 update-RMS-matched race fairness | **MEASURED-not-assumed**: per-gate `param_delta_rms` telemetry both arms; Adam's per-param normalization is the equalizer hypothesis; enforcement lever queued if arms diverge >2× | |
| 10 | POOLS LAW (same-pool levers COMPETE) | **HONORED** — no additive savings claims anywhere; the (D,c,levels,width) race IS the KKT waterfill instrument (Pareto rows, never summed); ledger streams are physically distinct sections, not same-pool levers | |
| 11 | seg↔pose collateral (cb1 +22.7) | **WATCHED-NOT-PRICED, deferred to composition** — the seg trunk renders frame_1 only (no pair exists to pose-score); collateral is absorbed by the TERMINAL pose solve on frozen composed frames (#383); another reason pose stays terminal | |
| 12 | Loss weights at stage boundaries only (#312) | **HONORED** — w_seg fixed; form switches only at the knee/fallback event (a stage boundary); no per-step re-weighting anywhere | |
| 13 | Island gradient starvation under composition (#300) | **N/A at T0** — whole-frame render, no compose mask; becomes live if a bulk-compose or control-token re-solve stage is added (flagged for that design) | |
| 14 | Curriculum synergy MEASURED (#430) | **OWED AT T3** — T1/T2 race single levers; the sealed long-burn ticket must carry the synergy A/B plan, not assume additivity | |
| 15 | Term-domination + gnorm-hijack alarms | **gnorm-hijack WIRED**; term-domination N/A-WITH-REASON — the trunk loss is single-term (seg only) by design until #360 forces enter; the alarm lands with the first multi-term config | |

## T1 — bounded smoke race (n24 × 60 ep, sealed DSL tickets; MEASURED)

Sealed tickets: plain `a7cbfb73…` · lotto `0ba53042…` (in each run dir). Detached via
`tools/launch_detached_process.py` (reaper-kill class honored). Kill criteria pre-registered:
nonfinite loss; 2 consecutive A1 alarms ⇒ refuse; flat realized telemetry ⇒ never scale.

**Plain arm (COMPLETE, 175 s wall):** knee event ep7 (ce→tau_softplus); ONE transient A1 alarm at ep19
(smooth −36% while realized rose 0.253→0.282 — the instrument fired LOUD, recovered next gate, no
refuse); zero confound alarms. Gate trajectory (realized d_seg, EMA shadow, frozen CPU scorer, all 24
pairs):

| ep | 4 | 9 | 14 | 19 | 24 | 29 | 34 | 39 | 44 | 49 | 54 | 59 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d_seg | .388 | .298 | .253 | .282⚠ | .212 | .107 | .0594 | .0464 | .0386 | .0330 | .0291 | **.0260** |

Every gate ep24→ep59 = **COUPLED_DESCENT** (smooth loss AND realized flips improving together — the
anti-fd2 signature this build exists to produce). Realized flips vs prev gate fell 268k→34k as the
partition locked in. Bytes at ep59 (n24, COUNTED-ESTIMATE): tokens 21,022 + renderer int8 20,027 +
selector 174 = 41,223 B.

**Lotto arm (COMPLETE, 171 s wall):** ZERO A1 alarms; every gate ep9→ep59 COUPLED_DESCENT; realized
d_seg 0.506 → **0.02767** at ep59; flips vs prev gate 3.04M → 36k. Renderer bytes **3,284** (mask+mods)
vs plain's 20,027 int8 — **6.1× fewer renderer bytes at a near-identical trajectory endpoint**
(0.0277 vs 0.0260). Total counted ep59: lotto 23,731 B vs plain 41,223 B (n24; tokens similar ~20–21 KB).

**T1 verdict (pre-registered criteria): BOTH ARMS PASS** — realized gate d_seg strictly decreasing
across ≥2 consecutive gates AND smooth loss descending (the coupled-descent anti-fd2 signature), no
kill criterion hit, zero confound alarms. **The one T1 alarm (plain ep19) is the A1 instrument proving
itself live** — smooth −36% while realized rose; loud, recovered, no refuse.

**Honest T1 confound (recorded, not silently absorbed):** the two arms ran skewed code versions (plain
launched pre-T0.1 → knee event fired ep7 off a transient CE rise; lotto loaded post-T0.1 code → F2
midpoint fallback fired ep30), so the arms switched seg forms at different epochs. T1 arm ORDERING is
therefore NOT load-bearing (verdict_scope: INSTANCE, single seed, n24). T2 runs both arms on identical
code `58835f6f29` with identical sealed schedules — the race adjudication surface is T2.

## T2 — bounded n600 windows (identical code `17166ee9c4`; sealed tickets d6eeefd7 / 0a6eba28)

Config: n600 × 40 ep × 50-min cap per arm; batch 8; D=16 c=4 L=16 round/shared_base; gates every 5 on
the fd2 36-pair geometry (block 447–450 + 32 rng(0)); full-confirm chunked (≤120 rule; chunk 32); memory
preflight MEASURED at the real config (peak RSS 12.8 GiB via /usr/bin/time -l — SAFE vs 89.6 GiB bound;
51 s/epoch). First launch aborted at ep9 on the #85 gate-basis confound (T0.3 above, custody preserved);
relaunch = the adjudication surface.

**Pre-registered adjudication (written BEFORE results):** primary = full-confirm realized d_seg (n600)
at window end per arm + total counted bytes at the final gate. An arm wins DECISIVELY only if it
Pareto-dominates (≤ d_seg AND ≤ bytes); otherwise BOTH-CONTINUE into the (D,c,levels) waterfill.
Trajectory reported vs the G1 rails {1e-3, 5e-4, 3e-4} honestly (a 40-epoch bounded window is NOT
expected to reach them — the rails bind the LONG burn). The gate-36 vs full-600 row is reported as the
subset/full disagreement observable (the fd2-owed class). Single-seed; INSTANCE scope; no noise floor.

### T2 MEASURED (both arms COMPLETE, 40/40 epochs, zero confound alarms, zero A1 refusals; `[macOS-CPU/MLX advisory]`)

**Gate trajectories (realized d_seg, fd2 36-pair geometry; basis live until warmup W=1500 steps ≈ ep20,
EMA shadow after; form ce→tau at the F2 midpoint ep20; every comparable gate COUPLED_DESCENT):**

| ep | 4 | 9 | 14 | 19 | 24† | 29 | 34 | 39 |
|---|---|---|---|---|---|---|---|---|
| plain | .03371 | .02278 | .01897 | .01712 | .02206 | .01763 | .01574 | **.01463** |
| lotto | .03343 | .02327 | .01940 | .01881 | .01908 | .01681 | .01564 | **.01439** |

† double rebase (form switch + live→shadow basis switch) — correctly FIRST_GATE, no false alarm; the
first launch died at exactly this point on the pre-T0.3 instrument.

**Full-n600 realized confirms (EMA shadow, chunked 32, ~163 s each) + counted bytes at final gate:**

| arm | full n600 d_seg | max pair | tokens | renderer | selector | total | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain | 0.014088 | 0.0298 | 529,538 | 20,214 | 175 | 549,927 B | 1,531 s |
| **lotto** | **0.013833** | 0.0255 | 531,097 | **3,284** | 216 | **534,597 B** | 1,477 s |

**Subset/full disagreement row (the fd2-owed observable):** the 36-pair gate reads HIGH vs full-600 in
both arms (plain +3.8%: .01463 vs .01409; lotto +4.1%: .01439 vs .01383) — small, sign-stable, consistent
⇒ a valid cheap inner instrument at this operating point (rank-only, never a kill).

**#685 race fairness MEASURED:** final-gate param_delta_rms plain 0.00994 vs lotto 0.00963 (3.2% apart;
no gate >2×) — Adam's per-param normalization equalized update magnitudes; enforcement stays queued.

**Lane topology channel (realized lane Betti-0 / GT-components-erased of 985, 36 gate frames):**
plain 23→264 / 979→**906**; lotto 9→164 / 983→**916**. Both arms nucleate LATE and are ACCELERATING at
window end (plain +91 realized components over the last 5 epochs; lotto +62). Bulk classes
component-complete both arms; Movable erased 44/77 both. **vs the G1 rails: 0.0138–0.0141 ≫ 1e-3** — the
bounded 26-min window does NOT approach the rails (pre-registered expectation); trajectories still
descending at window end (no plateau) — the window is time-bound, not converged.

### T2 ADJUDICATION (the pre-registered rule applied AS WRITTEN — no post-hoc edits)

**WINNER ARM = G1-LOTTO: Pareto-dominates on both pre-registered axes.** Arithmetic: full-confirm d_seg
0.013833 ≤ 0.014088 (−1.8% rel) AND total counted 534,597 ≤ 549,927 B (−2.8%). The structural (not
marginal) byte fact: the LOTTO renderer stream is **3,284 vs 20,214 B = 6.2×** — at the G3/G4 target
geometry (tokens compressed toward ~117 KB, renderer budget ≤64 KB) the renderer share grows and the 6×
advantage compounds; at the current config both totals are token-dominated.

**Named caveat (facet honesty — recorded, NOT promoted to a verdict-changer):** plain LEADS the
Lane-nucleation channel (realized lane Betti-0 264 vs 164; erased 906 vs 916), and Lane erasure is THE
terminal residual (742/756 unborn at T1; 1/persistence law). The pre-registered rule did not include the
topology channel; promoting it post-hoc would be the exact discipline violation pre-registration
prevents. Handling: (a) LOTTO seals as the long-burn arm; (b) the **Lane-pool lever race fires FIRST**
in the burn plan (`class_weight_lane` sweep — already wired — with OASIS inverse-frequency as a raced
setting; then clDice-RE-RACE / persistence / sn1 sided / σ_cc′ per the pools law, COMPETING never
stacked) and lane Betti-0 becomes a burn stage-exit facet; (c) the plain final checkpoint
(`t2_n600_plain/checkpoints/stage_seg_trunk_tau_final.npz`) is RETAINED as a live fallback — NOT killed
(margins 1.8%/2.8%, single seed, no noise floor; verdict_scope INSTANCE).

**Long-burn config decision (folding the adjudicated levers):** T3 ticket = LOTTO at the raced base
config (D16/c4/L16/w24/round/shared_base), compiled from EXISTING flags only (never-invent-flags). The
GATE-PASSED foveated banded-D grid (72.1% flip mass rows 160–240), boundary-gated code width, and
CLADE+geo-ICPE conditioning are BUILD-THEN-RACE items in DUTY_TO_MEASURE — they enter via short race
windows against the burn baseline, never as unmeasured adoptions inside a sealed ticket.

### T3 SEAL STATUS

**SEALED — READY_TO_FIRE_UNDER_STANDING_GO (the LONG burn fires from MAIN only).** Ticket:
`.omx/research/configs/ddm_tb1_t3_long_burn_lotto_20260728.json` (+ SSD copy
`t3_long_burn_lotto_sealed_ticket.json`) — variant **lotto**, n600 × 400 ep / 480-min resumable windows,
gate_every 10, full-confirm; `ticket_hash 007d8eacf402c4fe…`, `sealed_sha256 99b13a53fa412cbe…`, code
`17166ee9c4`; the adjudication arithmetic + caveat + plain-fallback custody are embedded in the ticket.
**Named owed items before/alongside the burn:** governed-launcher adaptation for the tr1 trainer; E4/WS1
exporter grammar wiring + numpy deploy-parity port (byte-close); the Lane-pool race windows; the
(D,c,levels) token-rate waterfill — the token stream at ~530 KB is the BINDING rate axis (G4 ≤130 KB;
the current temporal-delta zlib at L16 has not yet absorbed the shared structure the plateau law prices
at ~117 KB; shared_base deltas are dense while training keeps perturbing them — the entropy-coded
learned-prior coder + boundary-gated c are the named next levers on this axis).

## Canonical-vs-unique decision per layer

- **ADOPT canonical:** `make_loss_fn` + seg forms (measured arsenal; render_fn hook = designed extension
  point) · `_apply_R` contest-exact R · CPU verdict helpers (bit-exact batched) · GT memmap access ·
  serializer/review-gate/detached-launcher ops · `ema_decay_run_geometry_v1` LawRef.
- **UNIQUE (this substrate):** the token-grid + conv renderer module (both variants; MLX-native, opaque
  `_FixedBank` so the LOTTO bank is non-trainable + regenerable) · the A1 realized-flip gate + typed
  adjudication · the counted-byte ledger + symmetric selector accounting · the tr1 DSL program module
  (`spec_tr1_renderer_20260728.py`) with AST-based never-invent-flags validation (the 374 KB
  curriculum_dsl compiles the LEVELSET trainer; forking a thin tr1-specific program module is the
  principled mismatch case — different trainer, different flag algebra; the canonical `Lever` dataclass
  is reused as the composition primitive).
- **FORK reason recorded:** governed launcher `tools/launch_witness_run.py` hardcodes `_TRAINER` to the
  levelset entry point — T2 windows run detached with sealed tickets + measured-RSS receipts instead;
  adapting the governed launcher is a named T3 owed item before any LONG burn (which fires from MAIN).

## Observability surface

Per-layer: telemetry.jsonl rows for every epoch (ep_loss, seg_form, gnorm, liveness) + every gate
(realized d_seg per-pair max/mean, flips-vs-prev, A1 classification, byte ledger, param_delta_rms) +
typed confound_alarm rows. Decomposable: ledger per stream; gate vs smooth loss separable. Diff-able:
sealed ticket hash + config_hash in every receipt; checkpoints stage-encoded. Queryable: JSONL + window
receipts (schema `ddm_tb1_tr1_window_receipt.v1`). Cite-able: commit shas + ticket hashes + SSD paths.
Counterfactual: `--token-temporal-mode independent`, `--token-ste round|dither`, `--seg-form-start`,
variant arms — every lever a flag held by the DSL.

## Wire-in / hooks (Catalog #125)

Sensitivity-map: N/A (no per-tensor importance rows yet — first n600 Pareto rows will seed it).
Pareto: the T1/T2 gate rows are typed (d_seg, bytes) planner-consumable points. Bit-allocator: N/A until
the (D,c) waterfill has ≥2 measured n600 points. Cathedral autopilot: N/A (no paid dispatch).
Continual-learning: this memo + DAG FEED + receipts. Probe-disambiguator: the A1 gate IS the
disambiguator (coupled-descent vs realization-gap per gate).

## Honest boundaries

- **Nothing here is a score.** All realized d_seg rows are `[macOS-CPU/MLX advisory]` on the frozen
  scorer; no byte-closed archive exists yet; the E4/WS1 exporter + R6 chain are T3+ items. Pointer
  UNMOVED at 0.1910828242.
- The byte ledger is a COUNTED-ESTIMATE (real compressor, real quantized payloads, but zlib not the
  final entropy coder and no archive container overhead).
- T1 is n24 — NOT n600 evidence; it is the pre-registered smoke whose only claims are (a) the loop is
  live end-to-end on real GT, (b) realized-flip telemetry is coupled to smooth descent (anti-fd2), and
  (c) both variants run under matched accounting. n600 rows are the T2 section's.
- The A1 gate renders on the MLX **CPU** stream (fp32-class), not a numpy deploy port — the witness
  port-fidelity lesson says MLX-CPU vs numpy is a 3–18 px class drift; the numpy-portable deploy forward
  + byte-close parity is a named T3 owed item.
- The T1 lotto/plain comparison is single-seed; no noise floor measured — trajectory ordering is
  advisory, not a kill (verdict_scope: INSTANCE).
- lr and several training constants are PROVISIONAL raced defaults (labeled), not DE-derived (#318 owed).
