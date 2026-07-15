# FRESH-EYES RECURSIVE FRACTAL ADVERSARIAL REVIEW — v9-max composed stack (#507)

Date: 2026-07-15 · Reviewer: fresheyes-fractal-r2 (Fable, from-zero restart; predecessor died with nothing written)
Operator basis (verbatim): "Now would be a good time for fresh eyes for cursive adversarial review and
deep math and geometry and cargoculting and optimization pass al dimensions recursive fractal"
Research-only: `true` · Pointer: submittable 0.19108 UNMOVED (this review is MEANS).

Live-fire constraint honored: chain pid 91660 (`chain_507_r6.sh`) + any running trainers are READ-ONLY.
No config mutation of the live chain is performed by this review under any finding severity.

## CRITICAL findings (launch-invalidating)

### F10 [B][CRITICAL][correctness+measured-runnability] — annulus_plateau sensor is STRUCTURALLY UNSATISFIABLE at the live config's verdict cadence: 2 of the 5 "dissolved" transitions (seg-chroma-boundary, temporal-screw) can NEVER event-fire; both are guaranteed backstop-cap fires + spurious S5 alarms
DERIVATION (all values read from the LIVE compiled artifact
`experiments/results/levelset_n600_witness_20260715T180544Z/dry_start/launch.sh`):
- `--eval-every 25` → the annulus series gets ONE point per verdict epoch
  (trainer:9069-9080 — appended only inside the verdict-row block, 25-epoch spacing);
- `--annulus-plateau-dwell-windows 4` + `--annulus-plateau-min-epochs 150` → the detector
  (`event_wirings.annulus_plateau_event:427-443`) takes the trailing 4 points, span
  = (4−1)×25 = **75 epochs**, and requires `span >= 150` for `dwell_ok`;
- 75 < 150 ⇒ `dwell_ok` is False at EVERY epoch of the entire 3000-epoch run ⇒
  `annulus_plateau_event.fired ≡ False` ⇒ the chroma gate (trainer:11778-11791) and the
  temporal-screw gate (trainer:11685-11698) can never fire on the sensor.
CONSEQUENCE: both transitions fire on their fixed backstop caps (450) — i.e. exactly the epoch-scripted
incumbent the SPEC §2 claims is dissolved ("ALL transitions event-fired ... this landing closed the last
epoch-scripted transition") — AND each emits a `cap_fired_before_event` S5 row whose registered meaning
("the wired sensor never triggered ... re-calibrate the sensor", falsification-relevant) is SPURIOUSLY
triggered by a parameter inconsistency, not by run physics. The would-fire calibration signal for the
annulus sensor is likewise all-False for the whole run. The 5-sensor event-continuation purpose of the
10.2-day run is structurally unattainable for 2/5 transitions.
SATISFIABILITY LAW (the fix's derivation, no new constant): the detector needs
`(dwell_windows − 1) × eval_every >= min_epochs` → at eval_every 25 either
`--annulus-plateau-dwell-windows 7` (span 150, keeps min_epochs=150) or `--annulus-plateau-min-epochs 75`
(keeps 4 windows). The compile/schedule-provenance gate checks declaration consistency (12/12 OK) but has
NO sensor-satisfiability check — add `(W−1)·eval_every ≥ min_epochs` as a compile-time refusal.
CROSS-CHECK of the other 3 sensors at cadence 25 (all satisfiable): lane_nucleus fired by event at ep33
in run 20260715T095030Z (evidence in its run.log); powerlaw_meat needs 8 verdict points = 200 epochs
< cap 726; label_floor needs 3 same-stage rows = 75 epochs < cap 726. ONLY annulus_plateau is dead.
STATUS: live chain pid 91660 is still in drain-wait / dry-start phase (real launch NOT yet fired) —
there is time for an operator decision (stop chain → one-flag recompile → refire, or accept cap-fires
for this run and fix for the curvelet arm). Per the review's hard rule NOTHING was touched:
no config, chain, or module edit was made. Routed to operator.

## Round/seal state

- Round 1 (all 5 axes × surfaces A–E): 10 findings — F1..F9 + CRITICAL F10.
- Round 2 (adversarial re-examination of round-1 findings + escape-hatch hunt): F10 CONFIRMED against
  the live compiled launch.sh + the V9 typed-provenance manifest (the two custody notes "four verdict
  points must hold the plateau" and "min 150 epochs" are individually reasoned but JOINTLY unsatisfiable
  at `--eval-every 25`; no config override exists); F5's cancellation mechanism re-verified at the
  trainer call site (normalize applied to the already-clipped tree, 12496-12500); F2's premise verified
  (subpix/screw/satisfice within-pair phase forces are ON in the c1 argv before T1). No round-1 finding
  reversed; no new finding beyond refinements.
- SEAL STATE: **NOT SEALED — clean-pass counter 0/3.** A CRITICAL (F10) is open against surface B and,
  per the review protocol, surface-B review stopped at the CRITICAL; sealing is moot until the operator
  disposition of F10 (stop-chain→one-flag-recompile→refire vs accept-cap-fires-and-fix-next-arm).
- Per the hard rule, ZERO edits were made to the live chain, its config, the trainer, or any
  witness_dsl/witness_control module (the chain re-imports those at dry-start + real-launch spawn time;
  any edit would mutate the bytes the chain executes). Every fix in this ledger is ROUTED.

## Verdict per surface (round-1+2)

- A (composed config + SPEC + boot suite): 2 findings (F1 receipt name-only binding; F9 INFO —
  boot suite COST-JUSTIFIED, not recompute-poison, one overlap optimization routed). No CRITICAL.
- B (curriculum dissolution): **CRITICAL F10** (annulus_plateau structurally dead at cadence 25 →
  chroma+screw are cap-fires by construction) + F2 (label_floor lower-band masking) + F3 (band-HI
  rationale drift). EventBackstopGate core, resume protocol, label_floor wiring, tau octave ladder
  (derived rung count, deliberate flat cap with reasoned refusal to invent a rung-scaling constant):
  CLEAN.
- C (burn-down levers): F4 (VerdictParallelWorkers: untracked OFF in SPEC §3 + stale consumer-docstring
  scope) + F5 (AdaptiveGradClip carries the REFUTED lr/12 mechanism; autoclip+per-param composes inert
  with no refusal). #480 v2 disjoint spans: CLEAN (three spans measured from real perf_counter reads
  around existing statements; tail truncates at the telemetry block — small unattributed residual,
  conservative direction). verdict-parallel-workers LAW itself: CLEAN and honestly scoped.
- D (flicker-floor re-scope): SOUND (F6 is a docstring overstatement vs the registry's own careful
  first-order labeling; the re-scope's verdict-scope ladder application, existence proofs with L18
  caveats, forbidden-reading and licensed-lever fields are all correct).
- E (CUDA port): F7 (0.9997 inherited-uncited + INSTANCE-scope toy-loss probe licensing all compiled
  regions) + F8 (CudaGraphForwardBackward capture step returns undefined outputs / skips a real step;
  currently consumer-less and its test exercises only the eager fallback — must fix before the first
  CUDA-graph consumer). compile_mode override (6dfb43c96b): CLEAN (measured r5 root-cause, honest
  time-box reasoning; "reduce-overhead"→{} mapping is an unreachable-today nit folded into F8).

## Surface index

- A: composed c1_optimal_form config + SPEC ledger + dry-start boot-suite cost-justification
- B: curriculum dissolution (5 EventBackstopGates, tau octave ladder, co-annealed beta/LR, cap_fired alarm)
- C: burn-down levers (#480 v2 spans, AutoClip/GradNormalizeNone C0-confound custody, verdict-parallel-workers law scope caveat)
- D: flicker-floor re-scope derivation (0.005318 binds only smoother-than-GT formulations)
- E: CUDA port (cuda_v9_throughput.py, cuda_levelset_training.py compile_mode, 0.9997 cosine_phi threshold provenance)

## Findings ledger (incremental)

(rows appended below as found; format: F<N> [surface][severity][axis] — claim · evidence · route)

### F1 [A][MEDIUM][correctness+assumption] — dry-start GREEN receipt is bound to config NAME only, not config CONTENT
`spec_c1_optimal_form_20260715.py::_derive_launch_blockers` slot 2 clears `C1_COMPOSED_BENCH_NOT_MEASURED`
when ANY `experiments/results/*/dry_start_report.json` has `gate==full_config_dry_start AND config==<name>
AND green` — with no `dsl_compile_hash` / argv-hash binding and no freshness window. A GREEN receipt
measured on an OLDER compile of the same name (e.g. before a reviewed amendment to
`C1_OPTIMAL_FORM_EXPECTED_ADDITIONS`, or a changed lever value) would keep clearing the bench blocker
forever, so peak-RSS/sec-per-ep evidence can silently go stale relative to the config it vouches for.
Sister of memory `launch_readiness_gate_config_freshness_naive_launch_20260710` ("can the machine HOLD
it?" — freshness is the gate), applied to the RECEIPT surface instead of launch.sh.
NOT launch-invalidating for the live chain (receipt produced minutes before the real launch by the same
chain, config frozen in between). Route: record `dsl_compile_hash` (already computed by the launcher)
into `dry_start_report.json` and match on it (or argv sha) in `_derive_launch_blockers`; optionally a
max-age window. Design-route recommendation — no edit made (file feeds the LIVE chain; read-only rule).

### F2 [B][MEDIUM][deep-math+correctness] — label_floor sensor's floor-band LOWER edge can permanently mask the event for a run that pierces the floor early
`label_floor_detector.label_floor_reached` fires iff `FLOOR_BAND_LO 0.00496 <= d_seg <= FLOOR_BAND_HI
0.00700` AND flat AND label-smooth stage. But the c1 config runs within-pair phase-tail forces (subpix
0.3, temporal-screw @ annulus_plateau, satisfice 0.2, area-Lagrange) BEFORE T1 — exactly the forces the
2026-07-15 flicker-floor re-scope says can carry a witness BELOW the label floor. Two masked paths:
(a) d_seg descends BELOW 0.00496 while still in a label-smooth stage → `within=False` forever →
LABEL_FLOOR event NEVER fires → T1 engages only at the 726 backstop cap AND a spurious
`cap_fired_before_event` S5 "re-calibrate the sensor" alarm fires on a run that actually OUTPERFORMED;
(b) a fast descent CROSSES the band in <3 same-stage verdict rows (MIN_FLOOR_WINDOW) or before
flatness registers → same masking. Semantically the hand-off condition is "label descent exhausted"
= `d_seg <= HI AND (flat OR d_seg < LO)`; below-band is STRONGER evidence of exhaustion, not absence.
Behavior is SAFE (backstop guarantees T1 by 726) but late + misdiagnosing (S5 row semantics inverted
for the below-band case). Route: add a distinct below-band classification (e.g.
LABEL_FLOOR_PIERCED → fired=True) or extend `within` to `d_seg <= hi`; ledger-routed, NOT edited live
(detector is imported by the running chain's trainer at boot; the live run must keep its sealed bytes).

### F3 [B][LOW][value-provenance] — FLOOR_BAND_HI rationale/value mismatch
`label_floor_detector.py:58-60`: comment says upper edge catches a run "still 10-20% above" the oracle;
actual HI=0.00700 is 31.6% above 0.005318. Either the value or the rationale is stale. LOW (band width
is a sensitivity knob with backstop protection); route: align comment or re-derive HI as oracle×1.2.

### F4 [C][MEDIUM][assumption+optimization] — VerdictParallelWorkers: measured wall lever OFF in c1 with NO row in the SPEC ON/OFF table; consumer docstring scope stale
(a) `verdict_parallel_workers_speedup_v1` (MEASURED 2026-07-15T18:42Z, w=8 → 5.686× scorer-forward,
values float-equal, sized 8 workers) says "compose VerdictParallelWorkers(8) into the next n600 launch"
— but the c1 config does not emit `--verdict-parallel-workers` AND SPEC §3 ("every OFF is TRACKED +
REASONED — no orphans") has NO row for it. Timing explains it (law landed ~40 min AFTER the c1 compile;
mutating the mid-chain config is forbidden), but the untracked OFF is exactly the
default-off-is-orphan class. Route: add the SPEC §3 row (OFF-for-this-run, reason=post-compile landing;
duty-queued for the curvelet arm / next launch). Amortized value ~12 s/ep of ~295 s/ep (~4%).
(b) `curriculum_dsl.VerdictParallelWorkers` docstring still says "Targets the measured 2555.7 s/verdict
C0 wall; expected ~/min(workers, cores_free, n_chunks)" — the landed law shows workers divide only the
~370 s scorer-forward SHARE of that wall (render/realized stages un-parallelized). The equation's honest
scope caveat did NOT survive into this consumer's docstring. Route: docstring correction (deferred until
the live chain has spawned its trainer — see the no-edit rule note at F5).

### F5 [C][MEDIUM-HIGH][assumption/TELEMETRY≠MECHANISM] — AdaptiveGradClip carries the REFUTED C0 lr/12 mechanism AND can be composed inert (autoclip masked by per-param normalize) with no refusal
Evidence chain: trainer line 12491-12500 applies `per_param_normalize_grads` to the ALREADY-clipped tree
(g_p ← g_p/(‖g_p‖+eps) per tensor) — a uniform clip scale c cancels exactly, so ANY norm clip (fixed /
per-group / autoclip) is a NO-OP on the applied update whenever `--grad-normalize per-param` (the live
v9 incumbent). `GradNormalizeNone` (curriculum_dsl.py:2678) states this correctly ("per-param masks ALL
norm clipping"; C0 saturation telemetry "real but INERT"; lr/12 reading REFUTED) — consistent with
memory `perparam_normalize_masks_all_norm_clipping_c0_confound_20260715` and SPEC §3. BUT:
(1) `curriculum_dsl.AdaptiveGradClip` (2612-2675) still motivates itself with the REFUTED mechanism
    ("the effective step is lr·0.5/‖g‖ ≈ lr/12", "the LR cosine no longer controls the descent clock",
    "cures the C0-measured saturation") — same for `witness_control/adaptive_grad_clip.py:4-10`
    ("largest epochs-to-target lever in the stack") and the `scientific_declaration` provenance string
    (2661-2663) that would be custodied into a config manifest as law provenance.
(2) The trainer's autoclip arming block (7879-7911) has NO refusal/warning for
    `--grad-clip-mode autoclip` + `--grad-normalize per-param`: that composition runs AutoClip's whole
    percentile machinery, emits per-epoch `grad_clip_autoclip` telemetry (thresholds, frac_clipped) —
    while every clip it applies is divided out downstream. A counted-but-inert lever (#417-shaped) whose
    telemetry invites exactly the TELEMETRY≠MECHANISM misreading the C0 confound taught. The armed row
    does not even record `grad_normalize`.
c1 is UNAFFECTED (does not touch either flag — SPEC §3 grad-clip row is correct). Route (post-chain):
(a) trainer startup refusal (or LOUD confound_alarm + manifest row) for autoclip+per-param;
(b) docstring/provenance-string corrections in both files, APPEND-ONLY style ("C0 saturation telemetry
    was real but INERT under per-param normalize; magnitude-law A/B owed" — matching GradNormalizeNone);
(c) record `grad_normalize` in the `grad_clip_autoclip_armed` row.
NO EDITS MADE NOW: the live chain (pid 91660) will spawn fresh trainer/DSL processes at dry-start-r6 +
real-launch time — ANY edit to the trainer or witness_dsl modules would change the bytes the chain
executes. All fixes in this review are therefore ledger-routed until the chain's real launch has spawned.

### F6 [D][LOW][deep-math] — detector docstring overstates the bound its own registry entry carefully scopes
`label_floor_detector.py:15-16` "CANNOT descend below the GT temporal-majority oracle floor d_seg =
0.005318" — but the registry's `domain_refined` row labels the derivation "first-order in spike density
— adjacent-spike overlap neglected", and the in-repo MEASURED CE floor 0.00496 (#205/L67, the band's own
lower edge) sits 6.7% BELOW 0.005318. The flat "CANNOT" is contradicted at the ~7% level by the module's
own FLOOR_BAND_LO. Honest phrasing: "cannot descend materially below (first-order majority-oracle bound;
measured convergence band 0.00496–0.0070 straddles it)". Surface-D verdict otherwise: the 2026-07-15
re-scope derivation is SOUND — verdict-scope ladder correctly applied (FORMULATION), forbidden-reading +
licensed-levers made queryable law fields, existence proofs carry the L18 ancestor caveat, and the
derivation status is honestly labeled DERIVED-first-order + MEASURED-constant.

### F7 [E][MEDIUM][unproven-constant/poison-1] — 0.9997 cosine_phi adoption threshold: inherited cross-surface constant, uncited at use sites, INSTANCE-scope probe
The threshold is TRACEABLE (CLAUDE.md deterministic-repro spine: "MLX/torch match [numpy-fp32] (parity
≥ 0.9997)") so it is NOT a bare invention — but at its three hardcoded use sites
(`cuda_v9_throughput.py:43`, `cuda_levelset_training.py:941` + adoption_rule string) there is no
citation/LawRef, and the ORIGINAL bar was defined for scorer forward parity vs the numpy-fp32 authority,
not for compiled-training-region gradient quality under AMP/TF32 reordering. Worse, the probe that
produces cosine_phi (`compile_identity_probe` called at `train_..._torch.py:1097-1108`) runs on
feats[:64] with a SYNTHETIC loss `rgb²+phi²` — an INSTANCE-scope receipt (one sub-batch, toy loss) that
then licenses compiling model+seg+pose+R+ALL real loss ops (`:1497-1511`). argmax_equal on one probe
batch does not bound per-pair drift across n600. Route: (a) cite the parity-law provenance at the use
sites + register the adoption rule as a LawRef; (b) stamp `verdict_scope: INSTANCE` into the probe
receipt; (c) before the first paid CUDA dispatch, extend the probe to the real loss_fn (it is available
at the call site) — $0 change, closes the toy-loss gap.

### F8 [E][MEDIUM (latent HIGH at first CUDA dispatch)][correctness] — CudaGraphForwardBackward capture step returns UNDEFINED outputs and silently skips one real training step
`cuda_v9_throughput.py:155-170`: on the first post-warmup call the graph is captured via
`with torch.cuda.graph(graph): self._static_outputs = self.fn(*self._static_inputs)` and then
`return self._static_outputs` WITHOUT a replay. Under CUDA stream capture, kernels are RECORDED, not
executed — so (a) the returned static outputs hold undefined/stale memory, and (b) the backward pass
this call was supposed to perform never ran (grads stale/zero) → the capture call is exactly the
"unrecorded dummy update" the class docstring (97-100) claims cannot occur. Fix: `graph.replay()`
immediately after capture (inputs were already copied into the static buffers), then return.
Mitigations: the class currently has NO trainer consumer (only its own test), and the test exercises
only the eager fallback (`_cuda_ready` False off-CUDA) — i.e. the claimed mechanism is untested
(NO-FAKE class-2 adjacency: the test would pass with the capture path deleted). MUST land before any
consumer wires it for the first authorized CUDA dispatch. Also LOW: `mode_option_equivalents` maps
"reduce-overhead"→{} (its documented equivalent IS cudagraphs); unreachable today because
`select_torch_execution_policy` never emits it — remove or map honestly.

### F9 [A][INFO][cost-justification verdict] — boot suite: COST-JUSTIFIED, not recompute-poison; one optimization routed
Per the review question on surface A: the r5 boot (~3400-4200 s) decomposes into consumers-named work,
not poison-taxonomy-3 recompute: (1) baseline v0 verdict (~2556 s, trainer:9810-9840) — consumed by s0 /
implied-score, the event-sensor histories (label_floor/powerlaw need the verdict stream), liveness +
apparatus-validity rows; runs ONCE. (2) `jacobian_basin_t0` (trainer:8343-8409) — the t=0 anchor of the
σ_min(J_ξ) series consumed by the `sigma_min_plateau` pose-finish sensor (its own setup refuses/warns
when pose-finish is sigma_min_plateau but basin telemetry is off). (3) head_offset_solver — advisory
flip_median arbiter at the EMA-verdict call site; consumer = the #386 realized-through-R delta rows (the
owed A/B instrument, per the constants manifest). Each reads trainer streams; none recomputes per-epoch
state at boot. OPTIMIZATION (routed, not owed): the v0 verdict's consumers are all later-epoch sensors —
it could run on the async-verdict thread overlapped with the first training epochs, cutting ~40 min of
serial boot; requires care only with the baseline-print ordering. r6 budgets (5400 s boot + 1800 s/ep)
correctly sized from the r5 measured receipt.
