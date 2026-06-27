# AGENTS

> # ⛔ SUPREME RULE — NO FAKE IMPLEMENTATIONS ⛔
> **This is the #1 non-negotiable, ABOVE EVERY OTHER RULE in this file. Highest possible emphasis.**
> An implementation that does not ACTUALLY perform — on the REAL inputs — the work its name claims is a
> FAKE and is FORBIDDEN. No exceptions. It does not matter how many tests pass, how clean the metadata
> is, how plausible the proxy looks, or how close to the frontier it sits. A **score**, a **"done"**, an
> **"original"**, a **"compiler"**, or a **"solver"** that the exact contest oracle (`upstream/evaluate.py`,
> 600-sample, authority tier) AND the actual code do not JOINTLY prove is a FAKE CLAIM. When in doubt:
> do LESS, but make it REAL — a smaller honest result beats a larger fake one every single time.
> The **eight forbidden classes**, the catch-and-fix cascade, and the cross-references are the canonical
> **`## NO FAKE IMPLEMENTATIONS`** section below. READ IT before writing, claiming, marking-done, or
> submitting ANYTHING. A reuse/name/mechanism claim the code does not honor is a bug, not a shortcut.

> # 🎯 THE GOAL — SUB-0.15 EXACT SCORE — NON-NEGOTIABLE, HIGHEST EMPHASIS 🎯
> **The mission, second only to NO FAKE. The goal is to LOWER THE EXACT CONTEST SCORE below 0.15.**
> Success has EXACTLY ONE definition: the canonical frontier pointer
> (`.omx/state/canonical_frontier_pointer.json`) records a lower exact-eval `archive.zip` score from
> `upstream/evaluate.py` (600-sample, contest-CPU AND/OR contest-CUDA on 1:1 hardware). Ladder:
> **T_3 = sub-0.15 (THE target, the default aim) · T_1 = sub-0.19 (floor of acceptable).** Above T_1 = failing.
> **What is NOT goal progress (the failure mode this extincts):** tools, harnesses, solvers, codecs,
> runtimes, Rust crates; floor derivations, research/design memos, paper inventions; located cruxes,
> honest negatives, DEFER verdicts, deferral ledgers; holding the existing frontier; a measurement
> merely running; advisory/MLX/proxy rows. These are MEANS. The END is a LOWER EXACT SCORE. A unit (or
> session) that ends with the exact pointer UNMOVED and S > T_1 has NOT achieved the goal — say so
> plainly (narrating means as ends is a mission-level NO-FAKE violation) and aim the next unit DIRECTLY
> at an exact-eval row that crosses the threshold. **Bias every decision toward the action most likely
> to LAND A LOWER EXACT SCORE soonest** — not the most rigorous analysis or the most interesting crux.
> When a path walls, PIVOT to the next path that can produce a lower exact score, not to more
> characterization of the wall. Build infrastructure ONLY in service of an imminent exact row. The
> Modal <$5 budget exists to BUY exact rows — spend it to measure real byte-closed candidates; do not
> hoard it while the score sits unmoved. Canonical body: the **`## THE GOAL — SUB-0.15`** section below.

## Local Disk, SSD Spill, Auto-Cleanup, And Provenance — NON-NEGOTIABLE, HIGHEST EMPHASIS

Local disk is for source, small manifests, and live metadata. Bulky rebuildable
work belongs on the connected SSD tier first, in this priority order unless the
operator explicitly overrides it: `/Volumes/VertigoDataTier/pact`, then
`/Volumes/APDataStore/pact`, then local disk only by explicit opt-in.

Every new tool, runner, trainer, materializer, replay harness, profiler, VJP
producer, archive mutator, or eval wrapper that can create large artifacts MUST
include an automatic disk-hygiene path in the same landing. Large artifacts
include inflated videos/raw frames, scorer tensor caches, decoded PNG trees,
NPZ/VJP shard bundles, checkpoints, candidate archive sweeps, profiler traces,
temporary virtualenvs, build products, and copied public-PR worktrees.

The cleanup rule is "certify or block." Never delete or move a large artifact
unless a machine-readable record preserves deterministic reproducibility:
original path, bytes, SHA-256 or tree hash, command/config/argv/env where
available, source archive/runtime/content hashes where applicable, cold-store
destination when moved, false-authority score flags, and the reason the artifact
is rebuildable or safely externalized. If that proof is missing, emit a blocker
and keep the bytes. No signal loss ever.

Default cleanup should be lossless: use context-managed temp dirs for true
scratch, delete success-only scratch automatically, move certified rebuildable
bulk to SSD cold store before deleting local bytes, and leave a manifest or
symlink when existing tools still need the original path. Destructive delete is
allowed only for trivial caches/build products or explicitly certified
rebuildable scratch; all other cleanup defaults to move/cold-store. Any
operator-facing evidence path must be durable and must not cite `/tmp`.

Before launching long MLX training, full-video VJP, exact replay, inflate,
materializer, or archive-search jobs, run the storage waterfall/preflight path
and fail closed if no SSD/local tier has enough free space. Before finishing any
such landing, add or reuse the auto-clean hook so future runs do not leave
orphaned bulk files behind.

**Resumability + per-stage checkpoints are MANDATORY for every launch (operator
binding 2026-06-27, HIGHEST EMPHASIS).** NEVER launch anything — training run,
long job, detached daemon, sweep, paid dispatch — that is NOT crash-resumable
from disk (`--resume-from`) AND that does not SAVE ALL NECESSARY + PRESERVE a
complete, byte-close-loadable checkpoint at the END OF EACH STAGE (every
curriculum/phase boundary, e.g. CE / tau_softplus / l7 / Muon), plus periodic
intra-stage saves for long stages. PRESERVE every stage checkpoint under a
distinct stage-encoded filename (do NOT overwrite the prior stage); save the EMA
shadow (not live weights) per the EMA non-negotiable; write atomically
(tmp+rename); include every cfg key the byte-close + resume paths need.
Loop-end-only saving is FORBIDDEN. Rationale: a multi-day run on one GPU with
loop-end-only saving loses ALL work on any crash/OOM/operator-cut and blocks
early byte-close + per-stage A/B (which stage moved d_seg). The per-stage
checkpoint is both crash-insurance AND a measurement artifact (each stage's
output is independently byte-closeable → N early rows from one run). Binds all
subagents that launch runs. Memory:
`feedback_never_launch_non_resumable_per_stage_checkpoints_20260627.md`. Sister
of the daemon-durability + scale-measured-safeguarded + per-stage-treatment +
deterministic-reproducibility non-negotiables.

**Deterministic reproducibility principles (operator binding 2026-06-27, HIGHEST
EMPHASIS).** Deterministic reproducibility is ONE of our two hard limits (with
contest compliance); everything else is ours to turn. Every launch, measurement,
and submission MUST be deterministically reproducible: (1) **seeded +
deterministic** everywhere — all RNG (torch/numpy/random/MLX) from a single
recorded `seed`, deterministic algorithms where supported, same
seed+config+inputs → same result; (2) **resumable-from-disk** (the sister rule
above) — state on disk, continue bit-faithfully from the last checkpoint;
(3) **numpy-fp32 reference is the bit-identical verdict authority** — MLX/torch
match it (parity ≥ 0.9997), **MPS is NEVER an authority** (per "MPS auth eval is
NOISE"), macOS-CPU/MLX are advisory not contest-CPU, only `upstream/evaluate.py`
on the EXACT archive bytes is a score (per "Frontier scores are pointer-only" +
"Submission auth eval — BOTH CPU AND CUDA"); (4) **realized-through-R authority**
— d_seg/d_pose measured through the actual R operator + frozen CPU-torch scorer
on the exact shipped bytes, never a proxy/un-roundtripped field; (5)
**deterministic decode** — same `archive.zip` → bit-identical inflate output
every run/host within the 30-min budget; the GENERIC generator is FREE (rule
118), only LEARNED/video-derived payload is COUNTED, and **NO scorer weights /
SegNet / PoseNet / GT-argmax table** ship in the archive (nor smuggled into
inflate.py "code" — the hide-data-in-code fake); (6) **provenance with every
result** — git hash, seed, config, upstream snapshot sha, hardware/axis, archive
sha256+size, realized-through-R deltas; (7) **numpy-portable reference kept** so
any host reproduces byte-for-byte, CPU/CUDA separate axes never inferred from
each other. This substrate is what makes deterministic GENERATION a legal
score-mover: a seeded, bit-identical, host-portable generator reproduces the
witness exactly under the contest runtime from a tiny counted sufficient
statistic, shipping zero scorer weights. Memory:
`feedback_never_launch_non_resumable_per_stage_checkpoints_20260627.md` (sister)
+ the rule-118 "compile the generator" discipline above.

## Evaluator-Equivalent Witness Compiler Paradigm — NON-NEGOTIABLE

Current HiNeRV/SNeRV work is no longer ordinary video-codec fidelity work. The
contest-native objective is the shortest compliant `archive.zip` whose
`inflate.sh` output is a witness inside the same frozen evaluator cells as the
source video. Only three terms carry authority: SegNet last-frame RGB argmax
pixels, PoseNet official two-frame YUV6 output, and exact archive bytes. Human
visual fidelity is non-authority unless it causally improves one of those terms.

Treat NeRV/HNeRV/HiNeRV/SNeRV as possible witness backends, not as the goal.
They may generate evaluator-equivalent witnesses directly, or they may compress
mask grammar, pose trajectory grammar, evaluator-inverse renderer state,
score-effect codebooks, LF/HF carriers, pair-local residuals, and sparse
hard-pixel/hard-pair sidecars. The winning representation is whichever legal
mixture minimizes:

`100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`

Every actuator, optimizer, byte control, modelsize control, sidecar, codebook,
or residual section must report its score-unit value per byte whenever measured:
Seg delta, Pose delta using the nonlinear square-root term, byte delta, authority
surface, and receiver/parse-back survival. Do not admit updates merely because a
proxy loss improved. Admit only when the relevant evaluator-cell debt improves
or a fail-closed blocker records why it did not.

Before HiNeRV/SNeRV long MLX training is approved, the executable readiness DAG
must show the right gates in order: charged/free source-boundary hygiene, exact
scorer-oracle/cache geometry, archive parse-back selection, short receiver
surface smoke, joint Seg/Pose trust region, family-specific hard blocker closure,
full-video MLX replay, receiver-closed archive proof, and then exact CPU/CUDA
replay. HiNeRV's current hard blocker is target-region class birth that survives
uint8/resize/parse-back without total Seg spill or Pose harm. SNeRV's current
hard blocker is official MFU/HFR/TUB source-forward train/export/runtime binding
plus LF/HF representation collapse under real byte pressure. Build MLX-first,
with deterministic NumPy reference and Torch parity surfaces kept portable.

You are operating inside a dual-track lab for the comma video compression challenge.

Read `PROGRAM.md` before making changes.

## THE CURRENT FRONTIER + FOCUS + PRIORITY — THE NON-RGB TASK-SPACE WITNESS CAPSTONE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator binding directive 2026-06-25 verbatim *"the small basis we selected was likely too
small and we needed to pivot off hnerv to witness and nonlinear and new representation and carrier more
optimal than 50k epoch straight out the paper seg power law w pose and seg trained in fully full rgb and
that at bc20 that was not enough capacity"* + *"you're basically just running pr95 again and doing fake
implementation of what was supposed to be our capstone"* + *"Build the witness capstone"* + *"Chroma too"*
+ *"implementations are not optimal yet"* + *"establish as new frontier and focus and priority."* This
section is subordinate ONLY to the NO-FAKE supreme rule and THE GOAL (sub-0.15). Pointer UNMOVED at
contest-CPU **0.19110** (a borrowed PR101/PR110 recode) — that is the honest state until a measured
witness row moves it. Cross-refs (durable, compaction-survivable): memory
`[[muonjump-segplateau-pivot-to-step-nonlinear-CURRENT-STATE]]` +
`[[dont-rerun-pr95-reskinned-as-capstone-beware-looping-same-wrong-vehicle]]` +
`[[dag-survives-compaction-deterministic-repro-crux-convergence-standing]]` + the canonical work-graph DAG
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.

### What the frontier IS (the vehicle)

The capstone is a **non-RGB TASK-SPACE WITNESS** — OUR OWN carrier, designed from the measured deep-math
crux, NOT a PR95/HNeRV reskin. It is a **nonlinear coordinate-INR that amortizes the SegNet argmax
partition directly** (scorer-only-trained, no full-RGB reconstruction), spending its byte budget on the
**scorer-relevant manifold** instead of on full RGB. Reallocating the SAME bytes off full-RGB onto the
task-space manifold yields more EFFECTIVE capacity at equal/lower bytes — this is how we get capacity
WITHOUT scaling (which is the resolution of the capacity-vs-rate-headroom trilemma below). Canonical
substrate already in-tree: `src/tac/boundary_math/lever_b_generator.py` (`ScoreNativeSegGenerator`: MLX
coord-INR + deterministic Fourier features + FiLM-per-pair-mod + 5-class-logit head; numpy-portable
reference + npz save/load), the pose half `src/tac/boundary_math/amortized_luma_carrier.py` (byte-closed),
and `src/tac/torch_vehicle/boundary_routing.py` (KKT capacity-routing primitives).

### The trilemma this resolves (measured)

- **bc20 small basis:** rate cheap (~0.059) but d_seg UNDER-CAPACITY (floor ~0.0021–0.0037 vs need
  ~0.00087) → S~0.31. Measured, settled, do NOT re-open as "bigger bc20."
- **bc36 (PR95-size):** d_seg adequate (~6e-4) but rate at frontier (~0.118) → S~0.19 = **just PR95**.
- **The witness:** gets BOTH — adequate d_seg at low rate → the sub-0.15 path. This is the ONLY arm that
  is not dominated.

### Pose is SOLVED — the Quantizr-style stored-target sidecar (operator 2026-06-25)

**Pose is pretty much solved with the Quantizr-style sidecar we already have built — do NOT re-treat it as
an open problem.** The scorer computes `d_pose = MSE(PoseNet(generated_pair)[:6], PoseNet(original_pair)[:6])`,
so the GT target is just the 6 PoseNet scalars per pair. We STORE them (`src/tac/scorer_targets.py`:
600×6×fp16 = 7.2KB raw / <5KB zlib; further compressible to ~1–2KB via `src/tac/pose_from_embedding.py`'s
MLP, or 2.7× via the low-rank pose codec, task #140) and supervised-condition the render to hit them →
d_pose ~3.4e-5, contribution `√(10·d_pose)` ~0.018, near-free bytes. **The "pose collapse" (d_pose
2.67–12.66) was the amortized-luma-CARRIER composition — a different, suboptimal approach that tried to
RECONSTRUCT pose from a luma INR — NOT the stored-target sidecar.** Do not cite the collapse as a reason
to build a chroma pose carrier. The witness composes with the already-built stored-pose sidecar; the
witness's sole binding controllable job is **d_seg**.

### Chroma is a d_seg lever (operator 2026-06-25 "Chroma too")

**SegNet reads RGB** — its argmax depends on chroma, so chroma is a genuine d_seg actuator. The seg-frame
has RGB-slack: chroma channels carry argmax-relevant signal the witness should route capacity into where
it flips the partition (the codim-1 boundary annulus). This is chroma's PRIMARY value — a d_seg lever, not
a pose rescue (pose is solved above). Secondary: PoseNet reads YUV6 (4 luma + 2 chroma), so if a witness
frame is also pose-scored, its chroma planes feed PoseNet — but since pose rides the stored-target sidecar,
chroma is optimized for d_seg first. Any witness d_seg verdict that ignored chroma is provisional and must
be re-measured with chroma active.

### Optimal-form discipline (operator 2026-06-25 "implementations are not optimal yet")

No witness verdict (adopt / kill / "this lever doesn't move d_seg") is load-bearing until the
implementation is at OPTIMAL FORM: per-lever hyperparameters tuned to each lever's OWN optimum (not a
shared default), curriculum/round-trip bugs fixed, chroma active, capacity-routing engaged. Provisional
sub-optimal-form results are LABELLED provisional (per the deterministic-reproducibility spine). Compare
each lever at its own optimum, then the verdict is admissible. This is the parity discipline applied to
our own carrier.

### The measured d_seg levers (binding residual = union of ALL inter-class edges; measured 2026-06-25)

**Crux refined (MEASURED, witness build a922483dfc636ccc3):** the flip-prone (small-margin) pixels are
distributed across ALL classes — 50% class-0, 19% class-1, 13% class-2 — so the binding residual is the
**union of all inter-class edges**, NOT just class-1 lane islands (the prior lane-only framing was too
narrow; class-1 lanes are ONE component of the all-class edge set, and the ~8-dim lane-orbit manifold
remains the hard long-tail). Lever ranking, measured on the frozen CPU-torch SegNet argmax (n600,
`[macOS-MLX research-signal]`, baseline d_seg 0.008257):

1. **ALL-CLASS DIRECTIONAL (anisotropic/curvelet) Fourier basis [THE decisive lever, ~0 byte]** —
   orient the Fourier features to the all-class boundary tangent field: **−48%** d_seg (lane-only is only
   −8%). A 0-byte DETERMINISTIC train-time prior → compiles into inflate.py for FREE (converges with the
   inflate.py rate lever above: same object). This is the #1 lever; basis-match is PRIOR to capacity.
2. **Capacity-routing [pays ONLY after basis-match]** — KKT waterfill on margin-saliency
   (`boundary_routing.py`). Capacity ALONE on an isotropic basis does nothing / HURTS (+6%); once the
   basis is all-class-directional, modest capacity then pays (n96 −64% combined). Dominated until #1.
3. **Round-trip-survival [R_surv]** — train with the R operator in-loop (bicubic↑384→874 → uint8-STE →
   bilinear↓→512×384); flips are texture-dependent and a low-pass kills naive sine (Gibbs aliasing).
4. **Curriculum-fix** — only the smooth-stage RAISES d_seg (measured); CE+softplus LOWER, c1a→neutral,
   lambda+sigma neutral, Muon is THE drop. Drop the d_seg-harmful stage from the witness curriculum.
5. **Activation/representation** — step-native / partition-indicator basis (no Gibbs, O(1) params/edge,
   L∞-at-edge optimal) is the topology-matched chart for a piecewise-constant argmax target under the
   capacity-limit regime. (The `gauss`/step activation is in-code but UNSWEPT — the deep-math-predicted
   step-native lever, a named headroom item toward the ~0.001 need; best measured so far 0.004445 ≈4.4×.)

### inflate.py is a FREE interpreter — COMPILE the generator, count only the video-derived payload (operator 2026-06-25)

**The rate term scores ONLY `archive.zip` bytes (`upstream/evaluate.py:63` — `compressed_size = (submission_dir/'archive.zip').stat().st_size`); inflate.py / inflate.sh are NOT sized, and the score has NO time term (`evaluate.py:92`) — the only constraint is the 30-min full-eval budget (`README.md:114`, T4 16GB-VRAM or CPU 4×/16GB).** Contest rule (`README.md:118`): external CODE/tools are FREE and don't count; **large video-derived artifacts (neural-net weights, meshes, point clouds) MUST be in archive.zip and ARE counted.** The crisp boundary that IS the rate game:

| FREE in inflate.py (untimed except the 30-min budget) | COUNTED in archive.zip (rate term) |
|---|---|
| the generator ALGORITHM / forward-pass code | LEARNED neural-net weights (rule 118) |
| deterministically-GENERATED tables (Fourier `B` from a seed, fixed bases, parametric rasterizers) | the VIDEO-DERIVED payload (per-frame lane coords, learned residuals) |
| arbitrarily complex deterministic compute (iterative solvers, runtime-generated codebooks) | anything that is a "large artifact" |

**"Compile nonlinear d_seg" = move the maximal DETERMINISTIC GENERIC structure of the witness generator into inflate.py (zero rate) and store ONLY the irreducible video-derived statistic in archive.zip.** inflate.py is a Turing-complete interpreter run at decode time — it may run an arbitrarily sophisticated deterministic program (the coordinate transform, Fourier features, FiLM graph, a parametric lane-curve rasterizer via openpilot polynomial + homography, a big runtime-generated codebook) for FREE, finishing within 30 min. archive.zip then carries only the ~8-dim lane-trajectory coords per frame (AR-coded → hundreds of bytes) + whatever minimal LEARNED residual the texture-survival wall genuinely requires. This is the rate half of the sub-0.15 path: the indirect-RD sufficient statistic (8-dim) is counted; the deterministic generator that expands it to the argmax partition is free.

**NO-FAKE / compliance boundary (binding):** GENERIC ALGORITHM = free in inflate.py; VIDEO-DERIVED LEARNED content = counted in archive.zip. You may NOT smuggle a video-derived per-frame table/weights into inflate.py disguised as "code" to dodge the rate term — rule 118 forbids it and it is the hide-data-in-code fake (sister of NO-FAKE #6/#7). Synthesizes with the "Deterministic packet compiler" + "Native eval-time runtime discipline" non-negotiables (native code that EXPANDS the legal witness-program class is allowed; learned artifacts are counted). Every rate claim from this lever is MEASURED byte-closed (the archive.zip stat), never asserted.

### The anti-pattern this frontier extincts (binding, NO-FAKE #7)

A run is a FAKE of the capstone if it is PR95's curriculum (CE→softplus→smooth→QAT→c1a→lambda→sigma→Muon)
on PR95's HNeRV decoder (PixelShuffle+bilinear-skip+sin), full-RGB-trained, with only an activation/lever
bolt-on. That is borrowed-substrate-passed-as-original-work + means-as-ends. Before calling anything "the
capstone path," do the borrowed-substrate accounting (ours-original vs PR95) and check it against the
task-space witness, NOT against "which activation/curriculum tweak on PR95." Beware the loop trap:
babysitting one wrong-vehicle run for many ticks IS running the same over and over — if N ticks pass with
no decisive new EXACT-relevant signal, STOP and pivot.

### The END (deterministic, byte-closed)

Witness d_seg → ~0.001 at low rate → compose with the already-built Quantizr-style stored-pose sidecar
(pose solved, ~1–5KB) → byte-close in the L13 task-space format → exact eval (`tac.contest_score` /
`upstream/evaluate.py`, contest-CPU/CUDA, NEVER MPS as authority) = a real row below 0.19110, then toward
0.15. Every unit MEASURES a byte-closed row that sharpens d_seg(H)/bytes(H) OR tightens the crux with a
deep-math lens + existence-proof cross-check, then appends to the DAG — never a chat-only insight.

## Vehicle Operating System — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator binding directive 2026-06-09 (the fleet-wide-meta-bug crux). The repeating failure was
NOT picking wrong ideas — it was letting **names, partial mechanisms, and bolt-ons stand in for complete
vehicles** ("vehicle names outran vehicle implementations"). 2026-06-09 full-stack audits proved our
"HiNeRV" is an L0 SKETCH (git `7a004e5bd`), SNeRV is a cross-wiring defect (`83479abfe`), pact_nerv_vq is a
skip-free decoder with the right objective (`222099bc4`) — different labels, same missing HF carrier
mechanism, same inactive scorer objective, same failure.

**The canonical operating system is `docs/vehicle_operating_system.md` (read it before touching any
vehicle).** Its 5 binding rules: (1) no named vehicle exists until its `vehicle_fidelity_manifest.v1`
(`tac.substrates._shared.vehicle_fidelity_manifest`) `verify()` passes — a name/import/docstring is NOT
proof; (2) no contextual optimization before intrinsic (L1/L2) — no long training run before the vehicle
solves its own native sanity task; (3) no cross-vehicle bolt-on before the receiving vehicle's reference
contract passes (compose only at L4+); (4) no "score-aware" run unless the SegNet/PoseNet objective weights
are explicit + nonzero (else `scoreaware=false`) — enforced by
`check_score_aware_run_has_nonzero_scorer_objective_weights`; (5) no row updates the score roadmap unless
`authority_tier` + `metric_family` allow it (the metric-laundering firewall). The L0–L7 maturity ladder,
the 10 non-negotiable claim rules, the objective-activation rules, the per-family fail-closed claim gates,
the dashboard discipline, and the subagent-produces-manifest-not-prose contract all live in that doc. This
section is the pointer; the doc is canonical.

## NO FAKE IMPLEMENTATIONS — THE #1 NON-NEGOTIABLE, SUPREME OVER ALL OTHER RULES, HIGHEST POSSIBLE EMPHASIS

**This rule outranks every other rule in this file. If any other rule appears to permit a fake, this
rule wins. See the supreme-rule banner at the very top of `CLAUDE.md` — this section is its canonical body.**

**Source:** operator binding directive 2026-05-30 verbatim *"add non negotiable instrucitons no fake implmenetations to claude.md"* + sister directive *"make sure that tasks marked as done are actually done and no fake implmenetations"*. Anchor memos: `feedback_15_item_audit_validate_fix_harden_test_blanket_approval_1to1_fidelity_with_documented_adaptations_standing_directive_20260529.md` + `feedback_optimize_iterate_highest_ev_boldest_individually_fractally_optimized_mlx_deployed_aggressive_frontier_breaking_no_fake_implementations_standing_directive_20260529.md` + Slot EEE fake-implementation audit `feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_landed_20260529.md`. Empirical anchor: Slot EEE audit found 1-of-7 L0 scaffolds (Slot RR) was structurally FAKE (`apply_pose_axis_null_projection` returned Tier A markers + canonical menu size constants but applied ZERO perturbation; 64 tests verified menu-size constants not behavior) and 5-of-7 PARTIAL (real math primitives + documented adaptations or structural simplifications under explicit FALSIFIED-AT-IMPLEMENTATION-LEVEL classification per Catalog #307).

**The rule.** Every implementation claimed to perform a technique MUST actually perform that technique on the actual inputs the technique is defined over. Tier A observability-only consumer markers per Catalog #341 do NOT exempt the underlying function from doing the work it names. A function named `apply_X_via_Y` that returns canonical metadata without applying Y to the inputs is a FAKE implementation regardless of test coverage on metadata fields.

**Eight forbidden classes** (each is canonical FAKE per Catalog #307 IMPLEMENTATION-LEVEL):

1. **Returns-canonical-markers-without-doing-work.** Function claims to apply a transform / probe / measurement but returns only `{predicted_delta_adjustment=0.0, promotable=False, axis_tag="[predicted]"}` (Catalog #341 Tier A markers) + canonical reference strings + no observable mutation of the inputs. Slot RR canonical anchor.

2. **Tests-verify-constants-not-behavior.** Test suite asserts `canonical_menu_size == 64` + `axis_tag == "[predicted]"` + `provenance_dict` shape but never verifies that the function actually transforms inputs as the name claims. If every test would still pass when the function body is replaced by `return canonical_markers`, the test suite is verifying constants not behavior.

3. **Synthetic-fixture-instead-of-real-input.** Smoke / probe / scaffold runs on random-noise tensors or hand-crafted toy fixtures instead of the actual `upstream/videos/0.mkv` real frames the technique is canonically defined over per Catalog #213. The inverse-steganalysis cost-discrimination signal degenerates on uniform-random input; reported "validation" is structurally meaningless. Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" — synthetic-fixture scaffolds MUST tag `research_only: true` AND MUST NOT be cited as canonical empirical anchors.

4. **Placeholder-string-in-canonical-data-field.** `paired_cuda_targets`, `archive_sha256`, `canonical_helper_path`, `predicted_band_validation_status`, or any structured-data field carrying a literal placeholder string like `"pending_ratification"` / `"TBD"` / `"<value>"` / `"placeholder"` as if it were the canonical value. Per Catalog #287 placeholder-rationale rejection: data-content layer is canonical too.

5. **Enum-padding-without-distinct-implementations.** A 4-value canonical strategy enum where 3 of the 4 values dispatch to the same underlying allocation / filter / transform. Catalog #308 alternative-probe-methodology enumeration is structurally padded — looks comprehensive but each branch is the same code. Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — each enum branch must be a structurally distinct implementation OR the enum value MUST be removed from the canonical surface.

6. **Search-masquerading-as-a-solver/compiler (the candidate-search trap; operator anchor 2026-06-10).** A surface named `compiler` / `solver` / `synthesizer` / `quotient` / `inverse-solve` that, on the real inputs, only ENUMERATES candidates and scores them (brute-force / greedy / beam / random / sweep search) is a SEARCH, not a solver — naming it a compiler/solver is FAKE unless (a) the artifact is honestly labeled as candidate-search AND (b) its real mechanism is documented (the search space, the ranking signal, and the explicit absence of a closed-form / variational / preimage solve). Operator verbatim 2026-06-10: *"our interpreter/compiler architecture is interesting but still kind of a fake implementation ... there are much more direct and elegant ways of math ... derivatives and integrals and manifolds and partial diff equations."* The fix is BOTH: be honest (call the search a search) AND, where the directive actually demands a solver, build the real solve (variational / KKT / preimage / closed-form against the measured oracle geometry), not a fancier search wrapper wearing a solver's name.

7. **Borrowed-substrate-passed-as-original-work (the absorb-recode-as-innovation trap; operator anchor 2026-06-10).** Claiming a result is "original" / "innovative" / "novel" / "class shift" / "competitive" when it is in substance an absorb-recode of a competitor's published method (canonical anchor: the recoded-R3 hold, −2.59e-5 over PR#112 and within contest reporting precision, built FROM PR#112's codec) is FAKE originality. Every innovation/originality/competitive claim MUST be backed by an itemized `borrowed_substrate_accounting` that separates ours-original bytes/mechanisms from borrowed ones; a borrowed substrate is a DEFENSIVE BANK for readiness, NEVER the innovative submission. Per the Innovation Gate in `.omx/research/GOAL_standing_v3_20260610.md` — the "competitive OR innovative" statement on any submission must be UNQUESTIONABLE.

8. **Surrogate-optimized-but-not-exact-authority-verified (the proxy-score trap; sister of "MPS auth eval is NOISE").** Claiming a score improvement / frontier / promotion / kill / "beats baseline" from a surrogate — proxy loss, telemetry, PSNR, MLX research-signal, local-CPU advisory, foreign-host FP, and ABSOLUTELY MPS — WITHOUT the exact contest `evaluate.py` 600-sample row at the authority tier the claim actually asserts is a FAKE score claim. Surrogates are gradient rows + priors ONLY; the exact argmax-`d_seg` / official-`d_pose` / real archive bytes are authority; any surrogate↔exact gap is itself a finding, never the verdict. MPS is NEVER authority (corrupts 95.5% of orderings — a contamination marker requiring rebuild). GT decodes ONLY via `frame_utils.yuv420_to_rgb` (PyAV rgb24 manufactures ~100× phantom pose). Recompute the score from components — the rounded `final_score` field lies. Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA" + "Frontier scores are pointer-only" + the authority ladder.

**Sister rule: tasks marked done must actually be done.** Per operator directive 2026-05-30: never mark a TaskList row `completed` when the underlying work is a fake implementation per the 8 classes above. Per CLAUDE.md "Memos must be implemented" — a landed memo describing canonical work that was actually placeholder-emission is itself a fake-implementation incident at the documentation surface. The 6-hook wire-in declaration per Catalog #125 + the canonical apparatus mutation chain (lane registry + memory entry + landing memo + canonical posterior anchor) MUST reflect actual behavior, not claimed behavior.

**The fix when caught.** Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 paradigm-vs-implementation classification: a FAKE implementation surfaced via audit is canonical reactivation territory (the paradigm is intact; the specific implementation is falsified). The remediation cascade:
1. Classify the fake-implementation incident per the 8 forbidden classes above + record per Catalog #307 IMPLEMENTATION-LEVEL falsification.
2. Apply the canonical 2-landing pattern per "Bugs must be permanently fixed AND self-protected against": (a) immediate code fix that makes the function actually do the work it names; (b) NEW STRICT preflight gate OR canonical-helper invariant that refuses re-introduction of the fake-implementation class.
3. Honest reframe of any landing memo / canonical equation anchor / lane registry evidence that cited the fake implementation as empirical anchor — per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE, the original anchor is NOT mutated; a NEW row supersedes it with the falsification classification.
4. Sister-substrate audit: if the same FAKE pattern exists in sister substrates (per the META-meta finding from a8bc7e79 sweep — bug classes have 6-7× spread across the repo), enumerate + fix in the same commit batch.

**Cross-references**: CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" (the runtime-effect surface that THIS rule generalizes) + "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" (the dispatch-discipline surface) + Catalog #220 (substrate L1+ scaffold operational mechanism) + Catalog #272 (distinguishing-feature integration contract) + Catalog #307 (paradigm-vs-implementation falsification classification) + Catalog #341 (Tier A canonical-routing markers — markers are observability NOT score-claim) + Catalog #287 (placeholder-rationale rejection at data-content layer) + Catalog #105/#139 (no-op detector + byte-mutation smoke discipline) + the Slot EEE audit memo above.

## THE GOAL — SUB-0.15 EXACT SCORE — NON-NEGOTIABLE, HIGHEST EMPHASIS (mission, second only to NO FAKE)

**This section is the canonical body of the 🎯 banner at the top of `CLAUDE.md`. It is the mission. Only
the NO-FAKE supreme rule outranks it.**

**Source:** operator binding directive 2026-06-10 verbatim *"you did not actually achieve the goal as the
frontier score wasn't actually lowered, what is the new sub 0.15 goal non negotiable highest emphasis."*
**Empirical anchor (the failure this extincts):** a full 2026-06-10 session produced the NO-FAKE supreme
rule, the closed-spec/boundary-math/waterfilling architecture, the seg-core + boundary-solver + Rust
runtime-less decoder + the MEASURED information-theoretic floor (S_floor=0.11797) + five honest
score-native experiments + the deferral-recovery ledger + a running decisive d_seg-loss test — and the
**exact frontier score did NOT move (0.19110 unchanged, still above even T_1).** A large, disciplined,
honest body of work that did not lower the exact score is a MISS, not a success. This section makes that
verdict structural.

### The rule
The goal is to LOWER THE EXACT CONTEST SCORE `S = 100·d_seg + sqrt(10·d_pose) + 25·|archive.zip|/37_545_489`
below **0.15**. Success has EXACTLY ONE definition: `.omx/state/canonical_frontier_pointer.json` records a
NEW, LOWER exact-eval score from `upstream/evaluate.py` over the 600 samples, on contest-CPU AND/OR
contest-CUDA on 1:1 hardware, on the EXACT `archive.zip` bytes that would be submitted. The ladder:
* **T_3 = sub-0.15** — THE target. The default aim of every campaign. Pursue it, not a maybe.
* **T_1 = sub-0.19** — the floor of acceptable (a near-gate, NOT the goal). Above T_1 = failing.
* T_floor — the information-theoretic lower bound (MEASURED S_floor=0.11797 rate-dominated; the
  headroom proof, not the target).

### What does NOT count as goal progress (binding taxonomy — the means/ends firewall)
NONE of the following is goal progress, no matter how rigorous, original, or hard-won:
* tools, harnesses, solvers, codecs, archive grammars, runtimes, Rust/native lowerings;
* floor derivations, research/design memos, paper inventions, SOTA surveys;
* located cruxes, honest negatives, DEFER verdicts, deferral/orphan ledgers, root-cause diagnoses;
* holding the existing frontier; a measurement merely RUNNING; advisory / `[macOS-MLX]` / proxy / CPU-non-contest rows.
These are **MEANS**. The **END** is a lower EXACT score. Narrating a means as if it were the end is a
**mission-level NO-FAKE violation** (a "progress" claim the exact pointer does not honor). A unit or
session that ends with the exact pointer UNMOVED and S > T_1 **has not achieved the goal — state that
plainly** and aim the next unit DIRECTLY at an exact-eval row that crosses the threshold.

### Binding consequences
1. **Bias every decision toward the action most likely to LAND A LOWER EXACT SCORE soonest** — not the
   most rigorous analysis, not the most interesting crux, not the cleanest abstraction.
2. **When a path walls, PIVOT** to the next path that can produce a lower exact score — do NOT spend the
   next unit further characterizing the wall (that is means-hoarding). One crisp wall-verdict, then pivot.
3. **Infrastructure is built ONLY in service of an imminent exact-eval row.** If a build does not have a
   named, near-term exact-eval row it feeds, it is premature.
4. **Spend the Modal <$5 budget to BUY exact rows.** It exists to measure real byte-closed candidates on
   contest hardware. Hoarding it while the score sits unmoved is the wrong default; a fail-closed paid
   eval on a real candidate that beats the advisory bar is the RIGHT default (decide-don't-defer).
5. **The session report leads with the exact pointer delta.** Every wrap-up states, first: did the exact
   frontier move, by how much, on which axis — then everything else. If it did not move, that is the
   headline.

### 10-year horizon + long-term autonomous research (operator 2026-06-10)
Sub-0.15 is the near-term MILESTONE, not the terminus. The GOAL is a **10-year autonomous research
program**: build the proof-carrying evaluator-equivalent program compiler into a general system, and
sustain autonomous research + engineering that COMPOUNDS over a decade — each session's VERIFIED
artifacts, canonical equations, measured oracle geometry, and continual-learning posterior make the next
session smarter (per "Results must become system intelligence"). **Think in decades:** prefer the durable
class-shift and the reusable, self-improving system over the one-off hack; invest in autonomy (durable
disk state, resumable-from-disk, detached daemons, marker-on-exit waiters, continual-learning loops) so
the program runs across years and sessions without losing signal. Encourage long-horizon bets that a
short sprint would reject. **BUT the long horizon is NOT a license for means-without-ends:** every unit
still moves the exact score or HONESTLY reports it did not — the sub-0.15 / means-vs-ends firewall above
is unchanged. Patience for the big class-shift AND per-unit exact-score honesty are BOTH binding;
together they are how a decade of autonomous work compounds instead of drifting. **"Any score sub-0.19 is
good progress on the way down" (operator 2026-06-10):** incremental lowering of the frontier IS progress
— bank it (defensive lossless banks approved), then keep descending toward sub-0.15 and the decade vision.

### Cross-references
The 🎯 banner at the top of `CLAUDE.md` (the TL;DR) · the NO-FAKE supreme rule (means-as-ends is a fake) ·
`.omx/research/GOAL_standing_v3_20260610.md` (the operating-law threshold ladder + scoreboard this
section binds) · "Frontier scores are pointer-only" (the SoT) · "Frontier target" · "Long-burn
score-lowering campaign default" (how to convert a floor-breaking family into an exact-row campaign) ·
"Submission auth eval — BOTH CPU AND CUDA" (what makes an exact row authoritative) · "Results must become
system intelligence" (the decade-compounding mechanism).

## ANTI-SIGNAL-LOSS — no deferral of READY high-EV work; janky-prototype closures RE-OPEN; measurement-first — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator binding directive 2026-06-11 verbatim — *"everything we closed were like janky prototypes
and far from the top-AIML versions our project calls for ... many deferred and retired and orphaned are
causing similar signal loss which caused us to get leapfrogged over [241]LOC when everything was sitting
ready and your own research directive had that on the path ... but as deferred when it was obviously a top
high-EV priority."* Empirical anchor: the May 4 2026 race — a **241-LOC silver medal shipped past our ready
stack that we held as "deferred"** (sister of "Race-mode rigor inversion" below). This section is the
standing extinction of the **signal-loss-via-deferral**, the **janky-prototype-closure**, and the
**re-diagnose-instead-of-measure** failure class. Only NO-FAKE and THE GOAL outrank it.

### 1. Anti-deferral / EV-readiness gate (DEFER is FORBIDDEN for READY ∧ high-EV nodes)
`DEFER / RETIRE / ORPHAN` is a **forbidden resting state** for any node that is BOTH (a) **READY** — its
code/contract already exists OR its first step is a $0 local smoke — AND (b) **high-EV** — it clears the
break-even / sits on the critical path to a lower exact score. Such a node is **ACTIONABLE-NOW**, not parked.
Every node carries the gate tag `{READY: y/n | EV: high/med/low | STATUS: DOING-NOW / blocked-by-<named-
measured-X> / genuinely-deferred-because-<named-measured-blocker>}`. A node may be `genuinely-deferred` ONLY
with a NAMED, MEASURED blocker (e.g. "needs paid GPU $X" / "exact eval returned S=0.46"). "Looked hard,"
"felt lower priority," or "a prototype gave a bad result" are NOT valid deferrals — the last RE-OPENS the
node per rule 2. A READY ∧ high-EV node sitting un-launched is the bug this rule extincts; fan it out
(sister: "Race-mode rigor inversion + parallel-dispatch first" + the 2–3-subagents-running directive).

### 2. Janky-prototype → top-AIML RE-OPEN (closure requires top-AIML + measured row + recursive greenup)
Per Catalog #307 (paradigm-vs-implementation falsification): a verdict reached on a **janky prototype**
falsifies the IMPLEMENTATION, not the PARADIGM. **Every prior `closed / falsified / DEFER / retired /
orphaned` verdict that rested on a prototype-grade implementation is IMPLEMENTATION-LEVEL falsified,
PARADIGM INTACT, and RE-OPENED for a top-AIML (SOTA-grade) re-attempt.** A closure is valid ONLY when ALL
THREE hold: (a) a **top-AIML implementation** (export contract + numpy-portable inflate + torch-parity gate
from byte zero, per HNeRV-parity discipline — not a sketch); (b) a **MEASURED exact-scorer byte-closed row**
(advisory local is the gate, paired contest-CPU/CUDA is the verdict); (c) a **3-clean-pass recursive
senior-engineer greenup review** (per "Recursive adversarial review protocol") that QUESTIONS ALL
INTERPRETATIONS — any unresolved interpretation resets the counter. Anchors: the orphaned inverse-scorer
103-surface inventory (`evaluator_inverse_orphan_inventory_20260609.md`), the LEARNED-CNN postfilter
abandoned on proxy/PSNR (never closed on the exact scorer), the deterministic/boundary levers retired as
prototypes.

### 3. Measurement-first — stop re-diagnosing
Interpretations are cheap and have been wrong repeatedly: the capstone d_seg plateau was re-diagnosed FOUR
times in two days (0.505 wall → EMA-shadow artifact → true plateau → under-training → capacity+broken-
curriculum), each "wall" dissolving into an artifact. **Every cycle MUST produce a MEASURED byte-closed S
row OR a $0 smoke with a falsification threshold — NOT another interpretation.** Measured rows are the
currency; a session that ends with N interpretations and 0 new measured rows has lost signal. (Sister of
NO-FAKE "surrogate-optimized-but-not-exact-authority-verified" + "Results must become system intelligence.")

**Operating consequence:** when the conversation/DAG surfaces a finding (orphaned surface, futuristic path,
adapter lever), it is FOLDED INTO the work-graph with the rule-1 gate tag and, if READY ∧ high-EV, LAUNCHED —
not summarized as a means. The canonical work-graph is
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`. Cross-refs: "Forbidden premature
KILL without research exhaustion," Catalog #307, "Race-mode rigor inversion + parallel-dispatch first,"
"THE GOAL — SUB-0.15" (the means/ends firewall this protects).

## Race-mode rigor inversion + parallel-dispatch first — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Postmortem source:** `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_may_4_hnerv_race_postmortem_20260505.md`. The May 4 2026 contest was decided in a **4 hour 8 minute race window** after PR #95 (HNeRV root) was published at 07:47:15 UTC. Final top 3 (PR #101 / #103 / #102 at 0.193 / 0.195 / 0.195) all submitted between 11:50 and 11:55 UTC. Silver medal (rem2 PR #103) was **241 lines of code in 2 files**. Our PR #107 apogee landed at 0.229 (~11th) — we had every primitive needed but spent the race window building meta-Lagrangian + predictor + sanity gates (sequential validation harness) instead of fanning out parallel dispatches.

The two rules below structurally extinct that failure mode.

### Rule 1: parallel-dispatch is a FIRST-CLASS DELIVERABLE, not an afterthought

When the user says "parallel" / "high-throughput" / "search" / "automation" / "stacking sweep" — the **first file built** must be the actuator that fans out N concurrent paid-GPU dispatches. Build the actuator BEFORE the ranker, BEFORE the predictor, BEFORE the sanity gate. A ranker without a parallel actuator is a planner that produces ranked plans no one executes. The canonical actuator + harvest loop is now checked in:

- `tools/parallel_dispatch_top_k.py` — `concurrent.futures.ThreadPoolExecutor` over `tools/lightning_dispatch_pr106_stack.py` (Lightning T4) or `scripts/launch_lane_on_vastai.py` (Vast.ai 4090). Includes per-dispatch + total-cost gating, per-dispatch timeout, harvested-JSONL output.
- `tools/harvest_and_reseed.py` — ingests the harvested JSONL, drops any row not tagged `[contest-CUDA]`, appends new empirical anchors to `.omx/calibration/anchors_*.json`. Closes the prediction → empirical → updated-prediction feedback loop.

The canonical sweep loop is therefore three commands:
```bash
.venv/bin/python tools/meta_lagrangian_search_cli.py \
    --lane-class apogee_intN --auto-sweep-bits 4,5,6,7,8 \
    --top-k 16 --output reports/sweep_ranked.json

.venv/bin/python tools/parallel_dispatch_top_k.py \
    --ranked-input reports/sweep_ranked.json --max-concurrency 16 \
    --provider lightning --estimated-cost-per-dispatch 0.11 \
    --max-total-cost 5.00 --harvest-output reports/sweep_harvested.jsonl

.venv/bin/python tools/harvest_and_reseed.py \
    --harvested-jsonl reports/sweep_harvested.jsonl \
    --anchors-path .omx/calibration/anchors_apogee_intN.json
```

If any future work proposes a "search engine" / "candidate generator" / "ranking primitive" / "sanity gate" without naming the parallel actuator, that work is INCOMPLETE. Closure requires naming the actuator that turns the ranking into N concurrent dispatches.

### Rule 2: strategic-rigor inversion at leaderboard moves

The agent's default prior is "max rigor: validate before dispatching." This prior is correct PRE-leader-shift (you're optimizing alone, every wasted dispatch is your money). It is WRONG POST-leader-shift (you're racing; every minute of gating is a competitor shipping ahead of you).

**Detection:** before any new candidate is dispatched, the agent MUST check whether the public leaderboard has moved in the last 24 hours. If yes, the prior **inverts**: the next action is the smallest credible bolt-on submitted within ~60 minutes, not another sanity gate. Concretely, the May 4 race showed:
- BradyMeighan: PR #97 (0.23) → #99 (0.197) → #100 (0.195) — **3 PRs in 2h 12min**
- rem2 (silver): PR #96 (0.21) → #103 (0.195) — **2 PRs in 3h 24min, 241 LOC final increment**
- EthanYangTW (bronze): PR #98 (0.196) → #102 (0.195) — **2 PRs in 2h 23min**
- "kitchen_sink" PR #105 — 1776 LOC, 21 files — **lost** to rem2's 241 LOC

The right move when N hours remain is the smallest credible bolt-on, not the most thorough system. Public PRs are checkpoints that lock in score, force honest contest-CUDA measurement, surface what's NOT working in 30 min instead of 4 hr, and establish presence. Holding for one polished shipment is the failure mode.

### Rule 3: cron + same-prompt loop is a fan-out cadence

A cron job firing every N minutes on the same prompt is the natural cadence for "fan-out K candidates and harvest." It is the WRONG cadence for "do another sequential validation pass." The May 4 cron was firing every 5 minutes and the agent kept choosing depth (more validation gates) over velocity (more dispatches). When loop tick X is shorter than dispatch wall-clock Y, every tick should fan out a new batch. When Y > X (12-hour training jobs), the tick should be a "harvest results + reseed" cycle.

Translation table when the loop fires:
- "push toward Shannon floor in absolute minimum wall clock" → fan out next batch of K parallel dispatches; do NOT add another gate.
- "extreme rigor" → applies to the FIRST cycle (calibration of anchors), not every cycle. After cycle 1, rigor is in the actuator's gating thresholds, not in sequential validation.
- "fix all bugs everywhere" → fix the bug class that prevents the actuator from firing. A bug that doesn't block dispatch can wait.

### Concrete enforcement

- Task #309 ("HIGH-THROUGHPUT DISPATCH agent: parallel GPU orchestration") sat **pending** the entire May 4 race window. That task class is a NON-NEGOTIABLE priority-1 — if a similar task exists in any backlog, it must be claimed before any new validation-gate work.
- Any PR that adds a new ranking/predictor/sanity-gate primitive MUST link to the parallel-actuator file that consumes its output. If no such consumer exists, the PR description must explicitly state "actuator deferred — no race window currently active" with the operator's signoff on the deferral.

## Carmack MVP-first phasing — NON-NEGOTIABLE

**Source:** CASCADE COMPRESSION T3 symposium (`d125af6c3`) revision #2 + APPARATUS META-BUGS T3 symposium §8 cross-symposium consistency + consolidating T3 symposium `council_t3_carmack_mvp_first_elevation_symposium_20260521` + 5 distinct empirical vindication anchors (NSCS06 v6→v7 / ATW V2 byte-mutation / DP1 paired-smoke / VQ-VAE BUILD / CASCADE COMPRESSION 4x cascade) producing ~$4-16 paid GPU + ~$400-800 LOC + 3-4 weeks engineering saved at $0 + ~15 min wall-clock.

Every paid GPU dispatch >$0.30 MUST be preceded by an MVP-first phasing 5-step recipe:

1. **FREE local macOS-CPU smoke first** — every paid GPU dispatch >$0.30 MUST be preceded by an empirical anchor at $0 cost on the smallest faithful local-CPU surface that exercises the cargo-culted assumption.
2. **The smoke MUST falsifiably challenge the cargo-cult** — predict a measurable signature (e.g., zscore, KL, ΔS band) that distinguishes the cargo-cult assumption from the alternative; refusal verdict at empirical residual > 2σ.
3. **Emit canonical equation anchor + Catalog #344 reference** — anchor the smoke result against a registered canonical equation; FORMALIZATION_PENDING waiver if the equation is not yet registered.
4. **Land verdict in same commit batch** as the smoke landing memo — supersession marker on parent design memo per CLAUDE.md "Sister-supersession respect" non-negotiable (or `# NO_SUPERSESSION_NEEDED:<rationale>` waiver).
5. **Re-route operator priority queue** within ~1h of empirical landing per CLAUDE.md "Downstream-surface latency discipline" non-negotiable (or `# DOWNSTREAM_SURFACE_LATENCY_PENDING_OK:<rationale>` waiver).

Burden-of-proof shifts to "why we would NOT free-smoke-first?" — any cascade design memo proposing paid-dispatch-first MUST carry explicit `# PAID_DISPATCH_FIRST_WAIVED:<rationale>` waiver with substantive non-placeholder rationale (placeholder rejected per Catalog #287 sister discipline).

Sister of CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" (when leaderboard moves, rigor compresses; MVP-first phasing is the rigor cadence DURING non-race-mode); together they bind:
- PRE-leader-shift: MVP-first phasing (rigor maximizes information per paid dollar)
- POST-leader-shift: parallel-dispatch (velocity maximizes shipping rate)

Cross-reference: CASCADE COMPRESSION T3 symposium `d125af6c3` rev #2 + APPARATUS META-BUGS T3 symposium §12.5 + consolidating T3 symposium `council_t3_carmack_mvp_first_elevation_symposium_20260521`.

## Results must become system intelligence — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator standing directive 2026-05-22: "Instead of doing ad hoc
analysis against the results, canonicalize and model and wire up and integrate
and make the whole system smarter and more autonomous and dynamic."

Every result is feedback for the solver, not a standalone explanation. A useful
analysis must usually leave behind a reusable surface: a typed ledger row,
canonical schema, posterior/model update, acquisition rule, bit allocator hook,
sensitivity-map contribution, cathedral/autopilot consumer, probe-disambiguator,
planner/actuator wiring change, or regression guard. Markdown prose alone is
acceptable only as a dated bridge artifact that names the missing code/schema
integration and the consumer that should absorb it next.

The contest scoring function is always part of the model: rate term, SegNet
term, PoseNet term, eval loader, batching behavior, CPU/CUDA/MLX axis drift,
hardware substrate, raw-output custody, archive grammar, and exact inflate
runtime are coupled. Do not optimize bytes, frames, pairs, pixels, tensors,
latents, or configs as independent ad hoc knobs. Model their component-specific
marginal effects and feed those marginals into the next planner, sweep, dispatch
ranker, or stop/continue rule.

This rule complements UNIQUE-AND-COMPLETE-PER-METHOD: canonicalization is for
preserving signal and increasing autonomy, not for forcing every substrate
through the same helper when that suppresses score. Use, extend, or fork the
canonical surface according to the method's math and measured behavior, then
wire the result into the autonomous loop so the next agent inherits the improved
system rather than a chat-only insight.

Production standard: abstractions must be reusable, composable, testable,
fail-closed on false authority, and clear enough for OSS review. A result that
does not make the system smarter is incomplete until it is encoded, integrated,
or explicitly tagged `research_only=true` with a concrete integration blocker.

## Long-burn score-lowering campaign default — NON-NEGOTIABLE, HIGHEST EMPHASIS

When the operator says aggressive score lowering, no meat left on the bone, no
holds barred, funded reproduction, no budget/time limit, or escape the HNeRV
local minimum, do not answer with another research-only council loop. Convert
each plausible floor-breaking family into a campaign unless a dated ledger
records a real blocker.

A campaign must include:

1. `lane_id` plus dispatch-claim plan.
2. Source evidence and score-lowering hypothesis.
3. Timing-smoke command that measures seconds/epoch or seconds/candidate.
4. Full-run command with resumable checkpoints and harvest path.
5. Live provider rate/cost model.
6. Byte-closed archive/export/inflate plan for promotion.
7. Stop/continue thresholds for smoke, mid-stage, export, and exact eval.

Budget uncertainty is not a reason to defer. If cost is unknown, run or prepare
the smallest faithful timing smoke so cost becomes measured GPU-hours. Missing
final archive grammar blocks promotion and score claims; it does not block a
clearly tagged non-promotional timing smoke, source-faithful reproduction probe,
or campaign wiring pass. Older `$` caps, no-GPU notes, or no-dispatch memos are
superseded by a newer explicit operator directive to fund or launch a named
campaign, while claim lifecycle, provider import probes, artifact custody,
contest compliance, and CPU/CUDA axis separation remain mandatory.

Visible high-EV directions such as PR95/HNeRV reproduction, NeRV-family
replacements, SIREN/FINER/WIRE/BACON, Ballé/CompressAI, Cool-Chic/C3, wavelets,
RAFT/ego-motion, LA-pose/telescopic foveation, SABOR, S2SBS, arithmetic/range/
ANS compiler passes, and scorer-inverse representations must become either a
campaign ledger plus timing-smoke/launch decision in the same session or an
explicit blocker. `research_only=true` is not a resting state for frontier work:
if the signal is promising, name the next byte-closed prototype or campaign
gate.

If `.omx/state/RACE_MODE_ACTIVE.flag` exists, campaign actuation outranks new
grand-council text unless that text directly writes launchable commands, hardens
the actuator, or records the blocker preventing spend.

## Canonical leaderboard binding-depth discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS

**CANONICAL RENAME 2026-05-30 per operator binding META-correction** `[[pr-or-greater-parity-synergy-binding-integration-not-hnerv-specific-meta-class-lesson-correction]]` + `[[claude-md-hardening-streamlining-de-anchor-local-minima-language-canonical-rename-wave]]`. Prior section name "HNeRV / leaderboard-implementation parity discipline" anchored cognitive frame to HNeRV architecture; perpetuated the 0.196-0.199 plateau cluster (30+ substrates built as HNeRV variants in disguise per 18-shared-assumption empirical audit `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`). Canonical rename de-anchors to **binding-depth META principle**: substrate quality is canonical-leaderboard-PR-or-greater binding-depth across L1-L32 canonical techniques validated across PR95/100/101/102/103/106/110 GOLD/SILVER/BRONZE winners, NOT membership in a specific architecture class. Historical "HNeRV parity discipline lesson X" references throughout CLAUDE.md catalog rows + canonical equation prefixes `pr95_family_l<N>_*_v1` + sister memory are HISTORICAL_PROVENANCE per Catalog #110/#113 (preserved). Forward-looking framing per operator binding 2026-05-30 *"the parity lessons are not hnerv parity, they are PR parity or greater in terms of thinking of synergy and binding and integration"*: **canonical leaderboard binding-depth discipline**.

**Source:** `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` (operator-mandated retrospective, 2026-05-09). Cross-refs `.omx/research/representation_integration_gap_audit_20260508_codex.md` (codex parallel finding) + `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md` (claude-side framing) + `feedback_grand_council_fields_medal_phase2_floor_REBASELINE_with_integration_discipline_20260509.md` (Phase 2 floor rebaseline).

**The 2026-04-30 → 2026-05-04 retrospective:** we had HNeRV/MNeRV/NeRV/SIREN/Cool-Chic/C3 representation primitives in the repo BEFORE PR #95/#100/#101/#103 ever hit the contest. We never got sub-0.20 with them. The leaderboard HNeRV-family won not because of architectural novelty but because PR #100's hnerv_lc_v2 (268 LOC) bound architecture + score-aware training + archive grammar + inflate runtime + export contract simultaneously, and PR #101 (337 additional LOC of entropy bolt-ons) won gold at 0.193 by stacking on the verified substrate. Each layer was reviewable in 30 seconds.

Our internal NeRV/HNeRV/Cool-Chic/C3 work had every architectural ingredient but never bound them simultaneously. The integration loop was always 5-7 separate research artifacts that never converged into a single packet. Lane 12 NeRV mask codec targeted the WRONG slot (mask only, not full RGB renderer). Cool-Chic / C3 hit the FP4A export gate AFTER training, not before.

### The 13 inviolable lessons

Every representation/codec lane (NeRV / HNeRV / Cool-Chic / C3 / wavelet / VQ-VAE / grayscale-LUT / SIREN / coordinate-MLP / hyperprior / nonlinear-transform-coding / time-varying-FiLM / shared-codebook / etc.) MUST honor all 13 of these from byte zero:

1. **Substrate must be score-aware.** Train against the contest's actual `upstream/videos/0.mkv` with gradient-through-SegNet/PoseNet, not extracted masks, not L²/KL on raw frames, not synthetic data. Default loss `((mask_pred - mask_gt) ** 2).mean()` is FORBIDDEN as the primary training signal for any representation entering the archive.
2. **Export-first design.** Declare the archive grammar + parser-section manifest BEFORE writing the training script. If the variant cannot export into the contest packet format (e.g., Cool-Chic / C3 non-FP4A), the run is research-only by construction; tag `research_only=true` and ungate `--auth-eval-on-best` only after the export contract lands.
3. **Archive grammar = monolithic single-file `0.bin`** (or explicitly justified multi-file). Fixed offsets declared in `codec.py` source (e.g., `DECODER_BLOB_LEN = 162_164`, `LATENT_BLOB_LEN = 15_387`). ZIP-member-budget rows are invalid unless the packet really has separate ZIP members.
4. **Inflate.py ≤ 100 LOC** (default budget; explicit waiver for ≤ 200 with rationale). ≤ 2 external dependencies declared in the runtime tree. CUDA-or-CPU agnostic. Reviewable in 30 seconds.
5. **Architecture must be the FULL renderer** (RGB out), not a single-component slot (mask only / pose only). The contest scorer derives masks from frames; replacing the masks slot is dominated by replacing the frames the masks are derived from. Lane 12-style "mask codec only" lanes are DEFERRED-pending-research-with-renderer-rescope.
6. **Score-domain Lagrangian** (not weight-domain proxies like rel_err²). The Lagrangian must be `α·B(θ)/N + β·d_seg(θ) + γ·√d_pose(θ)` with `d_seg` and `d_pose` computed via the actual scorer (or a Hinton-distilled co-trained surrogate per Phase 2 / Phase 3). rel_err²-as-objective is FALSIFIED at rms ≥ 0.04 per `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508.md`.
7. **Bolt-on size ≤ 350 LOC** (substrate engineering may exceed; tag `lane_class=substrate_engineering` explicitly). Substrate engineering happens ONCE per architecture class; bolt-ons happen many times. PR101 was 605 total LOC = 268 substrate + 337 bolt-on. The kitchen_sink anti-pattern (PR105: 1776 LOC, 21 files, LOST to rem2's 241 LOC silver) is what happens when you violate this.
8. **Eval-roundtrip-aware and differentiable scorer-preprocess training.** The uint8 bottleneck (384 → 874 → uint8 → 384) MUST be simulated in the proxy loss. `eval_roundtrip=False` produces 2-11x proxy-auth gap and is FORBIDDEN per existing CLAUDE.md non-negotiable. The scorer preprocess must also be gradient-reachable: PR #95/#106 monkey-patched `rgb_to_yuv6` because the upstream challenge helper is `@torch.no_grad()` / in-place and otherwise severs PoseNet gradients. New NeRV/HNeRV/Cool-Chic/C3 renderer trainers need a PoseNet/SegNet gradient-reachability check before GPU dispatch.
9. **Runtime closure.** Run the exact contest `inflate.sh` signature in a clean environment BEFORE dispatch. Dependency closure failures (missing brotli, wrong wrapper signatures, hidden sidecars, local paths, CPU/CUDA mismatches) are runtime blockers, not method negatives. PR106 belt_and_suspenders FAILED its first replay due to missing `brotli` — exactly this bug class.
10. **Mask/pose coupling gate.** Any mask change requires pose regeneration + geometry diagnostics + decoded mask SHA-256s + mask disagreement record. Smaller mask bytes alone are insufficient.
11. **No-op detector.** Prove the targeted bytes changed AND were consumed by inflate. Reuse, decode/re-encode, provenance-only changes, and cosmetic ZIP repacks stay forensic until this proof exists.
12. **Single-LOC-per-LOC review discipline.** Every line in the bolt-on must be reviewable in 30 seconds. PR101's `codec.py` is 480 LOC of pure codec code (no training scaffold, no profile dispatch, no smoke/full mode flags). Our internal `nerv_mask_codec.py` is 1000+ LOC and includes coordinate sampling + training scaffolds + sample components + magic-byte versioning + ... — NOT a packetized codec.
13. **KILL/FALSIFIED is LAST RESORT.** Per the existing CLAUDE.md non-negotiable: if a representation lane returns negative, the default verdict is DEFERRED-pending-research-with-XYZ-applied with reactivation criteria, not KILLED. Lane 12 NeRV is DEFERRED-pending-renderer-rescope; Cool-Chic / C3 are DEFERRED-pending-export-design.

### Lessons L14–L32 — Canonical leaderboard-winning techniques validated across PR95/100/101/102/103/106/110 (APPENDED 2026-05-28; CANONICAL RENAME 2026-05-30 from "PR95-family" → "canonical leaderboard-winning" per `[[pr-or-greater-parity-synergy-binding-integration-not-hnerv-specific-meta-class-lesson-correction]]`; HISTORICAL canonical equation prefix `pr95_family_l<N>_*_v1` PRESERVED per Catalog #110/#113 APPEND-ONLY)

**Source:** Wave N+47 lesson-set expansion deep-research subagent ITEM 2 of operator-bound 7-cascade per `feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md` + `feedback_prioritization_metric_hygiene_vs_frontier_breaking_orthogonal_plus_13_lessons_incomplete_20260528.md` operator correction #1 verbatim *"theer are more lessons probably that we haven't identified or canonicalized yet"*. **The Wave N+41 audit's 11/13 + 12/13 hygiene-EV scores in TOP-5 are LOWER BOUNDS on the true PR-95-parity gap**; a substrate at 11/13 may still fail L14-L32. Wave N+48 audit re-runs against expanded lesson set.

Each lesson L14-L32 mined from `experiments/results/public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.md` (PR95 source) + `experiments/results/public_pr100_intake_20260504_codex/source/submissions/hnerv_lc_v2/{hnerv_model.py,inflate.py,schema.py,sidecar.py}` (PR100 substrate) + `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/{src/codec.py,src/model.py,inflate.py,inflate.sh,README.md}` (PR101 GOLD bolt-on) + `experiments/results/public_pr103_intake_20260504_codex/source/submissions/hnerv_lc_ac/inflate.py` (PR103 SILVER). Each lesson registered as canonical equation + sister canonical anti-pattern per Catalog #344 (canonical equations + anti-patterns registries; registry size 88 → 107 equations + 17 → 36 anti-patterns).

**HARD-EARNED-vs-CARGO-CULTED classification per Catalog #292 + Catalog #303**: every lesson L14-L32 below is **HARD-EARNED** (cite verifiable source evidence — specific PR + line range). Sister pre-L14+ cargo-culted variations registered as `pre_pr95_family_l<N>_*_v1` anti-patterns; each carries canonical_unwind_path = "adopt PR95-family L<N>".

14. **PR95 8-stage 29,650-epoch training curriculum.** PR95 winning training schedule is 8 stages totaling 29,650 epochs: stage1=CE (3k ep) → stage2=tau_softplus (5.65k) → stage3=smooth (1.5k) → stage4=QAT (500) → stage5=C1a-L7 (9k) → stage6=lambda_sweep (2k) → stage7=sigma_sweep (3k) → stage8=muon_finetune (5k). Each stage carries (loss_form, learning_rate, qat_active, c1a_lambda, sigma) tuple per `profile_pr95_hnerv_muon_intake.md` verbatim. Canonical equation `pr95_family_l14_eight_stage_29650_epoch_curriculum_v1`.

15. **Muon optimizer in final stage only (177K of 229K params).** PR95 stage 8 fine-tune ONLY uses Muon (Jordan-Bernstein 2024 momentum-based orthogonalized SGD via Newton-Schulz iteration); 177,156 of 228,958 decoder params under Muon (77%); remaining 51,802 under AdamW; stages 1-7 all AdamW. Per-param-group optimizer assignment is canonical. Canonical equation `pr95_family_l15_muon_optimizer_final_stage_only_v1`. Sister anti-pattern: AdamW-only training.

16. **C1a coder-aware regularization weight schedule (lambda 0.01 → 0.02).** PR95 adds C1a coder-aware regularization to decoder weight loss starting stage 5 (lambda=0.01) and sweeping to lambda=0.02 in stages 6-8. C1a is a structural prior that biases decoder weights toward brotli-friendly distributions. Canonical equation `pr95_family_l16_c1a_coder_aware_regularization_v1`. Sister anti-pattern: no coder-aware regularizer.

17. **Sigma noise injection schedule (0.2 → 0.1).** PR95 injects Gaussian noise sigma=0.2 in stages 1-6, sweeping to sigma=0.1 in stages 7-8. Structural regularizer that simulates uint8 quantization roundtrip during training; sister of eval_roundtrip=True CLAUDE.md non-negotiable. Canonical equation `pr95_family_l17_sigma_noise_schedule_v1`.

18. **PixelShuffle + bilinear-skip + sin activation decoder architecture.** PR95/PR100/PR101 decoder pattern: per-stage block = Conv(in, out*4, 3x3) + PixelShuffle(2) + bilinear-skip + sin activation; 6 upsample stages from 6x8 to 384x512; channel taper `[C, C, C, 0.75C, 0.58C, 0.5C, 0.5C]`. NeRF-style sin activation avoids dead-ReLU regions for single-video memorization; PixelShuffle is bandwidth-efficient at constant FLOP vs interpolate+conv; bilinear-skip stabilizes training. Canonical equation `pr95_family_l18_pixelshuffle_bilinear_skip_sin_v1`. Sister anti-pattern: ReLU + ConvTranspose2d (deadness + checkerboard artifacts).

19. **Per-frame-PAIR latent 28-d predicting 2 frames per latent.** PR95-family decodes 2 frames from each 28-d latent (frame_0 + frame_1 = PAIR); 600 latents = 1200 frames; ~94% of archive bytes are decoder weights vs ~6% per-pair latents. Per-pair structure exploits temporal redundancy in dashcam video. Canonical equation `pr95_family_l19_per_frame_pair_latent_28d_v1`. Sister anti-pattern: per-frame latent (2× storage with ~0 quality improvement).

20. **Monolithic single-file 0.bin archive grammar with 4 length-prefixed sections.** PR100/PR101 archive = single 0.bin file with 4 length-prefixed sections: `(u32 dec_len, dec_blob brotli) + (u32 sca_len, sca_blob fp16) + (u32 lat_len, lat_blob brotli) + (u32 wrp_len, wrp_blob brotli)`. `parse_archive(bytes) → (dec, sca, lat, wrp)` is ~10 LOC. Canonical equation `pr95_family_l20_monolithic_4_section_archive_grammar_v1`. Sister anti-pattern: multi-file ZIP with separate decoder/latents/sidecar members (per-member ZIP overhead ~150 bytes).

21. **Per-tensor byte-maps for entropy-friendly coding.** PR101 codec declares `DECODER_BYTE_MAPS = {tensor_idx → map_name}` where map ∈ {`zig`, `negzig`, `twos`, `off`}; `negzig` = -zigzag, `twos` = signed reinterpret, `off` = unsigned - 128, `zig` = standard zigzag; selected per-tensor based on weight distribution post-quantization. Canonical equation `pr95_family_l21_per_tensor_byte_maps_v1`. Sister anti-pattern: single zigzag applied uniformly.

22. **CONV4_STORAGE_PERMS per-tensor permutation for entropy-friendly storage.** PR101 reorders Conv2d weight axes per-tensor before brotli compression; 13 specific tensors get explicit storage perm (e.g., idx 14 = `(1,0,2,3)`); inverse perm applied at decode. Spatial locality of permuted axes reduces brotli output size by hundreds of bytes. Canonical equation `pr95_family_l22_conv4_storage_perms_v1`. Sister anti-pattern: default `(out_ch, in_ch, kh, kw)` axis order.

23. **Split brotli streams with explicit DECODER_STREAM_ENDS partition.** PR101 splits decoder weights into 7 separate brotli streams (not one big stream): `DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)`; each stream decoded independently. Stream-boundary aware grouping by tensor-distribution similarity saves bytes vs single stream. Canonical equation `pr95_family_l23_split_brotli_streams_v1`. Sister anti-pattern: single brotli.compress() for entire decoder weight blob.

24. **Raw LZMA latent coding (FORMAT_RAW + FILTER_LZMA1).** PR101 latents use `lzma.FORMAT_RAW + [FILTER_LZMA1 with dict_size=4096, lc=3, lp=0, pb=0]` instead of standard .xz format; saves bytes by stripping format headers; lzma compresses temporal-delta uint8 latent codes well. Final latent blob = 15,387 bytes. Canonical equation `pr95_family_l24_raw_lzma_latent_coding_v1`. Sister anti-pattern: lzma.FORMAT_XZ default OR brotli for latents.

25. **Temporal-delta uint8 latent coding with prefix-sum decode.** PR100/PR101 store latents as temporal deltas: `lat[i] = lat[i-1] + delta[i]` (centered uint8); prefix-sum reconstructs sequence; exploits temporal smoothness of dashcam video; reduces entropy vs raw float storage. Canonical equation `pr95_family_l25_temporal_delta_uint8_latent_v1`. Sister anti-pattern: raw fp16 latent dump.

26. **Canonical Huffman length-vector ranked sidecar (Wang-Rudin discipline).** PR101 sidecar uses canonical Huffman codes with length-vector RANK encoding: instead of storing Huffman tree, store rank of length-vector among all Kraft-valid vectors (per Wang & Rudin 2015 "Falling Rule Lists" canonical encoding); saves bytes vs explicit tree encoding. Sidecar canonical-Huffman-enum length = 607 bytes. Canonical equation `pr95_family_l26_canonical_huffman_length_ranked_sidecar_v1`.

27. **Per-pair single-dim latent correction sidecar (255-sentinel no-op).** PR100/PR101 add a sidecar of `(u8 dim_idx, i8 delta_quantized)` per pair: `dim_idx=255` means no correction; `delta` scaled by 0.01; ~1.2KB sidecar encodes targeted fine-tune corrections per pair selected to minimize SegNet+PoseNet distortion. Canonical equation `pr95_family_l27_per_pair_single_dim_correction_sidecar_v1`. **This single technique contributes -0.001 to -0.003 score improvement (substrate-ceiling → medal-class jump).** Sister anti-pattern: no sidecar at all.

28. **PR98 decode-side channel postprocess (subtract 1.0 from specific RGB channels).** PR101 inflate.py post-processes decoded frames: subtract 1.0 from frame_0 RED channel, frame_0 BLUE channel, frame_1 GREEN channel; learned during training to compensate for a known scorer bias; +1/+1/+1 cancellation pattern (3 lines at `inflate.py:49-51`). **0 archive bytes, ~-0.0001 to -0.0005 score points.** Canonical equation `pr95_family_l28_decode_side_channel_postprocess_v1`.

29. **fp16 scales per tensor for INT8 dequant.** PR100/PR101 store one fp16 scale per tensor (28 tensors = 56 bytes); `dequant = int8_code.astype(fp32) * fp16_scale`; preserves per-tensor magnitude info while keeping codebook at int8 granularity. Canonical equation `pr95_family_l29_fp16_per_tensor_scales_int8_v1`. Sister anti-pattern: fp32 scales (112 bytes overhead) OR per-layer-group scales (lose per-tensor info).

30. **Range/arithmetic coding via constriction.Categorical for specific tensors (PR103 silver).** PR103 silver-medal substitutes brotli with `constriction.stream.queue.RangeDecoder` + per-tensor Categorical histogram for 8 specific large tensors (`AC_INDICES = [0, 2, 4, 6, 8, 10, 12, 21]`); remaining 20 tensors stay brotli-encoded; `merged_ac_len = 153856 bytes`. Canonical equation `pr95_family_l30_range_arithmetic_coding_categorical_v1`. Sister anti-pattern: brotli-only on high-entropy tensors.

31. **Combinatorial colex rank encoding for no-op positions (Wang-Rudin SLIM-flavor).** PR101 encodes sidecar no-op positions via combinatorial colex rank: instead of storing position bitmap (N_PAIRS/8 bytes) or explicit position list (2×noop_count bytes), store rank of position-subset among `C(N_PAIRS, noop_count)` combinations; per Wang-Rudin SLIM canonical encoding; `SIDECAR_NOOP_INFER_RANK_LEN = 3 bytes`. Canonical equation `pr95_family_l31_combinatorial_colex_rank_noop_v1`.

32. **brotli quality=11 max compression for sidecar.** PR100/PR101 use `brotli.compress(payload, quality=11)` (max) for all sidecar encoding; quality=11 spends ~10× compression time vs quality=6 but saves ~5-10% bytes on small payloads; compression time is offline overhead so quality=11 is free at deploy time. Canonical equation `pr95_family_l32_brotli_quality_11_max_v1`. Sister anti-pattern: brotli default quality=6.

### L14–L32 enforcement

- Each lesson registered as canonical equation in `tac.canonical_equations` registry (queryable via `tools/list_canonical_equations.py --json`) with sister `pre_pr95_family_l<N>_*_v1` anti-pattern in `tac.canonical_anti_patterns` registry. Auto-discovered via `cathedral_equation_lookup_consumer` per Catalog #335 cathedral consumer auto-discovery + Catalog #344 canonical equations memo-reference enforcement.
- Sister `pr95_family_l<N>_*_v1` canonical equation `predicted_vs_empirical_residual` = 0.0 at registration (PR95-family source IS the empirical anchor); auto-recalibration per Catalog #371 triggered when 3+ NEW empirical anchors land from PR111+ training waves.
- **Wave N+48 audit RE-RUN plan**: a sister deep-research subagent re-runs the Wave N+41 substrate-family × PR-95-parity audit (`7f0617d6d`) against the expanded L1-L32 lesson set; substrates currently scored 11/13 or 12/13 may rank LOWER on the L1-L32 baseline. The CORRECTED HYGIENE-EV ranking re-ranks substrate inventory per the canonical 3-metric trichotomy (HYGIENE-EV ⊥ FRONTIER-BREAKING-EV ⊥ HIGHEST-EV-SHORTEST-WC) per `feedback_prioritization_metric_hygiene_vs_frontier_breaking_orthogonal_plus_13_lessons_incomplete_20260528.md` correction #1.
- **Sister 6-hook wire-in per Catalog #125**: hook #1 sensitivity-map = ACTIVE (each canonical equation has per-axis byte savings); hook #2 Pareto constraint = ACTIVE (per-lesson constraint on archive grammar); hook #3 bit-allocator = ACTIVE (L21/L22/L23/L24/L25/L26/L29/L30/L31/L32 are bit-allocator primitives); hook #4 cathedral autopilot dispatch = ACTIVE (canonical_equation_lookup_consumer auto-discovers); hook #5 continual-learning posterior = ACTIVE (canonical equation registry); hook #6 probe-disambiguator = ACTIVE (L14-L32 IS the canonical disambiguator between "PR95-parity-honored substrate" vs "pre-L14+ cargo-cult variation").

### The 8th forbidden pattern (named here)

**Forbidden representation-without-archive-grammar (the "research-substrate trap"):**

Building a representation (NeRV / Cool-Chic / C3 / wavelet / VQ-VAE / grayscale-LUT / SIREN / coordinate-MLP / hyperprior / etc.) WITHOUT simultaneously building (a) the `archive.zip` builder that emits scored bytes, (b) the `inflate.sh` runtime that reads them, (c) the parser-section manifest that locates them, (d) the export contract that converts trained weights → archive bytes, and (e) the score-aware training loop that backprops through SegNet/PoseNet on the contest video — is a research-only path by construction. The bytes never enter the contest packet; the score never moves.

This is the dominant representation-lane integration meta-bug from the 2026-04-30 → 2026-05-04 gap. It does not by itself explain the full miss: the postmortem also requires (a) failure to consume PR #95's open training stack during the race window, (b) failure to measure the CPU public-leaderboard axis early enough, and (c) missing differentiable scorer-preprocess training in our NeRV/HNeRV loops. STRICT preflight check #124 (`check_representation_lane_has_archive_grammar_at_design_time`) enforces the archive-grammar part; trainer-specific grad-reachability guards must cover the scorer-preprocess part.

### Five forbidden code patterns

1. **Forbidden NeRV-style coordinate MLP that targets the masks.mkv slot without rescope to the renderer.** Lane 12 mistake. If your representation's output shape is `(T, H, W, 5)` of mask logits and not `(T, 3, H, W)` of RGB frames, the lane is DEFERRED-pending-renderer-rescope.

2. **Forbidden `--auth-eval-on-best` gate bypass for non-FP4A export variants.** Cool-Chic / C3 mistake. `train_renderer.py:2099-2122` blocks `--auth-eval-on-best` for variants that lack full archive/export support — this is correct fail-closed behavior. NEVER add a workaround that runs auth eval against a non-exportable variant; instead, land the export contract first.

3. **Forbidden `make_synthetic_pair_batch` calls in any non-smoke training path.** Per `feedback_codex_finding_pr101_synthetic_targets_FIXED_20260508.md`. Train against `upstream/videos/0.mkv` decoded via pyav, not random Gaussian noise. Smoke-only mode does not generalize to non-smoke.

4. **Forbidden representation-lane Level 1+ promotion without `archive_grammar` / `parser_section_manifest` / `inflate_runtime_loc_budget` / `runtime_dep_closure` / `export_format` / `score_aware_loss` / `bolt_on_loc_budget` / `no_op_detector_planned` declared in lane-registry evidence.** STRICT preflight check #124.

5. **Forbidden cross-archive composition (HStack/VStack/cross-paradigm) without a single verified [contest-CUDA] substrate anchor.** Per substrate-vs-codec meta-pattern. T9 (cross-archive multi-substrate composition) is the kitchen_sink anti-pattern under a new name. DEFER until a verified composable substrate exists; or re-scope to single-axis branching from the ONE verified score-aware substrate (currently A1).

### Enforcement

- STRICT preflight check #124 `check_representation_lane_has_archive_grammar_at_design_time` lands warn-only initially; flip to STRICT after in-flight Phase 2 lanes (T1/T6/T10/T15/T17/T18) backfill the blueprint.
- `tools/lane_maturity.py` audit refuses to mark a representation lane as Level 1+ without the 8 declared fields.
- Council review of any new representation/codec lane MUST cite this section and walk through all 13 lessons.
- Memory file `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` is the canonical retrospective; future agents should re-read it before starting any new representation lane.

## UNIQUE-AND-COMPLETE-PER-METHOD operating mode — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator retrospective 2026-05-15 verbatim *"this has been a huge problem since the beginning of the competition and prevented us from actually building original implementations because you didn't understand me and didn't understand the domain and problem space well enough at the time but we learned from PR 95 and just learned the same lesson again but across the entire contest and submission"* + standing directive 2026-05-15 *"share what works but when it is stale or obsolete or suppressing signal or otherwise and when the optimal engineering calls for it we want full and complete and correct unique and distinct designs and implementations"*. Anchor memos: `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` + `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` + `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`.

This NUANCES (does NOT replace) the prior consolidate-canonical-helpers directive (see `feedback_consolidate_everything_into_meta_layer_or_canonical_helpers_standing_directive_20260515.md`). Both rules coexist via the falling-rule list below. This section also EXTENDS the substrate-level "HNeRV / leaderboard-implementation parity discipline" to the META infrastructure level: that section taught us bind-all-ingredients per substrate; this section teaches us the same lesson at the canonical-helper / META-layer / shared-engineering surface.

### The empirical anchor (the 0.1928 cluster) <!-- HISTORICAL_SCORE_LITERAL_OK:cluster_label_historical_anchor_2026-05-15_assumptions_challenge_audit -->

The 18-assumption audit (`feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`) empirically established that 90%+ of substrates share 18 structural assumptions (EMA 100% / archive.zip 100% / eval_roundtrip 97% / canonical scorer-preprocess 97% / canonical auth_eval routing 97% / Tier-1 engineering 78-100% / etc.). Variance between substrates IS the variance of the 10% NOT shared. **The 0.196-0.199 cluster IS the local-minimum produced by the shared 90%** — a flat plateau where every "new substrate" is structurally a variation of the SAME implementation under different names.

PR 95 / 100 / 101 / 102 / 103 winners bound ALL ingredients (architecture + score-aware training + archive grammar + inflate runtime + export contract + training curriculum + Tier-1 engineering + scorer routing) into ONE coherent unique implementation reviewable in 30 seconds (PR101 = 268 substrate + 337 bolt-on = 605 LOC total). Internal HNeRV/NeRV/Cool-Chic/C3 work had every ingredient but never bound them — kept trying to share with canonical helpers. Lost.

The same lesson now applies at the META infrastructure level: the canonical-helper-share + META-layer-consolidation reflexes have been suppressing substrate-optimal engineering across the contest. The 270+ catalog gates we added all session ARE part of the problem when they force-fit substrate engineering. PR 95's winners didn't have a 270-catalog META layer constraining them.

### The new default operating mode (binding, supersedes prior reflex)

**Going forward, the default mode for any new substrate / codec / method / composition is UNIQUE-AND-COMPLETE-PER-METHOD.** Not "share canonical and customize where needed." Not "extend the META layer." The default question becomes:

> *"What's the OPTIMAL ENGINEERING for THIS specific method to achieve the lowest score possible given the methods and techniques involved?"*

NOT:

> *"How do I share with the canonical?"*

Canonical helpers are TOOLS available for use when they serve. They are NOT OBLIGATIONS to extend or share with by default.

### The decision criterion (the falling-rule list)

For every shared canonical helper / META-layer field / engineering pattern adoption decision per substrate:

1. **EMPIRICAL** — IF a paired-comparison smoke ($5-15) has been run AND the substrate's score with canonical adoption is measurably worse (ΔS ≥ 0.005) than with unique implementation THEN fork to substrate-specific implementation; document rationale in the substrate's lane registry notes + design memo.
2. **PRINCIPLED** — ELSE IF the canonical's design assumption clearly does NOT fit the substrate's mathematical structure (e.g. canonical scorer-preprocess assumes `x[:, -1, ...]` slicing but the substrate's optimal score requires both frames) THEN fork; document rationale.
3. **UNCLEAR** — ELSE the burden of proof is on PROVING canonical-is-better, not assuming it. Default: fork OR run a paired-comparison smoke to measure.
4. **OBVIOUS-FIT** — ELSE adopt canonical (shared infrastructure value > unmeasured customization value).

### Concrete operating principles

1. **Each new substrate gets a complete focused implementation** — architecture + score-aware loss + archive grammar + inflate runtime + export contract + training curriculum + Tier-1 engineering + scorer routing — all bound into ONE coherent package reviewable in 30 seconds (PR101 model: ~600 LOC total).
2. **Canonical helpers used WHEN they serve** — e.g. commit serializer (no substrate-specific reason to fork), fcntl-locked state writes, Catalog #220 byte-mutation gate. These remain default.
3. **Canonical helpers FORKED WHEN they suppress** — e.g. scorer-preprocess pipeline if the substrate's optimal score requires a different gradient path; auth_eval routing if the substrate needs paired-CPU at smoke time; META layer 36-field contract if the substrate needs different fields. NO ceremony required to fork; document the rationale in the substrate's lane registry notes + design memo.
4. **CLAUDE.md non-negotiables remain across substrates** — MPS noise / eval_roundtrip=False / dead-flag wiring / phantom-score directory / Catalog #220 operational mechanism / etc. These prevent BUG CLASSES that are universal across substrates. They do NOT prevent unique engineering.
5. **The catalog protection layer protects against BUG CLASSES, not engineering preferences.** When a substrate forks a canonical helper, the catalog gates may need to be re-evaluated for that substrate. Use same-line waivers (`# CANONICAL_HELPER_FORKED_OK:<rationale>`) liberally; the gates exist to PREVENT BUGS, not to ENFORCE UNIFORMITY.
6. **The bolt-on vs substrate-engineering split per HNeRV parity discipline lesson 7 is the canonical articulation:** bolt-ons share (≤350 LOC; happens many times); substrate engineering unique-ifies (happens ONCE per architecture class; size budget exceeds). Treating substrate engineering like a bolt-on is the structural mistake.

### Mandatory design memo discipline

For every NEW substrate / codec / composition design from this point forward:

- Default to UNIQUE-AND-COMPLETE-PER-METHOD.
- Canonical helpers used as TOOLS not OBLIGATIONS.
- **Document a "## Canonical-vs-unique decision per layer" section in every design memo** that lists each adopted canonical helper / META layer field / engineering pattern with its rationale per the falling-rule above. Sister to the existing 6-hook wire-in declaration per Catalog #125.
- Accept the higher LOC cost as the price of breakthrough.
- Bind ALL ingredients (architecture + training + grammar + runtime + export + score-aware + Tier-1) into ONE coherent package.
- Bolt-on size budget (≤350 LOC) applies to BOLT-ONS; substrate engineering exceeds it explicitly per HNeRV parity L7.

### Concrete enforcement

- STRICT preflight Catalog #290 (`check_substrate_design_memo_has_canonical_vs_unique_decision_section`) refuses substrate scaffold landing memos dated >= 2026-05-15 that lack the literal section header. Initial wire-in is WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule"; strict-flip planned after the in-flight + just-completed scaffolds backfill.
- The canonical example of UNIQUE-AND-COMPLETE-PER-METHOD bind-all-ingredients is Carmack-Hotz Strip-Everything (`feedback_grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515.md` composite #4) — should be ELEVATED from the deferred queue.
- The 5 in-flight subagents spawned this turn with canonical-helper-defaults instructions (STC-DASHER, ATW-CODEC, WUNDERKIND-G1, U-DIE-KL, ASSUMPTIONS-CHALLENGE-AUDIT) need follow-up review per `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`: for each canonical helper they adopted, was that the OPTIMAL ENGINEERING for THIS method, or the path-of-least-resistance? Where path-of-least-resistance won, the substrate may have been suppressed.

### Cross-references

- HNeRV / leaderboard-implementation parity discipline (substrate-level PR 95 lesson; this section is the META-level extension).
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` — the principle.
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` — the historical depth + retrospective acknowledgment.
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` — the empirical anchor (18 shared assumptions matrix + 10 NSCS substrate-class shifts + top-stack composition matrix).
- `feedback_knowledge_preservation_pr95_meta_level_lesson_landed_20260515.md` — this gate's landing memo (canonical structural protection).

## 18-shared-assumption profile registration discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** D4 of the canonical rename + apparatus hardening wave 2026-05-30 (D1+D2 inline landing memo `feedback_claude_md_canonical_rename_wave_d1_d2_inline_landing_20260530.md` D3-D12 queue + operator approval 2026-05-30 verbatim *"All are approved, land inline"*). Empirical anchor: `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (the 18-shared-assumption matrix that empirically established 90%+ of substrates share 18 structural assumptions — EMA 100% / archive.zip 100% / eval_roundtrip 97% / canonical scorer-preprocess 97% / canonical auth_eval routing 97% / Tier-1 engineering 78-100% / etc.). The 0.196-0.199 cluster IS the local-minimum produced by the shared 90%: a flat plateau where every "new substrate" is structurally a variation of the SAME implementation under different names.

This NUANCES (does NOT replace) the "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable above. That section operates at the **per-layer** surface (Catalog #290 `## Canonical-vs-unique decision per layer` design-memo section). This D4 section operates at the orthogonal **per-assumption** surface: the 18 specific shared assumptions a substrate inherits by default are NOT the same as the canonical helpers / META-layer fields it adopts per layer. A substrate can adopt only substrate-optimal canonical helpers per Catalog #290 AND STILL silently inherit all 18 shared assumptions (because the assumptions are pre-architectural reflexes — they are the BACKDROP per the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable, not per-layer engineering choices).

### The rule

Every NEW substrate design memo (`.omx/research/*_design_<YYYYMMDD>.md`) MUST declare a per-substrate **18-shared-assumption profile** that classifies each of the 18 shared assumptions per the canonical audit matrix. Each assumption gets exactly one classification from the canonical 4-value falling-rule list (the same falling-rule structure as Catalog #290's canonical-vs-unique decision):

1. **ADOPT_CANONICAL** — the assumption is HARD-EARNED for THIS substrate (cite empirical or first-principles evidence that the assumption serves the substrate's optimal score). The default for most assumptions; but the classification MUST be explicit, not silent.
2. **FORK_PRINCIPLED** — the assumption's design premise clearly does NOT fit the substrate's mathematical structure (e.g. `eval_roundtrip` uint8 simulation assumes a specific quantization path the substrate does not use). Document the principled mismatch.
3. **FORK_EMPIRICAL** — a paired-comparison smoke (or post-training Tier-C measurement per Catalog #324) has empirically shown the substrate's score is measurably worse (ΔS ≥ 0.005) under the canonical assumption than under the forked alternative. Cite the empirical artifact.
4. **UNCLEAR_NEEDS_EMPIRICAL** — the assumption's fit is unproven for this substrate; the burden of proof is on PROVING canonical-is-better, not assuming it. Queue the disambiguating probe per Catalog #313.

The 18 assumptions are enumerated in the canonical audit matrix at `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`. A design memo that silently inherits an assumption without classifying it is the structural cause of the plateau cluster — the assumption-backdrop blindness the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable extincts at the session cadence surface; this D4 discipline extincts it at the per-substrate-design-memo surface.

### Sister relationship to existing surfaces

- Sister of **UNIQUE-AND-COMPLETE-PER-METHOD operating mode** (Catalog #290 per-LAYER canonical-vs-unique surface; D4 is the per-ASSUMPTION surface — orthogonal axes, both required).
- Sister of **META-ASSUMPTION ADVERSARIAL REVIEW** (Catalog #291 per-SESSION cadence + Catalog #292 per-DELIBERATION assumption surfacing; D4 is the per-DESIGN-MEMO surface).
- Sister of the **9-dimension success checklist evidence** (Catalog #294) + **cargo-cult audit per assumption** (Catalog #303) — both are per-design-memo discipline sections; D4 adds the per-assumption 18-profile dimension.

### Concrete enforcement

- The planned STRICT preflight gate **`check_substrate_design_memo_registers_18_assumption_profile`** (D5; still QUEUED per the cap=1-per-turn anti-pattern — NOT landed in this batch) will refuse repo-local `.omx/research/*_design_<YYYYMMDD>.md` substrate design memos dated after its strict-flip cutoff that lack the literal section header `## 18-shared-assumption profile` (case-insensitive) with all 18 assumptions classified per the 4-value falling-rule above. Until D5 lands, this discipline is enforced at council-review + adversarial-review time per the META-ASSUMPTION non-negotiable.
- The classification table is the operator-facing audit surface: a reviewer can scan the per-assumption profile and immediately see whether the substrate is a genuine class-shift (multiple FORK_PRINCIPLED / FORK_EMPIRICAL classifications on score-relevant assumptions) or a plateau-adjacent variation (all ADOPT_CANONICAL).

### Cross-references

- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` — the canonical 18-assumption matrix (the empirical anchor + the enumeration).
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` — the share-vs-fork principle this discipline operationalizes per-assumption.
- "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (above) — the per-layer sister surface.
- "META-ASSUMPTION ADVERSARIAL REVIEW" — the per-session + per-deliberation sister surfaces.
- Catalog #290 (`check_substrate_design_memo_has_canonical_vs_unique_decision_section`) — the per-layer STRICT gate D5 is modeled on.

## Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** D1 R3 dispatch 2026-05-14 (`feedback_d1_dispatch_phase2_harvested_20260514.md`) — call_id `fc-01KRKABYAC9C6MA161NKSGH9PY`, $0.15 Modal T4 smoke. The D1 substrate landed as L1 SCAFFOLD with `impl_complete=true` and a 43 KB sidecar; predicted contest-CPU score band was `[0.181, 0.188]` from the deep-math memo §10 D1 polytope theorem. R3 produced score ~0.222 — outside the band by +0.040 — because the L1 SCAFFOLD wired the encoder + archive grammar + runtime custody check but DEFERRED the per-pixel polytope-interior noise overlay to "L2 INTEGRATION" via Verdict V3. The sidecar bytes structurally consumed for the no-op detector (Catalog #105 / #139) but produced NO frame changes, so the rate-axis cost (+0.029) landed without compensating Δseg savings. Net effect: the lane was a **research-substrate trap** per HNeRV parity discipline lesson 2 + the 8th forbidden pattern.

This is the same anti-pattern that bit the 2026-04-30 → 2026-05-04 representation-integration gap: every architectural ingredient present, none bound simultaneously. The CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" section already declares lesson 2 (**Export-first design** — declare the archive grammar + parser-section manifest BEFORE writing the training script; tag `research_only=true` if the variant cannot export). This new non-negotiable extends that discipline to RUNTIME EFFECT: not just "exports correctly" but "operationally consumes the exported bytes for score improvement at inflate time."

### The rule

Every L1+ substrate lane that ADDS bytes to a composition archive (>1 KB) MUST satisfy ONE of the following from the moment its bytes appear in the lane registry:

1. **OPERATIONAL mechanism declared.** The inflate runtime MUST contain a non-trivial code path that READS the sidecar bytes AND USES them to modify the rendered frames (e.g. per-pixel noise overlay applied to frame_1 RGB at camera resolution, per-pair residual added to base render, score-aware quantization unpacking, etc.). The lane registry notes / gate evidence MUST carry one of: `score_improvement_mechanism_status=OPERATIONAL`, `l2_integration_landed`, `l2_overlay_active`, `runtime_overlay_consumed=true`, `operational_overlay=true`, `operational_consumption=true`.

2. **`research_only=true`** declared in lane registry notes. Per HNeRV parity L2: the lane is explicitly research-substrate (no contest-CUDA exact eval dispatch will fire because `ready_for_exact_eval_dispatch=false`). The bytes may sit dormant pending L2 integration; the lane is honest about being scaffold-only.

3. **Pre-build substrate-engineering scaffold.** The lane declares BOTH `lane_class=substrate_engineering` AND a pre-build gate signal (one of: `_full_main raises NotImplementedError`, `full_path_council_gated`, `scaffold_only`, `pre_build_substrate_engineering`). The substrate trainer's full path is council-gated so no $1+ Modal dispatch can fire until the design verdicts are adjudicated.

4. **Same-line waiver** `# SCAFFOLD_DEFERRED_INTEGRATION_OK:<rationale>` for the rare deliberate operator-approved L1 dispatch (e.g. empirical proxy-vs-actual band check). The rationale must be a real string (placeholder `<reason>` literal is rejected).

### Forbidden anti-pattern (the "L1 scaffold + sidecar bytes without overlay" trap)

A substrate lane that lands `impl_complete=true` at L1 with byte addition >1 KB AND no operational mechanism AND no `research_only=true` tag AND no pre-build substrate_engineering opt-out AND no rationale waiver is FORBIDDEN. The lane should be marked `research_only=true` or split into two lanes: (a) the design-time scaffold (research_only) and (b) the integration lane that wires the runtime effect.

### Concrete enforcement

- STRICT preflight Catalog #220 (`check_substrate_l1_scaffold_no_byte_addition_without_operational_score_improvement_mechanism`) — refuses any state of the lane registry containing an in-scope substrate lane with byte addition >1 KB but no acceptance signal. STRICT-from-byte-one per the "Strict-flip atomicity rule" (D1's L2 INTEGRATION lands in the same commit batch, driving live count to 0).
- Lane registry hygiene: every substrate lane's `impl_complete` evidence MUST declare `archive_bytes_added` + `score_improvement_mechanism_status` explicitly. D1's canonical evidence: `archive_bytes_added=~43 KB (full) or ~2.7 KB (shrunk); score_improvement_mechanism_status=OPERATIONAL via tac.substrates.d1_segnet_margin_polytope.overlay.apply_l2_overlay_for_video_list; runtime_overlay_consumed=true; l2_integration_landed=true`.
- Companion runtime: a substrate that lands its L1 SCAFFOLD MUST simultaneously land the L2 OVERLAY CONSUMER unless `research_only=true`. The runtime effect lives in `tac.substrates.<id>.overlay` (or sister inflate-side helper); the inflate.py must invoke it after the base substrate's `main(argv)` returns.

### Cross-reference

Sister of HNeRV parity discipline lesson 2 (export-first design) and Catalog #105 / #139 (no-op detector + structural consumption proof). HNeRV parity #2 says "if you can't export, tag research_only"; Catalog #220 says "if you exported but the bytes don't produce frame changes, ALSO tag research_only or fix the overlay." Together they extinct the 8th forbidden pattern (research-substrate trap) at TWO surfaces: design-time + runtime-effect.

## Substrate MUST be at OPTIMAL FORM before paid empirical dispatch — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator directive 2026-05-17 + cumulative lessons from the
2026-05-13 → 2026-05-17 substrate dispatch waves. Anchor memos:
`.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md`
(NSCS06 v6→v7 44% improvement via cargo-cult-unwind methodology),
`.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md`
(4 distinguishing-feature dispatch failures), `feedback_meta_framing_correction_optimal_form_before_paid_dispatch_landed_20260517.md`
(this section's landing). Sister of HNeRV parity discipline + UNIQUE-AND-COMPLETE-PER-METHOD operating mode + 9-dimension success checklist evidence + Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY.

### The structural failure this rule extincts

Across the 2026-05-13 → 2026-05-17 dispatch waves we **empirically
dispatched substrates at LIFTED-TRAINER form when operator's standing
directives required OPTIMAL FORM at implementation**. The pattern, with
empirical receipts:

- **NSCS06 v6 → v7 = 44% improvement** (105.15 → 58.89, contest-CUDA)
  achieved in ONE iteration via cargo-cult-unwind methodology
  (`.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md`).
  NSCS06 v6 was the ONLY substrate that got iterated to optimal form
  before the next paid dispatch wave.
- **4-of-5 distinguishing-feature dispatch failures** this session
  (Wunderkind G1 v2 reducer, ATW v2 D4 cooperative-receiver, Z6 FiLM
  ego-motion, NSCS01 nullspace-split, NSCS06 v8 Path B) were
  empirically tested at **lifted-trainer form**, falsifying SPECIFIC
  IMPLEMENTATIONS not the novel concepts. The implementations sat at
  PROCEED_WITH_REVISIONS council verdicts but were dispatched anyway
  for an "empirical anchor."
- **52+ other substrates** were lifted to LIFTED-TRAINER form
  (passes tests + has PR95-paradigm tokens) but NEVER ITERATED to
  optimal form before either dispatch or retirement.

The design-memo discipline gates (Catalog #290 canonical-vs-unique
per layer / #294 9-dim checklist / #303 cargo-cult audit / #305
observability surface) were enforced at the **memo surface** but
silently BYPASSED at the **implementation + dispatch surface**: a
sextet council could return `PROCEED_WITH_REVISIONS`, the council
memo could land with full v2 frontmatter per Catalog #300, and the
substrate trainer could still receive a paid Modal/Lightning/Vast.ai
dispatch the next hour because no STRICT gate bound the council
verdict back to the dispatch decision.

### Definitions

**OPTIMAL FORM**: substrate state where ALL of the following hold:
1. **Cargo-cult-unwind methodology applied** per the NSCS06 v6→v7
   pattern: enumerate the cargo-culted assumptions (per Catalog #303),
   apply the unwind path per assumption, and document the new
   implementation against the cargo-cult audit section.
2. **9-dim checklist satisfied AT IMPLEMENTATION** (not just declared
   in the design memo) per Catalog #294: uniqueness / beauty / distinctness
   / rigor / per-method optimization / stack-of-stacks composability /
   deterministic reproducibility / extreme optimization / optimal minimal
   contest score.
3. **Sextet / grand council returned PROCEED-unconditional** (not
   `PROCEED_WITH_REVISIONS`) on the iterated form. Every revision
   from the prior council pass has either been applied OR explicitly
   recorded as a waived design decision with rationale.

**LIFTED-TRAINER FORM**: substrate state where:
1. Basic implementation passes tests AND has PR95-paradigm tokens
   (eval_roundtrip + EMA + score-aware loss + canonical scorer
   helpers + archive grammar + inflate runtime); AND
2. EITHER no council deliberation exists yet, OR the latest council
   verdict is `PROCEED_WITH_REVISIONS` with un-applied revisions, OR
   the cargo-cult-unwind methodology has not been applied per
   Catalog #303.

### The rule

Every substrate at L1+ with `impl_complete=true` whose latest council
deliberation in `.omx/state/council_deliberation_posterior.jsonl`
returned `PROCEED_WITH_REVISIONS` AND no chronologically-later
PROCEED-unconditional anchor supersedes it MUST satisfy one of:

1. **Land iteration subagent commits** applying the council revisions
   + re-trigger sextet / grand-council deliberation that returns
   PROCEED (unconditional) per the canonical iteration methodology
   below. The new PROCEED anchor must be chronologically later than
   the PROCEED_WITH_REVISIONS anchor.
2. **Declare `research_only=true`** in the lane registry with
   `reactivation_criteria` pinned in notes per CLAUDE.md "Substrate
   scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable.
3. **Declare `lane_class=substrate_engineering`** per HNeRV parity
   discipline L7 (substrate engineering exceeds bolt-on size budget;
   not yet contest-dispatch eligible).
4. **Move to `archived/` state** with terminal verdict + reactivation
   criteria per CLAUDE.md "Forbidden premature KILL" (archived state
   is dormant-with-reactivation, not kill).
5. **Same-line waiver** `# OPTIMAL_FORM_DISPATCH_OK:<rationale>` in
   lane notes / evidence for the rare deliberate operator-approved
   pre-iteration dispatch (e.g. cheap $1-3 Catalog #167 smoke + probe
   disambiguator to inform the next iteration). The rationale must be
   a real string (placeholder `<rationale>` / `<reason>` literals are
   REJECTED so the gate's own docstring example cannot self-waive).

### Forbidden anti-pattern (the "dispatch-at-lifted-form" trap)

Empirically dispatching a substrate for a paid Modal/Lightning/Vast.ai
auth-eval anchor when the latest council deliberation returned
`PROCEED_WITH_REVISIONS` and none of the 5 opt-outs above applies is
FORBIDDEN. The dispatch produces a measurement that falsifies the
specific lifted-trainer implementation, not the novel concept; per
CLAUDE.md "Forbidden premature KILL without research exhaustion" + the
sister "Forbidden premature KILL" rule, this is the canonical way to
mis-classify an implementation-level finding as a paradigm-level kill.
The 4-of-5 dispatch failures this session are the empirical receipts.

### Canonical substrate iteration methodology (the NSCS06 v6→v7 pattern)

This is the **canonical method** for iterating a substrate from
LIFTED-TRAINER form to OPTIMAL FORM:

1. **Audit current implementation against operator's
   UNIQUE-AND-COMPLETE-PER-METHOD operating mode** (CLAUDE.md
   non-negotiable). Per layer, document the canonical-vs-unique
   decision per Catalog #290. The audit produces a per-layer
   classification: ADOPT_CANONICAL_BECAUSE_SERVES /
   FORK_BECAUSE_SUPPRESSES / FORK_BECAUSE_PRINCIPLED_MISMATCH /
   UNCLEAR_NEEDS_EMPIRICAL.
2. **Enumerate cargo-cults per Catalog #303** (`## Cargo-cult audit
   per assumption` section). Each cargo-culted assumption gets a
   HARD-EARNED-vs-CARGO-CULTED classification per the addendum +
   an explicit unwind path.
3. **Apply unwinds systematically** — produce v_n+1 implementation
   that addresses each unwind path with the substrate-optimal
   engineering choice (not the path-of-least-resistance canonical
   adoption). NSCS06 v6→v7 unwound 4-of-7 cargo-cults in one
   iteration and achieved 44% score improvement.
4. **Re-test sextet** (or grand council for tier-elevated lanes per
   Catalog #300). The new deliberation may return PROCEED-unconditional
   (OPTIMAL FORM reached), new PROCEED_WITH_REVISIONS (more iteration
   needed), DEFER_PENDING_EVIDENCE (probe gap), or REFUSE
   (paradigm-level question).
5. **Iterate until PROCEED-unconditional** before paid empirical
   dispatch. The cycle is not bounded by a fixed iteration count —
   the council verdict is the structural arbiter. Per CLAUDE.md
   "Forbidden premature KILL without research exhaustion", we
   iterate until the sextet says we're at OPTIMAL FORM or we
   document the architectural ceiling explicitly.

The cycle's chronological evidence is preserved in
`.omx/state/council_deliberation_posterior.jsonl` per Catalog #300
v2 frontmatter; sister Catalog #292 enforces per-deliberation explicit
assumption surfacing; Catalog #291 enforces per-session META-ASSUMPTION
cadence so the iteration loop itself does not silently drift.

### Concrete enforcement

- STRICT preflight Catalog #315
  (`check_substrate_at_optimal_form_before_paid_dispatch`) — refuses
  any in-scope substrate lane at L1+ with `impl_complete=true` whose
  latest council deliberation in `.omx/state/council_deliberation_posterior.jsonl`
  returned `PROCEED_WITH_REVISIONS` AND has no opt-out per the 5
  acceptance cascades above. STRICT-from-byte-one at landing 2026-05-17
  (live count: 0 — all 13 current PROCEED_WITH_REVISIONS council
  anchors map to lanes with structural opt-out via `research_only=true`
  or `lane_class=substrate_engineering`). The gate fires structurally
  the moment a future substrate is registered without opt-out and a
  council returns PROCEED_WITH_REVISIONS.
- The gate's substrate ↔ council join is via the
  `deferred_substrate_id` field in the council posterior schema. Lanes
  may declare a `substrate_alias` (or `substrate_aliases` list) in the
  registry so the v1-surface name used by the sextet can be matched
  back to the canonical lane id (e.g.
  `lane_substrate_z6` ↔ `z6_v1_ego_conditioning_surface`).
- Codex FIX-WAVE-R1 addendum (2026-05-17): the live C6 IBPS sextet
  row `council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517`
  was emitted with `deferred_substrate_id=null` while claiming Catalog
  #315 satisfaction. `check_substrate_at_optimal_form_before_paid_dispatch`
  now carries a narrow historical backfill keyed by immutable
  `deliberation_id` plus C6/MDL family tokens so active C6 dispatch
  lanes cannot be misreported as `no_council_anchor`. Generic
  time-traveler tokens are deliberately not used because the Z6 council
  row must not bind to older non-Z6 time-traveler lanes. FIX-WAVE /
  review lanes are explicitly out of Catalog #315 scope even when their
  names mention substrate families. Live count after this closure
  remains 1, correctly flagging the Z6-v2 PROCEED_WITH_REVISIONS
  blocker rather than C6 or review infrastructure.

### Cross-references

- Sister of Catalog #220 (substrate L1+ scaffold operational mechanism;
  runtime-effect surface) + Catalog #272 (distinguishing-feature
  integration contract; per-substrate-feature surface) + Catalog #233
  (L1→L2 promotion canonical 4-gate; promotion-discipline surface) +
  Catalog #298 (substrate L1 not stale dispatch; retirement-discipline
  surface) + Catalog #294 (9-dim success checklist evidence;
  design-memo surface) + Catalog #303 (cargo-cult audit section;
  design-memo surface) + Catalog #305 (observability surface;
  design-memo surface) + Catalog #300 (council deliberation v2
  frontmatter; council-discipline surface).
- Together they extinct the **dispatch-at-lifted-trainer-form bug
  class** across 8 orthogonal surfaces: design-memo (#290 + #294 +
  #303 + #305) + runtime-effect (#220) + per-feature (#272) +
  promotion-discipline (#233) + retirement-discipline (#298) +
  council-discipline (#300) + iteration-discipline (#315 — this
  gate).
- HNeRV parity discipline lesson 7 (bolt-on vs substrate-engineering
  split) — the canonical articulation of "iterate substrate engineering
  per method, share bolt-ons."
- CLAUDE.md "Forbidden premature KILL without research exhaustion" +
  "KILL/FALSIFIED memory verdicts" — Catalog #315 prevents the
  upstream cause of the symptom these gates address: paradigm-level
  KILL verdicts based on implementation-level falsifications.

## PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator standing directive 2026-05-17/18 verbatim *"all candidates
including c6 ibps may need further optimization and iteration and review and
audit and individual extreme passion and detail and effort and adversarial
grand council symposiums"*. Anchor memos:
`feedback_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_landed_20260517.md`
(empirical anchor: 22× miss of predicted band 0.113-0.163 → empirical 3.04
due to SegNet collapse — 4-of-5 distinguishing-feature dispatch failures this
session followed the same pattern), `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`
(Catalog #324 META-FIX at recipe-emit surface), and the canonical NSCS06 v6→v7
44% improvement via cargo-cult-unwind in ONE iteration (per
`.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md` per
Catalog #315 source). Extends (does NOT replace) the parent Catalog #315
OPTIMAL FORM discipline at the per-substrate-empirical-readiness surface.

### The structural failure this section extincts

The May-June 2026 dispatch waves empirically established that the SAME bug
class recurs across substrates: each substrate carries IDIOSYNCRATIC
cargo-culted assumptions that META-layer audits (Catalog #303 + #294 + #305 +
#296 + #324) CAN surface at design-memo + recipe-emit surfaces but CANNOT
adjudicate per-substrate without dedicated council deliberation. Per
operator: each ASYMPTOTIC pursuit candidate deserves *"individual extreme
passion and detail and effort and adversarial grand council symposium"*
BEFORE paid dispatch — not generic META-layer pass-through. Catalog #315
covers iteration-discipline (council verdict from PROCEED_WITH_REVISIONS to
PROCEED-unconditional); this section covers per-substrate symposium-evidence
discipline (the council ACTUALLY ran for THIS specific substrate, recently,
with the canonical 6-step contract below).

### The canonical 6-step per-substrate symposium contract

Every ASYMPTOTIC pursuit candidate (any L1+ substrate lane whose recipe
declares `dispatch_enabled: true` for paid dispatch >$0.30) MUST undergo
individual adversarial grand council symposium satisfying ALL 6 steps within
a 14-day window BEFORE paid dispatch is admissible:

1. **Cargo-cult audit per Catalog #303** — `## Cargo-cult audit per assumption`
   section enumerating each substrate-design assumption + HARD-EARNED-vs-
   CARGO-CULTED classification per the hard-earned-vs-cargo-culted addendum
   (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`)
   + unwind-test plan per cargo-culted assumption.
2. **9-dimension success checklist evidence per Catalog #294** — `## 9-dimension
   success checklist evidence` section documenting per-dimension evidence
   (UNIQUENESS / BEAUTY+ELEGANCE / DISTINCTNESS / RIGOR / OPTIMIZATION-PER-
   TECHNIQUE / STACK-OF-STACKS-COMPOSABILITY / DETERMINISTIC-REPRODUCIBILITY /
   EXTREME-OPTIMIZATION-PERFORMANCE / OPTIMAL-MINIMAL-CONTEST-SCORE).
3. **Observability surface declaration per Catalog #305** — `## Observability
   surface` section declaring all 6 facets (inspectable per layer /
   decomposable per signal / diff-able across runs / queryable post-hoc /
   cite-able / counterfactual-able).
4. **Sextet pact deliberation** (Shannon LEAD + Dykstra CO-LEAD + Yousfi +
   Fridrich + Contrarian + Assumption-Adversary) per CLAUDE.md "Council
   conduct" amendment + Catalog #292 per-deliberation assumption surfacing.
   Grand council attendees added per topic (e.g. Tishby memorial for IB
   substrates, Schmidhuber for compression-as-intelligence, Atick for
   cooperative-receiver, MacKay memorial for MDL, Rao+Ballard for
   predictive-coding, etc.). T2 tier or higher per CLAUDE.md "Council
   hierarchy: 4-tier protocol".
5. **Per-substrate reactivation criteria pinned** per CLAUDE.md "Forbidden
   premature KILL" — the symposium MUST enumerate reactivation paths
   (typically 3-4) with priority ordering, predicted cost, and structural
   verdict on which assumption each path tests.
6. **Catalog #324 post-training Tier-C validation discipline** —
   `predicted_band_validation_status` MUST be either `post_training_*ep_*`
   (validated against landed archive) OR explicitly `pending_post_training`
   (with reactivation criterion = post-training Tier-C re-measurement on the
   landed archive sha256). The symposium MUST note which status applies.

### Dispatch eligibility gate

Paid dispatch (Modal / Lightning / Vast.ai >$0.30) is ONLY permitted when:

(a) The substrate has a per-substrate symposium memo at
`.omx/research/council_*_<substrate_id>_*_<YYYYMMDD>.md` dated within the
last 14 days, AND
(b) The symposium verdict is one of {`PROCEED`, `PROCEED_WITH_REVISIONS`},
NOT one of {`DEFER_PENDING_EVIDENCE`, `REFUSE`, `ESCALATE_TO_HIGHER_TIER`,
`ESCALATE_TO_OPERATOR`}, AND
(c) The symposium memo carries the canonical 6-step contract per Catalog
#324 + #294 + #303 + #305 + #292 + #300 frontmatter, AND
(d) Council deliberation posterior at
`.omx/state/council_deliberation_posterior.jsonl` has matching anchor
written via `tac.council_continual_learning.append_council_anchor`.

EXCEPT explicit operator-frontier-override per Catalog #300 §"Mission
alignment" operational consequence 1 (override REQUIRES operator-verbatim
quote in `council_override_rationale` frontmatter; bypasses quorum + tie-
break + recusal for the specific decision but preserves maximum-signal
preservation).

### Frequency

Once per substrate per major design iteration. Re-convene when ANY of:

- The cargo-cult audit surfaces a NEW HARD-EARNED-vs-CARGO-CULTED
  reclassification (an assumption flips between rounds).
- An empirical anchor reveals predicted-band miss >2× (per Catalog #324
  post-training Tier-C validation discipline).
- A sister substrate's empirical anchor invalidates a shared assumption.
- 14 days have elapsed since the most recent per-substrate symposium AND
  paid dispatch is being considered.

### Operator-routable enforcement at dispatch time

The runtime gate (`tools/operator_authorize.py::_dispatch_modal` etc.) does
NOT enforce this discipline directly — Catalog #325 STRICT preflight gate
fires at design-memo + recipe surface BEFORE the operator-authorize harness
runs. The 30-second pre-deploy harness consults the per-substrate symposium
verdict via the canonical posterior at
`.omx/state/council_deliberation_posterior.jsonl` and refuses dispatch if no
symposium anchor satisfies the contract for the target substrate.

### Cross-references

- Sister of Catalog #315 (parent OPTIMAL FORM discipline; iteration-axis)
  and Catalog #324 (post-training Tier-C validation; recipe-emit surface)
  + Catalog #303 (cargo-cult audit section; design-memo surface)
  + Catalog #294 (9-dim checklist evidence; design-memo surface)
  + Catalog #305 (observability surface; design-memo surface)
  + Catalog #296 (Dykstra-feasibility predicted-band; design-memo surface)
  + Catalog #292 (per-deliberation assumption surfacing; council-discipline)
  + Catalog #300 (council deliberation v2 frontmatter; council-discipline).
- Together they extinct the "substrate dispatch without per-substrate optimal
  form symposium" bug class across NINE orthogonal surfaces: design-memo
  (#290 + #294 + #303 + #305 + #296) + runtime-effect (#220) + per-feature
  (#272) + promotion-discipline (#233) + retirement-discipline (#298) +
  council-discipline (#300 + #292 + #291) + iteration-discipline (#315) +
  post-training-validation (#324) + per-substrate-symposium-evidence (#325).
- Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4:
  frontier-breaking moves DOMINATE rigor budget — this gate's 14-day window
  + WARN-ONLY initial wire-in are calibrated so race-mode rigor inversion
  per CLAUDE.md "Race-mode rigor inversion" remains the operator-override
  path. The symposium discipline serves the mission; when the mission
  requires faster cadence, operator-frontier-override bypasses the gate
  with full audit trail.

## Apples-to-apples evidence discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator correction, 2026-05-10, after PR103 histogram packet review.

Never classify, promote, retire, or submit a HNeRV/public-frontier result from
an inferred equivalence. Every score conclusion needs an apples-to-apples
baseline on the same axis and runtime contract.

Hard rules:

1. **Decoded-state parity is not frame parity.** Identical decoded
   `state_dict` / latents / symbols proves parser consumption only. It does
   not prove same rendered RGB bytes, scorer components, CUDA numerics, or
   public-leaderboard behavior. A manifest may say
   `decoder_state_parity_passed=true`, but it MUST keep
   `full_frame_inflate_output_parity_missing` until source-vs-candidate
   `inflate.sh archive_dir output_dir file_list` outputs are compared
   byte-for-byte or exact same-runtime evals are available for both packets.
2. **CPU and CUDA are separate evidence spaces.** The HNeRV cluster often
   scores much better on `[contest-CPU]` than on `[contest-CUDA]`. CPU
   medal-band proximity is real public-axis evidence, but it is not CUDA
   readiness, CUDA frontier status, or a conversion shortcut. Never infer one
   axis from the other; run both when shipment/frontier language is used.
   Do not invert this into a universal "CPU is better" rule either: every
   packet must be measured per archive/runtime/inflate-device/evaluate-device,
   with inflated raw-output hashes and PoseNet/SegNet component deltas when
   diagnosing the mechanism.
3. **Source runtime must match the comparison.** For public PR clones, compare
   original archive + original `inflate.sh` against candidate archive +
   candidate runtime under the same evaluator path. If the candidate adapter
   changes `inflate.py`, `inflate.sh`, Python invocation, dependency closure, or
   section constants, the source replay used as baseline must be the matching
   source runtime, not a nearby repack or previous Modal/Lightning artifact.
4. **Negative exact evals need harness review before method verdicts.** If a
   byte transform preserves decoded tensors but exact eval changes, default to
   `indeterminate-harness-or-runtime-mismatch` until full-frame output parity,
   same-runtime source replay, and component recomputation agree. Do not call it
   a method negative just because a CUDA number returned.
5. **Generated reports must preserve the axis label.** Phrases like
   "rounds to gold", "medal-band", "submission-ready", "auto-promote", and
   "score gap" must include `[contest-CPU]`, `[contest-CUDA]`,
   `[macOS-CPU advisory]`, or `[proxy]` inline. Missing axis label is a bug.

When in doubt, downgrade the finding, write a supersession ledger, and run the
apples-to-apples proof before spending another dispatch.

## Bugs must be permanently fixed AND self-protected against — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator directive (2026-05-09): *"such bugs must be permanently fixed and self-protected against."*

Every adversarial-review finding (codex / grand council / sister subagent) that surfaces a real bug MUST be addressed with TWO landings, NOT ONE:

1. **The fix** — patches the immediate code surface
2. **A STRICT preflight check** — refuses any code surface in the repo that re-introduces the bug class, with a `check_<bug_class_name>` function in `src/tac/preflight.py`, wired into `preflight_all()`, with dedicated tests

Single-surface fixes are insufficient. Per the META-meta finding from a8bc7e79's proactive sweep: bug classes have **6-7× spread** across the repo. A fix at one surface leaves the same class active at 6 others.

### The codex-review fix-with-strict-preflight pattern (canonical)

For each codex review finding (HIGH or MEDIUM):

1. **Patch the cited file:line** with the recommended fix
2. **Claim a catalog #** via `tools/claim_catalog_number.py claim`
3. **Add a STRICT preflight check function** `check_<bug_class>` to `src/tac/preflight.py`:
   - Scans the targeted directories (`src/tac/`, `tools/`, `experiments/`, `scripts/`) for the bug-class signature
   - Allows opt-out via same-line `# <BUG_CLASS_OK>:<rationale>` waiver
   - Raises `PreflightError` in strict mode; warns in non-strict
4. **Wire into `preflight_all()`** — initially `strict=False` (warn-only)
5. **Write 15-25 dedicated tests** covering: positive (catches violation), negative (allows non-violations), waiver-respect, edge cases
6. **Verify live count = 0** by running the check strict against the current repo state
7. **Strict-flip the wire-in** to `strict=True` once live count = 0
8. **Add a row to the CLAUDE.md "Meta-bug class catalog (strict-mode preflight)" table** with catalog #, name, what it prevents, memory ref

### Strict-flip atomicity rule

If the fix subagent achieves live count = 0 in the same landing, the strict-flip should land in the SAME commit-batch (not a follow-up). This avoids the warn-only-purgatory failure mode where a check ships warn-only and the strict-flip never happens.

### Examples from this session (canonical pattern)

- Catalog #123 (`check_no_weight_domain_saliency_on_score_gradient_substrate`) extincts the Track 4 v1 Fisher-proxy-inversion bug class
- Catalog #124 + #125 + #126 + #127 + #128 + #130 + #131 — 7 META gates landed across the session, all following this pattern
- a00501f9's round-3 fix MUST land Catalog #132 (`check_locked_writes_preserve_deletions`) per the same pattern, plus #133/#134/#135 for HIGH 2 / MEDIUM 1 / MEDIUM 2 (one-strict-check-per-finding)

### Anti-pattern: fix-without-self-protection

Any commit that fixes a codex finding WITHOUT landing the corresponding STRICT preflight check is INCOMPLETE. The reviewer should reject the commit on the grounds that the bug class will re-emerge at a different surface.

## Subagent coherence-by-default — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator concern (2026-05-09): *"i am concerned we are building intelligent systems but they are not coherent and integrated and maybe duplicate ... the should just work and run in the background for us perhaps as skills or via mcp tools or something i'm not sure how to solve this problem ... or maybe just engineer correctly and then save related knowledge and instructions in claude and agents .md."*

**The answer is the latter**: don't add another orchestration layer (skills, MCP). Engineer the right primitives, save the discipline in CLAUDE.md + AGENTS.md, and EVERY future subagent honors it without an orchestrator. The non-negotiables in this file ARE the orchestration layer — they propagate via every subagent's mandatory pre-read.

### Mandatory pre-flight for every subagent (parent + nested)

Before starting any work, every subagent MUST:

1. **Read CLAUDE.md AND AGENTS.md** — both files. Honor every NON-NEGOTIABLE marker. The "I didn't read it" failure mode is a process bug, not an information gap.
2. **Check the lane registry** (`.omx/state/lane_registry.json`) for in-flight conflicts. If your lane shares a `lane_id` or `target_modes`/`deployment_target` with an active claim, coordinate via the file's notes column or pick a different lane.
3. **Check sibling subagents in the same conversation** — when the parent prompt says "running in parallel right now", read the listed sibling subagent IDs and their scopes. Do NOT duplicate their primary deliverable.
4. **Read latest top-of-MEMORY.md entries** — at least the top 10. Recent landings change the optimal next-step.
5. **Read all `.omx/research/*_directive_*` files** dated within the last 24 hours — they contain operator-routed inter-subagent directives that supersede the original prompt.

### Mandatory crash-resume protocol

**Source:** operator directive 2026-05-14 ("why did it die? need to investigate and fix permanently"). Empirical anchor: Wyner-Ziv research subagent (id `a1362a24d986029c3`) crashed mid-session with Anthropic API `Internal server error` after 17 minutes / 58 tool uses / 1704 tokens; all in-flight progress was lost. Sister pattern WAVE-3-HNERV-C-RETRY (DSNeRV + HiNeRV trainers) hit the same failure class but survived because intermediate commits had already landed.

Every subagent prompt MUST include resume-from-disk instructions:

1. **At start, before doing ANY work:** run `tools/subagent_checkpoint.py read --subagent-id <YOUR_ID>` and check if any predecessor exists. If yes: read predecessor's `next_action` + `files_touched` and resume from there. Do NOT restart from scratch.
2. **Every 10 tool uses (or after each major milestone):** call `tools/subagent_checkpoint.py --subagent-id <YOUR_ID> --step <N> --status in_progress --files-touched <...> --next-action <...>` so a successor can resume on API crash.
3. **On completion:** call `tools/subagent_checkpoint.py --subagent-id <YOUR_ID> --step complete --status complete --files-touched <...> --next-action ""`.

The canonical store is `.omx/state/subagent_progress.jsonl` (JSONL, fcntl-locked per Catalog #131). Schema fields: `subagent_id` / `parent_id_or_session` / `step` / `status` (`in_progress` | `blocked` | `complete`) / `files_touched` / `next_action` / `notes` / `written_at_utc` / `pid` / `host`.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — every long-running subagent MUST honor this protocol. Catalog #206 (`check_subagent_dispatches_use_checkpoint_discipline`) enforces it: subagent commits whose body lacks a checkpoint trace are refused. Short subagents (≤5 tool uses estimate) may carry a same-line `# CHECKPOINT_DISCIPLINE_WAIVED:<reason>` waiver in the commit body. Memory: `feedback_subagent_crash_resume_discipline_landed_20260514.md`.

### Mandatory wire-in for every landing (no orphaned signals)

Every landing must wire its outputs into the unified solver stack OR explicitly tag `research_only=true`. Per `feedback_unified_lagrangian_action_principle_GR_style_20260509.md`:

1. **Sensitivity-map contribution** in `tac.sensitivity_map.*` (or sibling)
2. **Pareto constraint** added to `tac.pareto_*` (or explicitly reasoned why non-binding)
3. **Bit-allocator hook** registered if per-tensor importance changes
4. **Cathedral autopilot dispatch hook** registered if archive-deployable
5. **Continual-learning posterior update** triggered on every empirical anchor
6. **Probe-disambiguator** built if 2+ defensible interpretations exist (`tools/probe_<track>_disambiguator.py`)

If any of the 6 hooks is N/A, declare it explicitly in the landing memo with rationale. **Silent omission is the orphan-work failure mode.**

### Anti-duplication primitive: the lane registry IS the deduplication layer

Two subagents working on the same lane is a registry failure, not a coordination failure. The fix is:

1. Pre-register every lane (even SKETCH at L0) the moment a name + verdict exists, per CLAUDE.md "Lane maturity registry" non-negotiable lifecycle discipline.
2. Subagent prompts MUST cite the registered `lane_id` in the prompt body so collisions surface at parent-coordinator review time.
3. The `tools/lane_maturity.py audit` table is the single source of truth for "what's currently being worked on." Use it.

### Anti-fragmentation primitive: the unified-Lagrangian action

Per `feedback_unified_lagrangian_action_principle_GR_style_20260509.md`, the migration target is `tac.unified_action.S_total(theta, archive_bytes, hardware)` — ONE scalar action, all track-Lagrangians composed via δS/δθ = 0 (GR-style variational principle). Until that lands, individual track wire-ins must explicitly call all 6 integration hooks above.

When the unified action lands, every track plugs in by adding a term to `S_total` — no new orchestration layer. The coherence is structural.

### Anti-arbitrariness primitive: the probe-disambiguator pattern

Per `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`, when a design choice has 2+ defensible interpretations, ship BOTH modes via callable interface + build `tools/probe_<track>_disambiguator.py` that returns the regime-conditional verdict. The probe IS the arbitration; the trainer/codec/solver consumes the verdict.

### Background-execution clarification

The operator floated "skills or MCP tools" as the orchestration mechanism. **Do not pursue this path.** The CLAUDE.md + AGENTS.md non-negotiables ARE the always-on, zero-token orchestration layer. Every subagent loads them by default; every behavior is encoded in inviolable rules. Adding another layer would be the kitchen_sink anti-pattern at the meta level.

If a behavior should be automatic across all sessions, write it as a CLAUDE.md non-negotiable. The skill-vs-rule decision: **skills are user-invocable patterns; rules are agent-binding contracts.** The coherence problem is solved by RULES, not skills.

### Cross-agent sister convergence patterns (canonical META-pattern; 2026-05-21 worked example)

**Source:** convergent multi-subagent session 2026-05-21 — 7 sister convergence patterns observed across 4 distinct structural variants in a single ~6-hour window; documented in slot 3-r5 STAND DOWN memo `149bdc6a1` + slot 3-r6 Catalog #359 cross-reference audit `a4ad7027b` + slot 3-r7 ATW V2 reconciliation memo `265431dfe` + slot 2-r reverse-directive issuance `7ea60e91f`.

Cross-agent sister-coherence (claude ↔ codex working on disjoint surfaces in the same session) is structurally extincted via the existing structural-extinction surfaces enumerated in "Anti-duplication primitive" and "Mandatory wire-in" above. The 4 distinct convergence pattern variants documented here are canonical worked examples that future subagents can recognize proactively rather than discovering post-hoc when a sister gate fires.

#### 4 distinct convergence pattern variants

1. **STAND_DOWN pattern**: claude subagent spawned → codex sister already landed equivalent work → claude STAND_DOWN per Catalog #340 sister-coherence → audit memo documents convergence; ZERO duplicate work. Canonical example: slot 3-r5 (`149bdc6a1`) — claude lane `lane_wave_3_vq_vae_indices_blob_procedural_variant_extension_20260520` verified sister codex `77081f991` covered complete TaskCreate #1154 scope; stood down without any commits to sister-owned files.

2. **COMPLEMENTARY pattern**: codex sister lands OPERATIONAL module (executable code + tests + CLI) + claude lands DESIGN SPEC (design memo + canonical equation routing + paradigm classification) + claude lands RATIFICATION (audit memo verifying sister routing correctness across structural surfaces) = 3-surface canonical ratification. Canonical example: canonical equation `procedural_predictor_plus_residual_correction_savings_v1` instantiation — codex `77081f991` lands `src/tac/substrates/vq_vae/indices_procedural_variant.py` operational module + claude slot 3-r6 (`a4ad7027b`) lands audit memo verifying codex's residual-hybrid routing correctness across 3 surfaces (module imports + callsite context strings + canonical equation registry anchors).

3. **SUPERSESSION pattern**: codex sister landings cover scope that claude subagent was queued for; queued TaskCreate marked completed-by-sister; sister-coherence preserved. Canonical examples: NULL-BYTE PROBE MATRIX + PAIR #4 ORTHOGONALITY SMOKE — sister codex landings covered the scope claude was queued for; claude TaskCreate marked completed-by-sister per the canonical task status ledger (Catalog #331).

4. **CODEX-EMPIRICAL-FALSIFICATION-OF-CLAUDE-DESIGN pattern** (NEW today): codex empirical anchor (byte-mutation smoke / parity probe / structural verification) FALSIFIES claude design memo core assumption; APPEND-ONLY reconciliation memo documents implementation-level falsification per Catalog #307 paradigm-vs-implementation classification + proposes paradigm reclassification + (optionally) proposes NEW canonical equation EXCLUDED context per Catalog #344 operator-decision protocol. Canonical example: slot 3-r7 ATW V2 cdf_table_blob (`265431dfe`) — codex byte-mutation smoke `057130de4` empirically proves `max_abs_raw_byte_delta=0` across all 2,560 cdf_table_blob bytes mutated; claude design memo `8441b702e` REPLACEMENT paradigm routing via canonical equation #26 IN-DOMAIN context `atw_v2_codec_quantizer_lut` is IMPLEMENTATION-LEVEL FALSIFIED; APPEND-ONLY reconciliation memo `265431dfe` documents falsification + proposes REMOVAL paradigm reclassification + proposes NEW canonical equation #26 EXCLUDED context `direct_byte_substitution_on_decode_opaque_raw_sections` per Catalog #344 operator-decision protocol.

#### Canonical structural extinction surfaces (8+ Catalog gates)

The 4 pattern variants are structurally extincted via the following Catalog gates working in concert (the gates are the canonical enforcement; the variants are canonical worked-example references):

- **Catalog #340** `check_subagent_commit_serializer_invokes_sister_checkpoint_guard` — edit-time staging-surface PREVENT (all 4 variants).
- **Catalog #110 + #113** APPEND-ONLY HISTORICAL_PROVENANCE discipline — no mutation of sister memos (all 4 variants; especially Variant 4).
- **Catalog #117 / #157 / #174** canonical serializer + pre-pre-lock hash + mandatory `--expected-content-sha256` — commit-time fcntl-locked arbitration (Variant 1).
- **Catalog #335** `check_cathedral_consumer_directory_package_exposes_canonical_contract` — auto-discovery of sister-landed cathedral consumers (Variant 2).
- **Catalog #344** `check_empirical_finding_memo_references_canonical_equation` — canonical equation evolution discipline + operator-decision protocol for adding NEW IN-DOMAIN / EXCLUDED contexts (Variants 2 + 4).
- **Catalog #359** `check_no_canonical_equation_misapplication_to_residual_hybrid_contexts` — STRICT preflight refusal of canonical equation #26 misapplication (Variant 2).
- **Catalog #105 + #139 + #272** byte-mutation smoke gates — sister codex empirical falsification mechanism (Variant 4).
- **Catalog #307** `check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification` — forces RATIFY-FALSIFICATION-OF-THE-SPECIFIC-IMPLEMENTATION verdict structure (Variant 4).
- **Catalog #331** `check_canonical_task_status_no_dangling_transitions` — canonical task status ledger transition discipline (Variant 3).
- **Catalog #333** `check_codex_inbox_open_questions_have_response_or_default_within_deadline` — codex-to-Claude inbox bidirectional channel (Variant 1 SOURCE-DIRECTION).

#### Worked-example chain canonical reference

- `7ea60e91f` — claude reverse codex-routing-directive issuance (slot 2-r; UPSTREAM Variant 1 source)
- `77081f991` — codex sister VQ-VAE indices_blob procedural variant scaffold landing (per directive #4; COMPLEMENTARY Variant 2)
- `149bdc6a1` — claude STAND_DOWN memo (slot 3-r5; STAND_DOWN Variant 1)
- `a4ad7027b` — claude Catalog #359 cross-reference audit (slot 3-r6; verifies sister routing correctness; COMPLEMENTARY Variant 2)
- `057130de4` — codex ATW2 CDF dead-section parity probe (byte-mutation smoke; COMPLEMENTARY Variant 2)
- `265431dfe` — claude APPEND-ONLY reconciliation memo (slot 3-r7; codex empirical smoke falsifies claude design memo; CODEX-EMPIRICAL-FALSIFICATION-OF-CLAUDE-DESIGN Variant 4)

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable: the structural-extinction surfaces (the 10 Catalog gates above) are the canonical enforcement; this section is the operator-facing canonical worked-example reference. A future subagent that violates the convergence pattern is structurally refused by the gates; the gates fire at edit-time / commit-time / dispatch-time without requiring subagent awareness of the 4 variants. The marginal value of this section is in pattern-recognition (subagent recognizes convergence opportunity proactively rather than discovering it post-hoc when a sister gate fires).

### Concrete enforcement

- New STRICT preflight check planned: `check_subagent_landing_has_solver_wire_in` — refuses any landing memo that doesn't declare all 6 wire-in hooks (or `research_only=true`).
- New STRICT preflight check planned: `check_lane_pre_registered_before_work_starts` — refuses subagent commits whose `lane_id` doesn't appear in the registry.
- The existing Check 90 `check_lane_registry_consistent` partially covers this; the two new checks extend it to subagent-discipline territory.

### Recommended `/commit` slash-command pre-hook (Catalog #340, 2026-05-19)

The operator's `/commit` slash command (commit-commands plugin at
`~/.claude/plugins/marketplaces/claude-plugins-official/plugins/commit-commands/commands/commit.md`)
does bare `git add` + `git commit` directly, OUTSIDE the canonical
`tools/subagent_commit_serializer.py`. When invoked while a sister
subagent has uncommitted edits in the shared working tree, the bare
`git add` packages whatever the LLM thinks is relevant — which can
include the sister's still-in-flight edits (the absorption-pattern
bug class anchored by Catalog #314 + #340).

The slash-command plugin is operator-owned and cannot be edited from
this repo. The integration pattern is to wrap the slash command in a
project-level pre-commit hook OR a wrapper shell function:

```bash
# .git/hooks/pre-commit (operator installs locally)
#!/bin/bash
set -euo pipefail
files=$(git diff --cached --name-only)
if [[ -n "$files" ]]; then
    .venv/bin/python tools/check_sister_checkpoint_before_git_add.py \
        --files-from-stdin \
        --label "${SUBAGENT_LABEL:-anonymous}" \
        <<<"$files"
fi
```

The helper exits 0 (PROCEED), 8 (ABORT), 9 (WAIT_AND_RETRY), 10 (bare
paired-env bypass attempt), or 11 (corrupt JSONL). Set
`SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE=1` AND
`SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_RATIONALE=<text>` to bypass
when coordination has been confirmed via Catalog #230 ownership map.

This protects the bare `git add` surface that Catalog #340's STRICT
preflight gate cannot reach (preflight runs BEFORE commit, not BEFORE
the bare staging operation). The two together (preflight gate +
optional pre-commit hook) close the staging-surface absorption pattern
across both the canonical serializer and the slash-command paths.

## Canonical helper 6-pillar landing discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** D3 of the canonical rename + apparatus hardening wave 2026-05-30 (D1+D2 inline landing memo `feedback_claude_md_canonical_rename_wave_d1_d2_inline_landing_20260530.md` D3-D12 queue + operator approval 2026-05-30 verbatim *"All are approved, land inline"* + operator binding 2026-05-30 *"wired + integrated + tested + individually fractally optimized for extreme synergy and positive externalities"*). This consolidates the recurring landing checklist that every recent canonical helper landing has converged on into a single binding non-negotiable so no future canonical helper can ship as orphan work.

This is the canonical articulation of the "Subagent coherence-by-default" Mandatory wire-in section (above) applied specifically to the **canonical-helper landing** surface. The "Mandatory wire-in" section enumerates the 6 unified-Lagrangian solver hooks (sensitivity-map / Pareto / bit-allocator / cathedral autopilot / continual-learning / probe-disambiguator); this D3 section enumerates the 6 PILLARS a canonical helper MUST satisfy to be considered LANDED (not merely written). Both are required; the wire-in hooks are pillar 4 of the 6 pillars below.

### The 6 pillars

Every NEW canonical helper (a reusable `tac.*` module, codec primitive, archive grammar, planner primitive, cathedral consumer, or canonical-equation/anti-pattern registry entry) MUST satisfy ALL 6 pillars before its landing memo may claim it is LANDED:

1. **Wired + integrated** — at least 1 production caller imports + invokes the helper (NOT only tests). If the helper is a cathedral consumer, it is auto-discovered per Catalog #335 canonical contract. If the helper is a meta-Lagrangian / Pareto / master-gradient surface, it has an invoker callsite in `main()` per Catalog #355/#372/#336/#337/#379. A canonical helper that no production caller imports is the orphan-signal failure mode per CLAUDE.md "Results must become system intelligence" — explicitly tag `research_only=true` with a concrete integration blocker OR satisfy this pillar.
2. **Tested** — at least 15 dedicated tests cover the helper's public API (positive / negative / edge / waiver-respect where applicable) AND the relevant sister regression suite passes. Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against": a fix-without-tests is incomplete.
3. **Catalog-Provenance-routing** — canonical Provenance per Catalog #323 is threaded through every score-claim return value (axis_tag + hardware_substrate + evidence_grade triple). If the helper is observability-only (a router / annotator / validator that does NOT mutate score), it carries the canonical Tier A non-promotable markers per Catalog #341 (`predicted_delta_adjustment=0.0` / `promotable=False` / `axis_tag="[predicted]"`).
4. **Memo-anchored** — a landing memo exists declaring all 6 unified-Lagrangian wire-in hooks per Catalog #125 (sensitivity-map / Pareto / bit-allocator / cathedral autopilot / continual-learning / probe-disambiguator; each ACTIVE or N/A-with-rationale) AND the `council_predicted_mission_contribution` per Catalog #300 (`frontier_breaking` / `frontier_protecting` / `rigor_overhead` / `apparatus_maintenance` / `mission_questioned`).
5. **Lane-registered** — the work is registered in the lane registry per Catalog #90 + #126 (`lane_<NAME>_<YYYYMMDD>` pre-registered before work starts; gates marked as evidence is produced) AND a probe-outcome row per Catalog #313 records the verdict (PROCEED / DEFER / etc.) with its staleness window.
6. **Retroactively-swept** — if the landing introduces a NEW STRICT preflight gate, a Catalog #348 retroactive-sweep memo (`.omx/research/retroactive_sweep_for_catalog_<N>_<utc>.md`) records the bug-class symptom signature + pre-fix window + historical-KILL/DEFER/FALSIFY search + per-finding RE-EVAL-priority. (Pillar 6 is N/A for canonical helpers that do NOT land a STRICT gate; declare N/A-with-rationale.)

### Falling-rule per pillar

A landing memo that claims LANDED but cannot satisfy a pillar MUST either (a) satisfy it before claiming LANDED, OR (b) explicitly tag the helper `research_only=true` with a concrete integration blocker per pillar 1, OR (c) declare the pillar N/A with substantive rationale (pillar 6 N/A for non-gate helpers; pillar 3 observability-only path). Silent omission of any pillar is the orphan-work failure mode.

### Concrete enforcement

- This discipline is enforced at landing-memo review time + adversarial-review time. The existing Catalog #125 (`check_subagent_landing_has_solver_wire_in`) STRICT gate covers pillar 4 (the 6-hook wire-in declaration). The existing Catalog #335 (canonical cathedral consumer contract) covers pillar 1 for cathedral consumers. The existing Catalog #348 (retroactive-sweep evidence) covers pillar 6. Pillars 2 / 3 / 5 are enforced by their respective sister gates (test-presence convention, Catalog #323 Provenance umbrella + Catalog #341 Tier A markers, Catalog #90 + #126 + #313 lane/probe discipline). D3 consolidates these into a single operator-facing 6-pillar contract so a reviewer can audit a landing against ONE checklist rather than reverse-engineering which sister gates apply.
- A dedicated umbrella STRICT gate consolidating all 6 pillars into one consult-call is a candidate for a future cap-window landing per Catalog #299 gate-consolidation discipline (do NOT land it pure-additive; it must subsume >=3 sister cases or REPLACE them). The planned **D6** STRICT gate `check_canonical_helper_landing_satisfies_6_pillar_discipline` is the canonical landing surface (still QUEUED per the cap=1-per-turn anti-pattern — NOT landed in this batch).

### Cross-references

- "Subagent coherence-by-default" Mandatory wire-in section (above) — the canonical 6-hook source (pillar 4).
- "Results must become system intelligence" — the orphan-work non-negotiable (pillar 1).
- Catalog #125 (6-hook wire-in declaration; pillar 4) + Catalog #335 (cathedral consumer canonical contract; pillar 1 auto-discovery) + Catalog #323 (canonical Provenance umbrella; pillar 3) + Catalog #341 (Tier A canonical-routing markers; pillar 3 observability-only) + Catalog #300 (council deliberation v2 frontmatter + mission contribution; pillar 4) + Catalog #90 / #126 / #313 (lane registry + probe outcome; pillar 5) + Catalog #348 (retroactive sweep; pillar 6).

## Main branch source of truth — NON-NEGOTIABLE

`main` is the sole source-of-truth branch. Do not do production work, recovery
work, public-frontier intake, or contest-custody edits on any other branch.
Detached public PR clones, stashes, quarantine trees, provider workspaces, and
subagent forks are forensic inputs only; promoted code, docs, artifacts, and
ledgers must land back on `main` after explicit review.

## Frontier scores are pointer-only — NON-NEGOTIABLE

**Source of truth for OUR LOCAL FRONTIER + PUBLIC LEADERBOARD scores:**
`.omx/state/canonical_frontier_pointer.json` (machine-readable; updated via
`tools/refresh_canonical_frontier.py` or auto on dispatch completion per
Catalog #343).

**FORBIDDEN**: hardcoded score literals in CLAUDE.md / MEMORY.md / memory
files for our local frontier or current public leaderboard. The pointer
file is the SoT; hardcoding causes drift that produces misleading operator
briefings.

Empirical anchor: the pre-pointer state let a frontier-score conflation
between our local frontier (lane
`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`, archive sha
prefix `6bae0201`) and the PR101 GOLD UPSTREAM baseline (archive sha
prefix `b83bf348`) go undetected until the operator corrected it
2026-05-19. The pointer is the structural extinction of that drift class.

**ALLOWED**: hardcoded score literals in HISTORICAL-CONTEXT (catalog row
docstrings, falsification verdicts, postmortems, historical PR refs per
CLAUDE.md "KILL/FALSIFIED memory verdicts" + Catalog #110 / #113
HISTORICAL_PROVENANCE non-negotiable) — these MUST be tagged with same-line
`# HISTORICAL_SCORE_LITERAL_OK:<rationale>` waiver to pass Catalog #343
strict gate.

**Operator-facing access**:

```bash
# Print current frontier in human-readable form.
.venv/bin/python tools/refresh_canonical_frontier.py

# Opt in to upstream public leaderboard fetch (~30s network call).
.venv/bin/python tools/refresh_canonical_frontier.py --update-upstream

# Strict mode: exit rc=1 if pointer is stale (>24h).
.venv/bin/python tools/refresh_canonical_frontier.py --strict

# Machine-readable JSON for autopilot / dashboard consumers.
.venv/bin/python tools/refresh_canonical_frontier.py --json
```

**DX auto-update**: every successful Modal / HF Jobs dispatch completion
fires `tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`
from inside `tac.deploy.modal.call_id_ledger.update_call_id_outcome` +
`tac.deploy.hf_jobs.job_id_ledger.update_hf_jobs_outcome`. The pointer
auto-refreshes; operators rarely need the manual refresh CLI.

**Sister discipline**: Catalog #316 (`check_reports_latest_md_not_stale_vs_canonical_frontier`)
keeps `reports/latest.md` aligned. Catalog #131 + #138 (fcntl-locked atomic
write + strict-load fail-closed) keep the pointer's persistence layer
consistent. Catalog #245 (Modal call_id ledger) is the canonical 4-layer
exemplar this pointer mirrors. Catalog #343 (this gate) refuses NEW
hardcoded literals in CLAUDE.md without canonical pointer reference or
HISTORICAL_SCORE_LITERAL_OK waiver (warn-only initially per Strict-flip
atomicity rule because legacy CLAUDE.md has dozens of historical anchors
that need backfill OR HISTORICAL-CONTEXT waivers).

**2026-05-19 PR-submission status (APPEND-ONLY per Catalog #110
HISTORICAL_PROVENANCE):** Per operator blanket approval 2026-05-19 verbatim
*"all operator decisions and approval granted and provided fuly and
completely"*, an operator-administrative subagent attempted to submit the
current frontier (archive sha `6bae0201fb08...` / lane
`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515`) as a
contest PR to `commaai/comma_video_compression_challenge`. **DEFERRED-to-operator**
per CLAUDE.md "Executing actions with care" + "Submission auth eval — BOTH
CPU AND CUDA" non-negotiables: contest PR submission requires (a) hosted
`archive.zip` URL (Cloudflare/Lightning/release manifest — we have local
bytes only); (b) fresh `report.txt` generated from T4-equivalent CUDA auth-eval
on the exact archive bytes; (c) `scripts/pre_submission_compliance_check.py
--contest-final` passing with expected sha + size + runtime-tree + auth-eval
JSON + dispatch-claim linkage. The submission_dir at
`experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/`
contains canonical archive + inflate.sh + inflate.py + README but LACKS the
report.txt + auth-eval JSON required for `--contest-final` mode. Per CLAUDE.md
"Operator gates must be wired and used": running `pre_submission_compliance_check.py
--contest-final --strict` is the non-negotiable gate BEFORE submission.
`canonical_frontier_pointer.submitted_pr_number_for_current_frontier`
remains `null`; operator-routable to either (1) host archive + run paired
CUDA+CPU auth-eval + run compliance gate + invoke `gh pr create
--repo commaai/comma_video_compression_challenge` per `docs/submission_template.md`,
OR (2) re-route the submission to a different sister subagent with the
hosted-URL + report.txt prerequisites resolved. The internal
`tools/create_fork_pr_for_submission.py` tool is for self-eval GHA-CPU
fork PRs to `adpena/comma_video_compression_challenge`, NOT for contest
submission. See `feedback_operator_administrative_bundle_landed_20260519.md`
for the full DEFER blocker report.

## Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS

The target is the best contest-faithful public frontier, not an obsolete
absolute threshold. During an active contest, deadline, or replay window, any
public PR/archive/body/comment/release that plausibly beats the local exact
A++ frontier takes priority over saturated local polish. Claimed public scores
remain `external` until exact CUDA replay, but they must enter intake and exact
replay immediately.

Every frontier action must produce or advance a concrete artifact: candidate
archive, bit-level intake record, dispatch claim, queued exact eval, harvested
JSON, compliance packet, preflight guard, or release/report update. Grand
council and strategy text are advisory only unless they change the next build,
guard, replay, or dispatch.

Deadline mode requires submission escrow: keep a sanitized current-best packet
ready, submit the best exact A++ archive before operator sleep or hard deadline
risk, then update with better replays if they land. Do not wait for the perfect
future candidate when a valid current frontier can be disclosed now.

## Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS

The meta-Lagrangian, Pareto, field-equation, and cross-paradigm selector stack
is a living solver, not a one-off planning report. Any work on score lowering,
stacking, entropy coding, hidden gems, public PR deconstruction, categorical
labels, foveation, pose, sensitivity, or paradigm wiring must either improve
this solver or explicitly record why the new signal is not yet actionable.

Continuously improve the solver toward learnable and solvable theoretical-floor
discovery. No signal loss: keep exact CUDA outputs, archive bytes/SHA,
runtime-tree hashes, commands, assumptions, negatives, calibration residuals,
and cross-paradigm rows machine-readable so future agents can reseed the
planner without reverse-engineering prose.

Every stackable or substitutive idea should move toward a typed row consumed by
the planner: candidate id, family, pareto scope, charged bytes, predicted
SegNet/PoseNet/rate deltas, uncertainty, evidence grade, archive/runtime
custody, interaction assumptions, conflicts, Volterra or higher-order terms,
KKT/ADMM residuals, expected information gain, blockers, and next proof. If a
research artifact can affect score but is not visible to the selector, it is
orphaned work.

Prefer solvable math over arbitrary sweeps. New knobs must be grounded in
entropy/MDL, Fisher/Hessian/Jacobian or Frechet sensitivity, Dykstra/ADMM
feasibility, Bayesian experimental design, optimal transport/camera geometry,
component-response evidence, or a documented ablation. Heuristics stay tagged
`planning_only` until evidence closes the loop. Every exact CUDA result and
high-quality negative should reseed calibration, Pareto constraints, trust
regions, interaction terms, or strict guards.

Every returned result must receive adversarial custody review before it changes
lane status or solver routing. Record the archive bytes/SHA, runtime-tree SHA,
command, hardware, sample count, structured JSON/log path, dispatch-claim
state, recomputed score components, payload-consumption proof, failure class,
and reactivation criteria. A bad result retires only the measured config unless
research-path exhaustion plus consensus review supports a broader conclusion.
Proxy, MPS, non-`contest-CPU` CPU, byte-only, or stale results may seed priors
and TODOs, but they must not promote, rank, falsify, kill, or close a family.
`contest-CPU` ranks only the public leaderboard CPU axis; it does not replace
the CUDA axis or justify extrapolating a missing paired result.

The desired loop is: formulate objective and constraints -> emit typed atoms ->
Pareto/KKT/interaction prune -> select by score delta plus expected information
gain -> build deterministic archive -> exact CUDA eval and exact contest-CPU
eval when the archive is a frontier/submission candidate -> reseed the solver.
Keep this path simpler, faster, more deterministic, and more complete every
time it is touched.

Planner recipes and dispatch snippets must use the current tool surfaces. Grep
the real argparse/help contract before writing or invoking any flag, and record
a blocker or add a reviewed interface when the solver needs a capability the
tools do not expose. Never invent flags, schema keys, or evidence fields to
make a theoretical plan look executable.

## CROSS-AGENT DISPATCH COORDINATION — NON-NEGOTIABLE

**Before dispatching ANY training, eval, or remote-GPU job, claim the lane with `tools/claim_lane_dispatch.py claim ...`.** The helper owns the file lock, reads `.omx/state/active_lane_dispatch_claims.md`, inserts the newest row at the top, and refuses active same-`lane_id` conflicts inside the 24-hour TTL unless an explicit force flag with notes is used.

If you find an active conflicting claim:
- Do NOT dispatch
- Coordinate via the file's notes column or pick a different lane

When your dispatch completes (success or fail): append a terminal row with the
same `lane_id` and `instance/job_id` via `tools/claim_lane_dispatch.py claim
--force --status completed_...`, `--status failed_...`,
`--status stopped_...`, `--status refused_dispatch...`, or a precise
`--status stale_superseded...` row. Do not leave completed jobs as phantom
active claims.

This rule exists because 2026-05-01 ~23:50 UTC the user reported a possible Q-FAITHFUL dispatch conflict between Claude (H100 SXM via Vast.ai) and codex (Lightning) — no formal cross-agent coordination existed and we may have burned $5-10 of duplicate GPU spend. Level 2 is now the norm: use the helper script and strict submitter checks, not manual table edits except for emergency recovery.

## Council hierarchy: 4-tier protocol — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator-approved 2026-05-16 verbatim *"all of that sounds great, we want to ensure maximum signal and continual learning"* on the COUNCIL-HIERARCHY-V2 spec. Extends the existing "Council conduct" sextet pact + "Grand Council (advisory)" 20-seat roster + "Design decisions — non-negotiable" + "Adversarial council review of design decisions" + "Recursive adversarial review protocol" + "META-ASSUMPTION ADVERSARIAL REVIEW" non-negotiables with a structured **4-tier hierarchy** that makes quorum / tie-break / recusal / elevation / operator-attention-budget all explicit. Memory: `feedback_council_hierarchy_v2_landed_20260516.md`. Anchor memos: Catalog #291 (per-session META-ASSUMPTION cadence), Catalog #292 (per-deliberation assumption surfacing), HARD-EARNED-vs-CARGO-CULTED addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`).

### The 4 tiers

| Tier | Name | Quorum | Tie-break | Binding scope | Cadence budget |
|---|---|---|---|---|---|
| **T1** | Working Group | All summoned (1-3 named members) | Working-group lead | Bounded-scope recommendation; NO veto power; output feeds a T2/T3 deliberation | UNBOUNDED (many/day OK) |
| **T2** | Inner-Skunkworks | 5-of-6 sextet (Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary) | Shannon LEAD (information-theory grounding); Dykstra CO-LEAD on optimization-feasibility ties | In-flight engineering tradeoffs; loss-function choices; architecture parameters; trainer wire-ins | ≤3/day, ≤90/30d |
| **T3** | Full Grand Council | 5-of-6 sextet + ≥12-of-20 grand council (existing 12 seats + 8 new 2026-05-15 seats per the roster expansion) | Shannon LEAD; Dykstra CO-LEAD fallback; specialist tiebreaker for paradigm-specific deliberations | CLAUDE.md non-negotiable additions/amendments; cross-cutting wire-ins; strategic redirection within a track | ≤3/week, ≤13/30d |
| **T4** | Symposium | 6-of-6 sextet + ≥16-of-20 grand council + ≥1 specialist seat per affected paradigm/path | Operator-resolves on remaining ties | Kill-and-replace decisions; multi-month directional shifts; operator-pre-attention escalation when council cannot reach consensus | ≤2/30d (≤2/month) |

### Recusal triggers

A council member MUST recuse from a deliberation when ANY of:

1. **Authorship conflict** — the member is the canonical author of the work being deliberated (e.g. Shannon recuses from a deliberation specifically critiquing his information-theory framing for THIS problem; Yousfi recuses from a deliberation on Yousfi-PR-related decisions).
2. **Sister-subagent conflict** — the member is the subagent that produced the artifact under review in the SAME session (prevents lazy self-approval per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" reviewer-vs-author separation).
3. **Prior-position-precommit** — the member has already cast a binding position on the EXACT same question within the last 7 days (forces the council to surface NEW signal in the new deliberation rather than re-litigating).

Recusal is announced at the start of the deliberation; the remaining quorum threshold applies (e.g. T2 with Shannon recused → 5-of-5 of remaining sextet seats; if quorum cannot be met, elevation trigger #2 fires).

### Tier elevation triggers

* **T1 → T2** when ANY of: (a) the working group's recommendation requires changing a loss function / training schedule / scorer routing; (b) the working group encounters a HARD-EARNED-vs-CARGO-CULTED classification disagreement; (c) the working group's empirical finding contradicts a sister T2 anchor on the same topic.
* **T2 → T3** when ANY of: (a) the deliberation touches a CLAUDE.md non-negotiable (addition / amendment / interpretation); (b) recusal drops quorum below 5-of-6; (c) Contrarian veto invoked AND alternative does not reach 4-of-6 consensus within ~30 min; (d) Assumption-Adversary identifies the deliberation as operating within a CARGO-CULTED assumption (per the Catalog #292 + HARD-EARNED-vs-CARGO-CULTED framework) AND the sextet does NOT reach consensus on whether to violate the assumption.
* **T3 → T4** when ANY of: (a) decision is kill-and-replace of an existing substrate/codec class (not a single lane; a class); (b) decision would commit >1 month of operator/agent attention to a new direction; (c) ≥3 grand-council members invoke specialist-disagreement (e.g. Boyd and Tao disagree on convex feasibility; Atick and Tishby disagree on cooperative-receiver framing); (d) recursive adversarial review protocol cycle hits 5 unsuccessful clean-pass attempts on the same topic (per CLAUDE.md "Recursive adversarial review protocol — close paths" R12-D structural unsatisfiability).

### Memory file naming + frontmatter (v2 contract)

Council deliberation memos MUST be named per the canonical pattern:

```
feedback_<grand_council|skunkworks_council|reunion_symposium>_<topic-slug>_<YYYYMMDD>.md
```

The YAML frontmatter MUST include the v2 fields (enforced structurally by Catalog #300 `check_council_deliberation_declares_tier_in_frontmatter`):

```yaml
---
council_tier: T2   # one of T1/T2/T3/T4
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED   # PROCEED / PROCEED_WITH_REVISIONS / DEFER_PENDING_EVIDENCE / REFUSE / ESCALATE_TO_OPERATOR / ESCALATE_TO_HIGHER_TIER
council_dissent:
  - member: Contrarian
    verbatim: "the argument elides the cost-band uncertainty; I want a paired-comparison smoke before commit"
council_assumption_adversary_verdict:   # required at T2+
  - assumption: "EMA decay 0.997 + use shadow at inference"
    classification: HARD-EARNED
    rationale: "PR101 empirical + Quantizr 0.33 anchor"
  - assumption: "canonical scorer-preprocess routing always optimal"
    classification: CARGO-CULTED
    rationale: "untested on this substrate's gradient path"
council_decisions_recorded:
  - "op-routable #1: dispatch paired-comparison smoke at $0.15"
  - "op-routable #2: defer canonicalization until smoke result"
---
```

### Operator-attention budget per tier (over-cadence alerts)

Sustainable cadence per tier (window = 30d unless noted):

* **T1** — UNBOUNDED. Elevation triggers handle the "too many T1s producing crossing findings" case.
* **T2** — ≤3/day, ≤90/30d. Over budget = design tradeoffs are coming faster than 5 humans can deliberate rigorously.
* **T3** — ≤3/week, ≤13/30d. Over budget = CLAUDE.md non-negotiable changes / cross-cutting wire-ins are coming too fast for council coherence.
* **T4** — ≤2/30d. Over budget = strategic redirection is happening too often (strong signal of unstable directional commitment).

Cadence verdicts emitted by `tools/audit_council_tier_cadence.py`:

* **WITHIN_BUDGET** — count ≤ 80% of budget.
* **APPROACHING_LIMIT** — 80% < count ≤ 100% of budget.
* **OVER_CADENCE** — count > 100% of budget. Operator-visible alert: STOP AND CONSOLIDATE. Review whether recent deliberations could have been resolved at a LOWER tier; re-cadence the operator-attention budget.

### Maximum signal preservation rule

Per the operator 2026-05-16 meta-principle *"ensure maximum signal"*, every council deliberation MUST record (NO lossy summarization is permitted):

1. **Verbatim dissent** — every minority opinion preserved verbatim in `council_dissent`. The Contrarian's vote pattern + verbatim is queryable across deliberations via `tac.council_continual_learning.query_dissent_history` (so future deliberations can trace which members have consistently flagged X as cargo-culted).
2. **Per-member operating-within assumption** — at the top of each member's position, the explicit assumption surface required by Catalog #292 + CLAUDE.md "Council conduct" Fix-7 amendment. Lossy paraphrase is FORBIDDEN.
3. **HARD-EARNED-vs-CARGO-CULTED classification** — per surfaced assumption, the Assumption-Adversary's verdict in `council_assumption_adversary_verdict`. Required at T2+. Classifications are queryable across deliberations via `tac.council_continual_learning.query_assumption_classification_history` (so future Assumption-Adversary deliberations can trace classification stability — an assumption flipping HARD-EARNED ↔ CARGO-CULTED across deliberations is a red flag).
4. **Full vote tally** — including abstentions and recusals (e.g. `"T3 grand council: 14 PROCEED / 2 DEFER / 1 REFUSE / 3 recused (authorship); quorum 5-of-6 sextet met; 14-of-17 grand council voted; verdict PROCEED with PROCEED_WITH_REVISIONS dissent recorded"`).
5. **Cite-chain to prior deliberations** — `related_deliberation_ids` lists prior council deliberations on the same topic so future recursive review can trace position evolution. Queryable via `tac.council_continual_learning.query_anchors_by_topic`.

Lossy summarization that drops any of (1)-(5) is a CARGO-CULTED engineering shortcut and a forbidden pattern.

### Continual learning wire-in rule

Per the operator 2026-05-16 meta-principle *"continual learning"*, every T2+ deliberation MUST emit a continual-learning anchor via the canonical helper:

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="<slug>_<YYYYMMDD>",
    topic="<short subject>",
    council_tier=CouncilTier.T2,
    council_attendees=("Shannon", "Dykstra", ...),
    council_quorum_met=True,
    council_verdict="PROCEED",
    council_dissent=({"member": "Contrarian", "verbatim": "..."},),
    council_assumption_adversary_verdict=(
        {"assumption": "...", "classification": "HARD-EARNED", "rationale": "..."},
    ),
    council_decisions_recorded=("op-routable #1: ...",),
)
append_council_anchor(record)   # appends to .omx/state/council_deliberation_posterior.jsonl
```

T1 working groups SHOULD emit an anchor when their finding crosses an elevation trigger (so the downstream T2/T3 deliberation has the prior anchor as side information). T1 outputs that do NOT cross an elevation trigger MAY skip the persisted anchor (recommendation is captured in the deliberating T2/T3's memo body).

The persisted council verdicts + dissent + assumption classifications become signal that:

* Future deliberations consume via `query_anchors_by_topic` (cite-chain detection).
* The autopilot ranker consumes via the upcoming `tac.cathedral_autopilot_*` hook for council-verdict-aware candidate weighting.
* The Rashomon ensemble (Catalog #252 sister) consumes via fcntl-locked posterior reads.
* The Assumption-Adversary consumes via `query_assumption_classification_history` (classification-stability monitoring).

### Mission alignment — non-negotiable

**Source:** operator binding standing directive 2026-05-16 verbatim *"and all in service of innovation and rigor and extreme optimization and performance and score lowering"*. Anchor memo: `feedback_council_apparatus_in_service_of_innovation_rigor_optimization_score_lowering_20260516.md`.

**Discipline serves the mission, NOT the reverse.** The 4-tier council hierarchy + maximum-signal preservation + continual-learning wire-ins + per-tier cadence budgets are INFRASTRUCTURE for innovation, NOT replacements for it. When procedural rigor blocks a frontier-breaking move, the rigor yields — not the mission. Operator-frontier-override at ALL tiers (T1-T4) is the documented escape hatch.

This subsection pairs with:

* CLAUDE.md "Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the target IS the best public frontier; not an obsolete absolute threshold).
* CLAUDE.md "Council conduct — non-negotiable" (council MUST NEVER have a conservative bias).
* CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the apparatus adapts when the leaderboard moves).
* CLAUDE.md "Forbidden premature KILL without research exhaustion (the kill-too-fast trap)" (deferred substrates get re-audited).

#### 5 operational consequences (binding across all tiers)

1. **Operator-frontier-override at ALL tiers** — every tier (T1 / T2 / T3 / T4) supports operator-frontier-override as a documented escape hatch when the operator declares a time-critical innovation cannot wait for tier-required quorum or sextet pact. The override (a) BYPASSES quorum + tie-break + recusal rules for the specific decision; (b) REQUIRES operator-verbatim quote in the council memo's `council_override_rationale:` frontmatter field (enforced by Catalog #300 paired-field validation); (c) PRESERVES maximum-signal preservation (dissent still recorded; assumption classification still done; continual-learning anchor still emitted); (d) TRIGGERS a 30-day score-impact retrospective per consequence 3.

2. **Annual gate audit by empirical score contribution** — every Catalog # STRICT preflight gate undergoes an annual audit where the operator (or operator-spawned subagent) evaluates: *"What empirical incidents did this gate prevent in the last 12 months? How many false positives blocked real innovation? What's the gate's net score contribution?"* Gates with negative net contribution (more innovation-blocking than bug-prevention) are candidates for retirement or scope-narrowing per Catalog #299 (gate consolidation discipline). The audit cadence: every year on the catalog's landing anniversary; an `annual_gate_audit_alert` in `tools/audit_council_tier_cadence.py` surfaces gates whose landing date is ≥ 365 days ago AND no audit-verdict memo exists at `.omx/research/annual_gate_audit_catalog_<N>_<YYYY>.md`.

3. **30-day score-impact retrospective on every deferred/killed substrate** — every substrate that received a DEFERRED / KILL / research_only verdict (across any tier deliberation) MUST be re-audited 30 days later for: *"Did the deferral cost us actual score improvement? Did a sister substrate land in its place that captures the same gain? Should the deferral be reconsidered?"* Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "KILL/FALSIFIED memory verdicts" non-negotiable. The retrospective surfaces lanes where the discipline apparatus over-protected against perceived risk and under-served the mission. Tracked via `deferred_substrate_retrospective_due_utc` + `deferred_substrate_id` fields on the council deliberation record + the `overdue_retrospective_alert` in the cadence audit tool.

4. **Frontier-breaking moves DOMINATE rigor budget** — when the contest leaderboard moves (per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first") OR an empirical anchor reveals a sub-A1 frontier-breaking opportunity, the council apparatus + discipline gates MUST adapt: parallel-dispatch first; rigor compressed; council deliberation foregrounds the time-critical decision; operator-override invoked liberally. The May 4 2026 race postmortem is the empirical anchor: PR105's kitchen_sink (1776 LOC) lost to rem2's 241 LOC because rigor outpaced velocity.

5. **Every T2+ verdict includes `council_predicted_mission_contribution:` frontmatter** ∈ `{frontier_breaking, frontier_protecting, rigor_overhead, apparatus_maintenance, mission_questioned}` (enforced by Catalog #300). Category semantics:
   * `frontier_breaking` — the verdict opens a class-shift path predicted to lower score.
   * `frontier_protecting` — the verdict prevents a regression that would raise score (sister of strict-mode preflight gates).
   * `rigor_overhead` — the verdict is procedural-only; no direct score contribution but enables future contributions.
   * `apparatus_maintenance` — the verdict updates infrastructure without score implications.
   * `mission_questioned` — the verdict triggered the "is this serving the mission?" question; documented for retrospective.

   Operator-visible alert when `rigor_overhead + apparatus_maintenance > 60%` of T2+ verdicts in any 30-day window — the council is producing more apparatus-maintenance than frontier-breaking work and operator review is required. Surfaced by `tools/audit_council_tier_cadence.py::compute_mission_contribution_distribution_alert` + sister `tac.council_continual_learning.is_rigor_dominant`.

#### Concrete enforcement

* Catalog #300 STRICT preflight gate extended to require the 3 mission-alignment frontmatter fields at T2+: `council_predicted_mission_contribution` (enum-validated against the 5 categories) + `council_override_invoked` (boolean; REQUIRED field even when false) + `council_override_rationale` (REQUIRED when `council_override_invoked: true`; verbatim operator quote).
* `tac.council_continual_learning.CouncilDeliberationRecord` dataclass extended with the 5 mission-alignment fields (`predicted_mission_contribution` / `override_invoked` / `override_rationale` / `deferred_substrate_retrospective_due_utc` / `deferred_substrate_id`); validated in `__post_init__` via `CouncilRecordValidationError` so the helper-construction surface refuses missing/malformed fields BEFORE persistence.
* Query helpers (`query_overrides` / `query_due_retrospectives` / `query_mission_contribution_distribution` / `is_rigor_dominant`) surface the audit surfaces for downstream consumers (cathedral autopilot ranker / Rashomon ensemble / Assumption-Adversary).
* `tools/audit_council_tier_cadence.py` extended with 3 new alert classes (`mission_contribution_distribution_alert` / `overdue_retrospective_alert` / `annual_gate_audit_alert`); CLI rc=1 when ANY mission-alignment alert is fired (sister of the existing rc=1 on `any_over_cadence`).

### Backward compatibility (hybrid backfill)

Pre-2026-05-16 council memos are EXEMPT from Catalog #300 frontmatter requirements (the v2 fields did not exist when those memos were written). However, the ≤10 most-actively-cited pre-cutoff council memos are backfilled with v2 frontmatter (NO body mutation per Catalog #110/#113 HISTORICAL_PROVENANCE discipline; only the YAML frontmatter is added/extended) AND persisted as continual-learning anchors via `append_council_anchor` so the autopilot ranker / Rashomon ensemble can see the canonical historical baseline.

Mission-alignment backfill: HISTORICAL anchors lacking explicit `predicted_mission_contribution` data are loaded with `apparatus_maintenance` as the backfill default (most common historical pattern; safe default per the mission-alignment binding directive). The legacy v1 rows are preserved per Catalog #110 HISTORICAL_PROVENANCE; the backfill manifests as NEW rows referencing the same `deliberation_id` with `event_type="backfilled_extension"` so query helpers see the extended fields via `latest-row-wins` semantics.

### Concrete enforcement

* STRICT preflight Catalog #300 (`check_council_deliberation_declares_tier_in_frontmatter`) refuses post-2026-05-16 council memos lacking v2 frontmatter. WARN-ONLY at landing per "Strict-flip atomicity rule"; strict-flip planned after 5 deliberations land in v2 format.
* Sister Catalog #292 (`check_grand_council_deliberation_has_explicit_assumption_statements`) enforces per-DELIBERATION body-level assumption surfacing.
* `tools/audit_council_tier_cadence.py` emits OVER_CADENCE alerts per tier; CI / operator-authorize harness may consult to gate over-budget tier dispatches.
* `tac.council_continual_learning` is the canonical wire-in; bare writes to `.omx/state/council_deliberation_posterior.jsonl` outside the canonical helper are refused by Catalog #131 sister discipline.

### Cross-references

* "Council conduct" — sextet pact baseline + per-round explicit-assumption-statement discipline.
* "Grand Council (advisory)" — 20-seat roster expansion (12 existing + 8 new 2026-05-15 seats).
* "Design decisions — non-negotiable" — quintet pact base + council-grade tradeoff requirement.
* "Adversarial council review of design decisions" — the canonical council-deliberation pattern this hierarchy operationalizes.
* "Recursive adversarial review protocol" — the per-round assumption-challenge axis + 3-clean-pass discipline that T2+ deliberations inherit.
* "META-ASSUMPTION ADVERSARIAL REVIEW" — the per-session cadence Catalog #291 enforces.
* "Bugs must be permanently fixed AND self-protected against" — the structural-protection pattern Catalog #300 + #292 + #291 jointly satisfy.
* `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` — the canonical HARD-EARNED-vs-CARGO-CULTED framework T2+ Assumption-Adversary verdicts use.
* `feedback_council_hierarchy_v2_landed_20260516.md` — this section's landing memo (eats own dogfood; carries v2 frontmatter as the first T3 deliberation in v2 format).

## Max observability — non-negotiable

**Source:** operator binding standing directive 2026-05-16 verbatim *"the xray and autopilot and all tools and the experiment and designs themselves should be built so as to support absolute max observability into behavior"*. Anchor memo: `feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`.

**Every substrate design + experiment + tool + canonical helper + dispatch wrapper MUST be built with absolute maximum observability into behavior.** Sister of the mission-alignment directive (apparatus serves mission); together they bind: discipline serves the mission (per the mission-alignment section above) AND the apparatus exposes ITS OWN behavior maximally (this section). Observability is INFRASTRUCTURE for innovation; without observability the apparatus cannot evaluate its own behavior + cannot enable continual learning + cannot serve the mission.

### The 6-facet observability definition (binding)

Behavior is observable when it is:

1. **Inspectable per layer** — every layer's input + output + intermediate state can be captured at runtime without re-instrumentation (xray-style).
2. **Decomposable per signal** — composite metrics (e.g. `final_score`) can be decomposed into constituent contributions (seg + pose + rate; per-pair / per-class / per-axis / per-stage).
3. **Diff-able across runs** — two runs of the same substrate can be byte-level + activation-level + score-level diffed to surface drift.
4. **Queryable post-hoc** — the run artifacts support arbitrary queries without needing to re-run (machine-readable JSON / JSONL / SQLite / TensorBoard event files / Modal artifacts; NOT just stdout grep).
5. **Cite-able** — every behavior signal is anchored to a (substrate / commit / call_id / config / random_seed / upstream_snapshot_sha256) tuple per Catalog #245 modal_call_id_ledger.
6. **Counterfactual-able** — the byte-mutation discipline (Catalog #139 packet compiler + #272 distinguishing-feature contract + #105 no-op detector) allows asking "what if this byte changed?" without re-running training.

### 5 operational consequences (binding across all surfaces)

1. **Every substrate design memo declares its OBSERVABILITY SURFACE.** Per the 9-dim checklist + UNIQUE-AND-COMPLETE-PER-METHOD operating mode, every substrate design memo MUST include an `## Observability surface` section listing per-layer inspection hooks + per-signal decomposition + run-to-run diff manifest + post-hoc query interface + cite-chain + counterfactual hooks. STRICT preflight Catalog #305 refuses substrate design memos lacking this section.

2. **Every canonical helper + tool emits structured observability.** Per the existing fcntl-locked JSONL store pattern (cost_band_posterior / modal_call_id_ledger / council_deliberation_posterior / subagent_progress / continual_learning_posterior), every NEW canonical helper / tool MUST emit structured observability per invocation: input snapshot + output snapshot + wall-clock + CPU/memory + decision-path + cite-chain. Sister of CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable: *"make artifacts human-readable where possible and machine-checkable always"*.

3. **Existing infrastructure audit + extension landings.** Existing tools MUST be audited for observability gaps + extended. The canonical audit tool is `tools/audit_existing_infrastructure_for_observability.py` (scoring 8 tools across 6 facets; current overall = 69.8%); the highest-ROI extension target is the `tools/audit_*.py` family (3/12 observability — biggest gap). Specific extensions queued: `tac.sensitivity_map` cite-chain backfill (commit + call_id + upstream_snapshot_sha256 per axis-weight row); `src/tac/xray/wire_in.py` completion per #711 ORPHAN-SIGNAL-AUDIT (register each lens with cathedral_autopilot ranker); canonical `AuditReport` dataclass for the audit_* family.

4. **Per-experiment observability budget.** Every paid dispatch (Modal / Vast.ai / Lightning) MUST emit observability artifacts as harvested return values, NOT just final score. The Modal `.spawn()` HARVEST OR LOSE non-negotiable already requires harvest; this directive extends: harvest MUST include the observability surface per consequence 2. Per CLAUDE.md "Apples-to-apples evidence discipline" + "Bit-level deconstruction and entropy discipline" — these are observability disciplines applied to evidence + bytes respectively. This directive generalizes to ALL behavior.

5. **Observability-driven design choices (architectural implication).** When designing a new substrate, prefer architectures whose behavior is structurally observable over architectures whose behavior is opaque. Falling-rule lists / SLIM / GOSDT (Rudin interpretability per Catalog #273-#278) are observability-MAX. Black-box neural networks are observability-LOW unless paired with per-layer activation hooks + saliency / sensitivity-map + Hinton-distilled scorer surrogate. Joint codec architectures (NSCS03 Ballé end-to-end) need explicit hooks at the rate/distortion decomposition surface. Composition substrates (A-STACK) need per-substrate slot observability so the composition's failure mode is decomposable per-substrate.

### Concrete enforcement

* STRICT preflight Catalog #305 (`check_substrate_design_memo_has_observability_surface_section`) refuses substrate design / landing / composition memos dated >= 2026-05-16 that lack the literal section header `## Observability surface`. Initial MAX-OBSERVABILITY landing was warn-only while the OBSERVABILITY-ADDENDUM backfill drove the count to 0; WAVE-1 APPARATUS HARDENING strict-flipped it on 2026-05-16 after the 3 sister-landed design memos were backfilled.
* `tools/audit_existing_infrastructure_for_observability.py` is the canonical audit surface — operator-runnable any time; CLI emits JSON or `--summary` human-readable; scores 8 canonical tools across the 6 facets.
* Sister of Catalog #290 (canonical-vs-unique decision per layer — Dimension 5) + Catalog #294 (9-dim success checklist evidence) + Catalog #303 (cargo-cult audit section) + Catalog #296 (predicted-band Dykstra feasibility) — together they close substrate-design-memo discipline across 5 orthogonal axes.

### Cross-references

* "Mission alignment — non-negotiable" — sister directive (apparatus serves mission); together they bind discipline + observability serve the mission.
* "9-dimension success checklist evidence" — Dimension 8 (EXTREME OPTIMIZATION + PERFORMANCE) gains an observability axis.
* "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — canonical helpers MUST serve observability as part of "serves".
* "Beauty, simplicity, and developer experience" — observability is the structural manifestation of "make artifacts human-readable where possible and machine-checkable always".
* "Apples-to-apples evidence discipline" — observability discipline applied to evidence.
* "Bit-level deconstruction and entropy discipline" — observability discipline applied to bytes.
* Catalog #245 modal_call_id_ledger — observability discipline applied to dispatches.
* Catalog #128/#131 fcntl-locked JSONL store pattern — observability discipline applied to state mutations.
* Catalog #273-#278 Rudin-Daubechies autopilot — observability via interpretability (SLIM + falling-rule + Rashomon ensemble + GOSDT decision-path).
* `feedback_max_observability_landed_catalog_305_20260516.md` — this section's landing memo (eats own dogfood; carries v2 frontmatter + the `## Observability surface` section).

## Canonical equations + models registry — NON-NEGOTIABLE

**Source of truth for systems of equations + predictive models that codify empirical findings:** `.omx/state/canonical_equations_registry.jsonl` (canonical JSONL ledger; per-equation `EmpiricalAnchor` rows track calibration residuals; auto-recalibration trigger when new continual-learning anchors land per Catalog #344).

**FORBIDDEN**: introducing a new empirical-finding memo (predicted vs measured / ratified / falsified / refined / etc.) without ALSO registering the underlying canonical equation in `tac.canonical_equations` OR carrying `# FORMALIZATION_PENDING:<rationale>` waiver (placeholder rationale rejected per Catalog #287 sister discipline). Operator's 2026-05-19 verbatim: *"we need to formalize all of this and canonicalize and operationalize because I am afraid we are learning but if we don't have systems of equations and models and such we are just gaining tribal knowledge"*. The framework extincts tribal knowledge structurally.

**Operator-facing access**:
- `tools/list_canonical_equations.py` — human-readable registry view; `--json` for machine-readable
- `tools/recalibrate_equation.py --equation-id <id>` — trigger calibration refresh from new continual-learning anchors
- `python -c "from tac.canonical_equations import query_equations; ..."` — programmatic query

**Producer→consumer audit**: every equation declares `canonical_producers` + `canonical_consumers` lists; the framework refuses orphan equations (no producers AND no consumers) per the `CanonicalEquation.__post_init__` invariant. Sister of Catalog #265 / #335 (canonical contract auto-discovery) + Catalog #245 / #313 (canonical 4-layer ledger pattern this registry follows).

**Initial population (2026-05-19)**: 6 canonical equations covering brotli cascade bounded-per-stream (commit 0696a1488) / MPS drift architecture-class dependent (commit 65db9f570 slot 16) / per-byte leverage uniformly distributed (convergent multi-signal) / per-pair master-gradient Taylor + Cauchy-Schwarz (commit ab7f8f7e2 slot 9) / master-gradient locality violation by codec (slot 15+17+18) / canonical frontier pointer (slot 14 commit 023a2374f). Each equation carries Provenance per Catalog #323 + ≥1 EmpiricalAnchor backed by a canonical artifact path.

## Operator gates must be wired and used — NON-NEGOTIABLE

Recovered tools are not done when the source file exists. They must be wired
into normal operator flows and documented where future agents will actually see
them.

Required gates:

- `tac.preflight.preflight_all()` includes
  `check_dispatch_cli_shell_hazards(strict=True)`.
- `tools/all_lanes_preflight.py` runs
  `tools/check_dispatch_cli_shell_hazards.py --strict` before lane dry-runs.
- Before any judge-facing or public submission packet, run
  `scripts/pre_submission_compliance_check.py --contest-final --strict` with
  explicit `--expected-archive-sha256`, `--expected-archive-size-bytes`, the
  canonical auth-eval JSON, archive manifest, and any dispatch-claim linkage.

These gates close concrete bug classes: `--rmote` and other dead/typo flags,
adjudicator-only flags passed to Lightning launchers, zsh `path` mutation,
GNU-only `find -printf` in local/macOS surfaces, unsafe ZIP names, stale
archive manifests, CPU/MPS promotion leakage, missing runtime-tree custody,
and public supplement provider/private-path leaks.

If you create a new profiler, packer, recovery script, hidden-gem tool, or
submission checker, wire it into `preflight_all()`, `tools/all_lanes_preflight.py`,
`tools/operator_briefing.py`, a runbook, or a dated `.omx/research/` ledger.
Do not leave high-signal tooling buried under an obscure filename.

### State JSONL archival policy (NEW 2026-05-16 — premortem #10)

`.omx/state/*.jsonl` files exceeding 10 MB MUST archive older rows to
`.omx/state/archive/<filename>_<YYYY-MM>.jsonl` and keep only the most-recent
90-day window in the live file. The canonical helper is
`tools/archive_jsonl_state.py` which scans `.omx/state/` for over-size JSONLs,
partitions rows by their UTC timestamp field, archives older rows under fcntl
LOCK_EX, atomically rewrites the live file (write-tmp + os.replace per Catalog
#128 / #131 / #245 sister discipline), and updates the archive manifest at
`.omx/state/archive/_index.json`.

The existing append-only ledgers honor the policy:

- `.omx/state/commit-serializer.log` (Catalog #117 / #157 / #174 sources)
- `.omx/state/modal_call_id_ledger.jsonl` (Catalog #245 source)
- `.omx/state/subagent_progress.jsonl` (Catalog #206 source)
- `.omx/state/cost_band_posterior.jsonl` (Catalog #175 / #177 source)
- `.omx/state/lane_maturity_audit.log` (Catalog #90 source)

Per the 12-month premortem (`.omx/research/12_month_frustration_premortem_and_
recommendations_20260516.md` Category L + Section 3 #10): the 12-month
manifestation without this policy is `.omx/state/` >5 GB; serializer log scans
in Catalog #117/#157/#174/#206/#289 become hot-path bottlenecks; `git status`
performance degrades. No STRICT preflight gate for now — operational hygiene
enforced by the monthly `tools/archive_jsonl_state.py --apply` operator
cadence.

Usage:

```bash
# Preview the archival plan:
.venv/bin/python tools/archive_jsonl_state.py

# Execute (creates .omx/state/archive/<filename>_<YYYY-MM>.jsonl):
.venv/bin/python tools/archive_jsonl_state.py --apply

# Per-file with custom retention:
.venv/bin/python tools/archive_jsonl_state.py \\
    --target .omx/state/commit-serializer.log --retain-days 60 --apply
```

## Substrate retirement discipline — NON-NEGOTIABLE

**Source:** 12-month premortem (`.omx/research/12_month_frustration_premortem_
and_recommendations_20260516.md` Category E + Section 3 #1) +
PREMORTEM-CONSOLIDATION-WAVE landing 2026-05-16. Per the premortem: at current
cadence we project 200+ stale L1 SCAFFOLD substrate lanes by 2027-05-16; the
cathedral autopilot ranker over-fits on the 200+ scale; every dispatch wave
that DOES fire spends $20-100 chasing one substrate, leaving the other 199
untouched; the "research-substrate trap" (8th forbidden pattern per HNeRV
parity discipline) becomes systemic.

### The rule

Every substrate at L1+ for >30 days without paid dispatch (or `lane_maturity.py
mark` activity) MUST satisfy one of:

1. **Advance to L2+** with empirical anchor (real-archive empirical / contest-
   CUDA gate satisfied). The lane is no longer L1 by definition.
2. **Declare `research_only: true`** with `reactivation_criteria` pinned in
   lane registry notes per CLAUDE.md "HNeRV / leaderboard-implementation
   parity discipline" lesson 2 (export-first design).
3. **Move to `archived/` directory** with terminal verdict + reactivation
   criteria. Declared via top-level `archived: true` field OR `lane_state=
   archived` token in notes per CLAUDE.md "Forbidden premature KILL" — the
   lane is not killed, it is archived with explicit reactivation criteria.
4. **Same-line waiver** `# RETIREMENT_DISCIPLINE_WAIVED:<rationale>` in lane
   notes / evidence for the rare deliberate operator-approved deferral
   (placeholder `<rationale>` literal rejected).

### Concrete enforcement

- STRICT preflight Catalog #298 (`check_substrate_lane_l1_scaffold_not_stale_
  dispatch`) — refuses in-scope L1 substrate lanes with `impl_complete=true`
  and no audit-log activity within 30 days unless one of the 4 opt-outs
  applies. WARN-ONLY at landing per "Strict-flip atomicity rule"; strict-flip
  pending operator-routed backfill sweep of existing stale L1 lanes.
- Canonical operator-facing audit tool: `tools/audit_stale_l1_substrates.py`
  produces the monthly retirement candidate list with per-lane verdict
  (ACTIVE_RECENT_DISPATCH / ACTIVE_RECENT_MARK / STALE_PENDING_DECISION /
  OPT_OUT_RESEARCH_ONLY / OPT_OUT_SUBSTRATE_ENGINEERING / OPT_OUT_ARCHIVED).
- Pair with the "stop adding new substrates without retiring one" informal
  discipline (premortem Section 5 anti-pattern #2): new substrate adoption
  requires explicit retirement of an L1 SCAFFOLD that has been idle >60 days.

### Cross-references

Sister of Catalog #220 (L1 scaffold operational mechanism declaration) +
Catalog #272 (distinguishing-feature integration contract) + Catalog #233
(L1->L2 promotion canonical 4-gate). Together they extinct the "L1 SCAFFOLD
substrate accumulates indefinitely without dispatch resolution" failure mode.
Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the
default verdict is DEFERRED-pending-research / `research_only=true`, NEVER
KILL. Archived state is dormant-with-reactivation, not kill.

## Gate consolidation discipline — NON-NEGOTIABLE

**Source:** 12-month premortem (`.omx/research/12_month_frustration_premortem_
and_recommendations_20260516.md` Category A + Section 3 #5 + Section 5
anti-pattern #1) + PREMORTEM-CONSOLIDATION-WAVE landing 2026-05-16. Per the
premortem: at current cadence (~5-10 new gates/week) the catalog # crosses
500 by Q4 2026 and 700+ by 2027-05-16. `preflight.py` swells past 100K LOC.
Operator-authorize harness's 30-second budget breaks. New gates get added but
reviewed only by their author. META-meta phantom-row incidents (the Catalog
#273-#278 phantom landings 2026-05-15) become the rule, not the exception.

### The rule

No new STRICT preflight gate may be claimed past **Catalog #400** without one
of:

1. **Retirement of an existing strict gate.** Mark the old gate DEPRECATED in
   the CLAUDE.md catalog table; remove the orchestrator callsite from
   `preflight_all()` after a 30-day deprecation window.
2. **Operator-explicit "exceed-quota" approval with rationale** — file-level
   `# CATALOG_QUOTA_EXCEEDED_OK:<rationale>` waiver in the first 200 lines of
   CLAUDE.md (placeholder rationales `<rationale>` / `<reason>` rejected).
3. **The new gate REPLACES an existing strict gate.** Delete the old entry +
   its orchestrator callsite in the same commit batch.

New gates that would push count past #400 trigger an explicit "stop and
consolidate" pause: review the existing 295+ gates for retirement candidates
BEFORE adding the new one. The pause is operator-facing — the agent surfaces
the candidate retirements, the operator decides.

### Concrete enforcement

- STRICT preflight Catalog #299 (`check_catalog_quota_under_400`) — refuses
  CLAUDE.md catalog table entries above the #400 quota without the file-level
  waiver. Initial wire-in is WARN-ONLY because the current registered catalog
  max is ~299 so the quota is not yet binding. Strict-flip planned when
  catalog # actually approaches 400 (so the operator gets the explicit "stop
  and consolidate" pause at the right moment).
- Sister of Catalog #118 (no duplicate numbers) + Catalog #159 (catalog text
  matches strict value) + Catalog #176 (strict callsites have CLAUDE.md row)
  + Catalog #185 (LIVE_COUNT drift detection) + Catalog #186 (catalog claim
  committed via serializer). Together they extinct the catalog # drift /
  phantom-row / exhaustion failure modes at FIVE surfaces: number uniqueness
  (#118) + text-strict parity (#159) + callsite-has-row (#176) + live-count
  drift (#185) + claim-transactional (#186) + quota brake (#299).

### Cross-references

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
EVERY new STRICT gate's introduction MUST evaluate whether it could be
written as a META-meta gate that subsumes >=3 sister cases (one gate kills
three bug classes) BEFORE landing. Pure-additive gate landings are the slow
death per premortem Section 5 anti-pattern #10.

## Memory file rotation discipline — NON-NEGOTIABLE

**Source:** 12-month premortem (`.omx/research/12_month_frustration_premortem_
and_recommendations_20260516.md` Category D + Section 3 #6) +
PREMORTEM-CONSOLIDATION-WAVE landing 2026-05-16. Per the premortem: memory
directory exceeds 3,000 files by 2027-05-16; MEMORY.md grows past 1,000
indexed lines; only top ~50 are loaded at session start; cross-reference graph
rots silently; new subagents repeatedly rediscover lessons the system already
learned.

### The rule

Operator + agent memory hygiene cadence:

1. **MEMORY.md index lines exceed 200 chars** after summarization → triage to
   category-summary file via `tools/cluster_summarize_memory_category.py
   --category <prefix> --write-skeleton`. The skeleton is filled in (operator
   or LLM-using subagent writes 2-3 paragraph cluster summary), then the
   constituent MEMORY.md index entries are replaced by a single pointer to
   the cluster memo. Detail stays in originals; the cluster is the new
   top-of-mind summary.
2. **Memory files older than 60 days superseded by a newer memo** → mark
   `superseded_by: <newer_name>` in YAML frontmatter (or the first 30 body
   lines). The `tools/audit_memory_file_freshness.py` tool surfaces
   candidates monthly.
3. **Broken cross-references** (memo X cites memo Y by name; Y renamed and
   no longer on disk) → rename Y back OR update X to point at the canonical
   successor. The audit tool's "broken_references" verdict surfaces these.

### Concrete enforcement (operational, NO STRICT gate)

- Canonical audit: `.venv/bin/python tools/audit_memory_file_freshness.py`
  produces the 3-class report (stale-by-age + index-line-overflow +
  broken-references).
- Canonical cluster helper: `.venv/bin/python tools/cluster_summarize_memory_
  category.py --category feedback_grand_council_ --write-skeleton` writes
  `.omx/research/MEMORY_CLUSTER_<category>_<YYYYQQ>.md` skeleton.
- No STRICT preflight gate at landing — operational hygiene only. The
  operator runs the audit monthly; the agent surfaces candidates in
  session-start summaries.
- Per premortem Section 5 anti-pattern #5: do NOT add new memory file
  category prefixes (`council_audit_*`, `decision_log_*`, etc.) — extend the
  audit + cluster tools first to handle new prefixes structurally.

## Preflight failure messages must cite the rule chain — NON-NEGOTIABLE

**Source:** Catalog #273-#278 META principle (2026-05-15) — see
`feedback_rudin_daubechies_preflight_composite_landed_20260515.md`. Mirrors the
sister CLAUDE.md "Comment-only contracts are FORBIDDEN" non-negotiable from
gate output to gate failure-message output.

Every preflight gate's failure message MUST cite the rule chain that fired AND
the recommended fix as a rule chain. Comment-only failure descriptions are
FORBIDDEN per CLAUDE.md "Comment-only contracts are FORBIDDEN" extended to
gate output. The catalog table itself is a falling-rule list with hit-rate
sorting (per Wang & Rudin 2015 "Falling Rule Lists" canonical discipline).

Operationalized via `tac.preflight_rudin_daubechies` package:

- `tac.preflight_rudin_daubechies.PreflightSLIMRiskScorer.explain(panel)` —
  the canonical rule-chain readback per Rudin's interpretability principle:
  `predicted_dispatch_risk N = intercept(0) + 25*gate_146(VIOLATED) + 0*gate_167(PASSED) + ...`
  Every term auditable by eyeball arithmetic.
- `tac.preflight_rudin_daubechies.PreflightFallingRule.recommended_fix` field —
  every rule MUST carry the operator-readable fix as a rule chain element.
- `tac.preflight_rudin_daubechies.GOSDTDispatchRouter.decide(...).explain()` —
  the canonical decision-path readback per the Lin-Zhong-Hu-Hu-Rudin-Seltzer
  2020 GOSDT discipline: `decision_path: cost_band==smoke AND substrate_class==score_aware AND ... -> ACTION (rationale: ...; predicted band [...])`.

Anti-pattern: docstring-only failure descriptions ("the gate fired because the
contract is wrong"). The gate output MUST include the named rule that fired,
its concrete value (which threshold / which token), and the operator-actionable
recommended fix (canonical helper to call, env-var to set, file to edit). The
output IS the documentation; no hidden state, no "see the gate's source for
why."

## Production-hardened dispatch optimization protocol — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator directive 2026-05-15 *"remember the multiple deployments
that have failed over and over because of missing optimizations? we should
investigate and develop a protocol for those too and enforce best practices
and production hardened optimization, extreme optimization and correctness
and performance and scalability"*.

Every paid Modal / Vast.ai / Lightning dispatch MUST satisfy the canonical
**dispatch optimization protocol** = AND(Tier 1 engineering primitives,
Tier 2 hardware correctness, Tier 3 substrate correctness). Empty
conjunction = REFUSE dispatch.

### Empirical anchors (the audit)

- **D1 NVML 999** — `substrate_d1_segnet_margin_polytope_modal_t4_dispatch
  _..._smoke__50ep` 2026-05-15T08:26:38Z crashed at `nvml error (999)`
  inside DALI `fn.experimental.inputs.video` because the lane driver was
  missing the canonical 3-export Modal/CUDA env block. 6th occurrence of
  the same bug class in 24h on D1 alone (closed by Catalog #244).
- **Z3 v2 + Z4 paired $2 waste** — `fc-01KRNHEGC9ZE48Y68GGJHP7FXN` + `fc-
  01KRNHE942JSV7VRGXGR1FJGHQ` 2026-05-15 both crashed on recipe-vs-trainer-
  state divergence (Z3 v2 / Z4 / Z5 bug class; closed by Catalog #240).
- **Z3 v2 phantom CUDA score** — Modal A100 2026-05-15T11:41:15Z wrote
  CPU eval results to `contest_auth_eval_cuda.json` because the trainer
  hardcoded `_cuda` suffix; parent agent quoted "0.19869 [contest-CUDA T4]" <!-- HISTORICAL_SCORE_LITERAL_OK:z3_v2_phantom_cuda_score_anchor_2026-05-15_catalog_249_landing -->
  from filename despite metadata saying CPU. Paired re-eval revealed true
  CUDA = 0.2317 (closed by Catalog #249).
- **C6 auth_eval rc=2** — `fc-01KRKG566Z2F48CVCGF8JFA0S1` 2026-05-14 5ep
  smoke crashed because the trainer hand-wrote `--archive-zip` / `--output-
  json` flags that don't exist on the canonical contest auth-eval CLI
  (closed by Catalog #226 18-trainer refactor wave).
- **T1 Balle 23h timeout** — `fc-01KR955JSYQAVTTYZA48VAV7WJ` 2026-05-10
  timed out rc=124 at 84,608s wall-clock for missing autocast fp16 + TF32
  + torch.compile (Tier 1 engineering hygiene; closed by Catalog #172 +
  #178 + #179 + #180 sister wave).
- **D4 T4 OOM** — `fc-01KRK9RKD3QV4C276Y5KXFMF65` 2026-05-14 121s OOM at
  T4 14.56GB capacity because `reconstruct_pair` ran 600-pair forward at
  full resolution (closed by Catalog #218 + #170 min_vram_gb declaration).
- **Z3 Balle / 30+ legacy substrate Tier 1 gaps** — current audit finds
  26 of 32 scanned substrate trainers fail at least one tier signal
  (typically Tier 1 canonical scorer loss helper routing missing OR
  Tier 2 recipe `min_smoke_gpu` undeclared); WARN-ONLY at landing per
  the "Strict-flip atomicity rule" pending operator-routed backfill.

### The three tiers

**Tier 1 — Engineering primitives** (the trainer MUST declare / use):
autocast_fp16 (Catalog #172), TF32 (Catalog #178), torch.compile
(Catalog #179), no_grad-at-eval (Catalog #180), GTScorerCache F3
consumption (Catalog #228), canonical scorer-loss helper routing
(Catalog #164 — `tac.substrates._shared.score_aware_common.score_pair_components`).

**Tier 2 — Hardware correctness** (the recipe + lane driver MUST declare):
min_vram_gb (Catalog #170), min_smoke_gpu (Catalog #215), video_input_
strategy (Catalog #171), pyav_decode_strategy (Catalog #181), target_modes
(Catalog #182), canonical 3-export NVML/CUDA env block at the lane driver
(Catalog #244 — `DALI_DISABLE_NVML` + `CUBLAS_WORKSPACE_CONFIG` +
`PYTORCH_CUDA_ALLOC_CONF`).

**Tier 3 — Substrate correctness** (the trainer + recipe MUST be consistent):
canonical auth-eval helper routing (Catalog #226 — `gate_auth_eval_call`),
canonical inflate device (Catalog #205 — `select_inflate_device`),
scorer-loader assignment order `(posenet, segnet) = ...` (Catalog #222),
recipe-vs-trainer-state consistency (Catalog #240 — research_only tag
when `_full_main` raises NotImplementedError), no phantom device-named
output directories (Catalog #249).

### Concrete enforcement

- Canonical helper `tools/canonical_dispatch_optimization_protocol.py::verify_dispatch_protocol_complete(trainer, recipe)` returns a typed `ProtocolVerdict` with per-tier verdicts + overall pass/fail + machine-readable blockers list.
- STRICT preflight gate **Catalog #270** `check_dispatch_optimization_protocol_complete` ANDs over every scanned `experiments/train_substrate_*.py`; refuses any state with a non-passing trainer; same-line `# DISPATCH_OPTIMIZATION_PROTOCOL_OK:<rationale>` waiver in the trainer file's first 30 lines (placeholder rationales rejected); WARN-ONLY at landing per the "Strict-flip atomicity rule".
- Wire-in: `tools/local_pre_deploy_check.py` 8th check `dispatch_optimization_protocol`; the operator-authorize 30s harness consults the protocol BEFORE paid dispatch via the existing `_run_local_pre_deploy_check` route. The harness rc=1 in strict mode if the protocol fails.
- Strict-flip plan: when the operator-routed Tier 1 backfill sweep drives violations to 0 (substrate trainers route through canonical scorer-loss helper + declare engineering primitives in argparse), flip the wire-in to `strict=True` in the same commit batch per "Strict-flip atomicity rule".

### Scope clarification: substrate trainer vs tool dispatch (2026-05-17)

**Source:** operator directive 2026-05-17 *"Fix all now first"* + lane
`lane_catalog_270_scope_fix_tool_vs_substrate_dispatch_20260517` per the
master-gradient extractor dispatch incident (paid Modal CPU dispatch refused
by Catalog #270 with 5 substrate-only blockers categorically inapplicable to
the `tools/extract_master_gradient.py` one-shot CPU extractor).

The Tier 2/3 fields above are SCOPED to substrate trainers
(`experiments/train_substrate_*.py`). Tool dispatches (`tools/*.py`) are
categorically NOT subject to the substrate-only primitives. The runtime
dispatch protocol at `src/tac/deploy/dispatch_protocol.py` (consumed by
`tools/operator_authorize.py::_native_dispatch_preflight`) and the canonical
helper at `tools/canonical_dispatch_optimization_protocol.py` (consumed by
`tools/local_pre_deploy_check.py`) both apply the following scope rule:

**Substrate-only fields skipped for tool dispatches:**
- Catalog #172 `--enable-autocast-fp16` (substrate training primitive; tool
  one-shot inference does not benefit and may be CPU-only).
- Catalog #178 TF32 (CUDA matmul-only; inapplicable to CPU tool dispatches).
- Catalog #179 `--enable-torch-compile` (substrate training primitive; one-shot
  tool inference does not benefit from compile overhead).
- Catalog #226 `gate_auth_eval_call` (substrate auth-eval routing; tools that
  are not contest_auth_eval invocations do not produce contest-CUDA score
  claims).
- Catalog #215 `min_smoke_gpu` GPU class enforcement (tool dispatches MAY
  declare `CPU` case-insensitive when the tool is CPU-only one-shot inference).

**Universally enforced for both substrate AND tool dispatches:**
- Tier 1 engineering (lane_id pattern / dispatch_enabled / cost_band /
  driver+trainer existence / native platform legality).
- Tier 2 hardware correctness EXCEPT min_smoke_gpu (min_vram_gb /
  video_input_strategy / pyav_decode_strategy / target_modes / canary_status
  / Modal NVML env block per Catalog #244).
- Tier 3 no_grad/inference_mode eval-time memory hygiene (Catalog #180).

**Detection surfaces** (either short-circuits to "this is a tool dispatch"):
1. **Explicit**: `dispatch_kind: tool` in the recipe frontmatter. Reserved
   values: `tool`, `substrate` (default).
2. **Implicit**: trainer_path matches `tools/*.py` AND not
   `experiments/train_substrate_*.py`.

Detection is implemented in `tac.deploy.dispatch_protocol.is_tool_dispatch`.
Tests at `src/tac/tests/test_dispatch_protocol_tool_scope.py` cover both
detection surfaces + every scope-fix invariant + a live-recipe regression
guard that loads the actual
`.omx/operator_authorize_recipes/master_gradient_fec6_modal_cpu_dispatch.yaml`
and verifies it passes the dispatch protocol post-fix.

**Anti-pattern this prevents:** band-aid waivers
(`# AUTOCAST_FP16_WAIVED:<reason>`, `# TF32_WAIVED:<reason>`, etc.) on tool
files claiming substrate-only primitives are categorically inapplicable. The
scope clarification makes the category distinction structural at the
protocol surface rather than per-file via copy-pasted waivers.

The protocol is the operational guard against the "missing optimization"
class. Future bug-class incidents that fit one of the three tiers MUST
extend the canonical helper's detection vocabulary AND the sister
strict-gate's signal map; the umbrella gate then refuses dispatch
structurally at the source-text + recipe-schema + lane-driver layers
before the paid GPU meter starts.

## `tac` stays clean; comma-lab owns research state

`tac` is the reusable Task-Aware Compression library and runtime-contract
surface. Put real reusable Python implementation there: codec primitives,
archive grammars, payload parsers, scorer/eval contracts, byte profilers,
planning primitives, visualization primitives, and contest-relevant algorithms.
Thin CLIs may live in `experiments/`, `scripts/`, or `tools/`, but they should
delegate to `tac` modules instead of burying implementation in ad hoc entry
points.

Do not add Claude/OMX/provider/recovery policy to `tac` unless it is truly
reusable codec, contest-runtime, or contest-preflight logic. Checks that protect
archive validity, inflate/runtime compliance, CUDA-score custody, and package
safety are canonical in `src/tac/preflight.py`. Put research-state custody,
public-frontier intake, hosted supplement builds, provider ledgers, and recovery
audits in `src/comma_lab/`, `tools/`, `docs/`, and `.omx/`.

Use `reverse_engineering/` for clean public-submission deconstruction: curated
runbooks, bit-level anatomy notes, adapter boundaries, and small manifests.
Keep raw PR clones, archives, provider transcripts, and large generated
artifacts in ignored custody locations with ledger links. Reusable parsers and
planners still belong in `tac`.

Track small durable `.omx/research` ledgers and small structured summaries.
Do not track raw `.omx/state/*.json`, provider transcripts, auto-memory
snapshots, generated public-site bundles, `reports/raw`, `reports/private`, or
large rebuildable artifacts. Canonicalize interesting ignored state into dated
`.omx/research` ledgers or `docs/paper/ara`, and host large canonical artifacts
externally with a committed manifest.

Use `python tools/audit_research_state_tracking.py --repo-root .` before
release or cleanup. Its implementation lives in `src/comma_lab/research_state.py`
on purpose. `src/comma_lab/preflight/strict_checks.py` is only an adapter/catalog
surface for ARA, reports, hosted supplements, and dashboards.

## Public frontier watch and intake — NON-NEGOTIABLE

During active contest or replay windows, refresh public PRs and official
leaderboard state frequently enough that late submissions are not missed while
internal lanes run. For any public target that can beat the local exact
frontier, immediately collect PR number, title, author, URL, head SHA,
created/updated time, archive URL, bytes, SHA-256, member names, source
runtime, dependencies, claimed components, recomputed public score, compliance
risks, and fastest exact-replay path. Use detached clones or artifact
directories; do not checkout public PRs into the dirty shared worktree.

If a lower public claim appears, the default order is:

1. Download archive and source.
2. Build bit-level anatomy and compliance-risk record.
3. Claim replay lane.
4. Queue exact CUDA eval on T4/equivalent or fastest available faithful path.
5. Harvest JSON, adjudicate, then build/update submission packet.

Council review cannot block steps 1-4 for a public lower-score replay unless
there is a specific contest-compliance violation that would make the replay
invalid.

## Bit-level deconstruction and entropy discipline

For archive/packer work, inspect bytes before arguing from prose. Record ZIP
header parity, member order, compression method, sizes, CRCs, duplicate names,
magic, section offsets, length prefixes, section hashes, entropy estimates,
decoded tensor shapes, side channels, and no-op/provenance detection.

Arithmetic coding, range coding, ANS/Huffman-style coders, brotli/zstd/lzma
transforms, tensor grouping, histogram overhead, fixed-section removal, and
deterministic pack ordering are first-class score lanes. If a dense stream
remains in a generic compressor, estimate entropy and test a real coded payload
before declaring the area saturated.

## FORBIDDEN PATTERNS — NON-NEGOTIABLE, READ BEFORE WRITING ANY CODE

These are exact code patterns I have written multiple times despite the rules below saying not to. They are FORBIDDEN at the typing moment. If a default would land here, refuse it before typing — do not "fix it on review."

**Forbidden device-selection defaults (the MPS-fallback trap):**
```python
device = "cuda" if torch.cuda.is_available() else "mps" if ... else "cpu"  # FORBIDDEN
device = torch.device(env.get("DEVICE", "cuda" if cuda.is_available() else "mps"))  # FORBIDDEN
```
Correct: default to CUDA-REQUIRED. Raise on no-CUDA. Provide explicit `--device cpu` opt-in with a banner that the bytes/score will differ. (See `feedback_default_to_convenience_trap`.)

**Forbidden CLI flag inventions (the dead-flag trap):**
Adding `--auth-eval-masks` to a subprocess call without `grep "add_argument" target.py` first. Inventing flag names from intent is FORBIDDEN. Always grep the target's argparse before emitting any flag. (See `feedback_dead_flag_wiring_pattern`.)

**Forbidden silent-skip cascades (the bootstrap trap):**
Writing `set -uo pipefail` (no `-e`). Calling `zip` shell binary instead of python `zipfile.ZipFile`. Passing empty captured variable to argparse. (See `feedback_zip_dep_bootstrap_trap`.)

**Forbidden score claims:**
Reporting any score that did not come from `upstream/evaluate.py` on the EXACT archive bytes that will be submitted. No proxy MSE. No MPS. No "looks reasonable" extrapolation. Tag every reported score by axis: `[contest-CUDA]` for CUDA promotion truth, `[contest-CPU]` for explicit public-leaderboard reproduction, or `[advisory only]` for everything else. (See `feedback_proxy_auth_math_useless`, `feedback_mps_cuda_drift_critical`.)

**Forbidden component-aliasing for baselines:**
Treating a directory of components as the "baseline" without verifying every file SHA against the archive ZIP that produced the baseline score. Components from different lanes leak into the same dir; SHA-vs-archive is the only check. (See `feedback_phantom_baseline_pattern`.)

**Enforcement:** Before typing ANY of the above patterns, STOP. The non-negotiable wins over the convenient default. This list is in CLAUDE.md so it is loaded into context at session start; if I write one of these patterns anyway, that is a process failure, not an information failure.

**Forbidden empirical-claim-without-evidence-tag (the docstring-overstatement trap):**
Writing "saves 49%" / "improves N%" / "beats baseline" / "verified" in a docstring/report/script without an adjacent `[empirical:<artifact path>]` or `[contest-CUDA]` or `[prediction]` tag. Lane PD docstring stated 49% savings; empirical regression test caught actual 18.5%. The 49% was a derivation, not a measurement. Tag every claim. (See `feedback_three_active_bug_classes_needing_strict_checks_20260429.md`.)

**Forbidden fix-lands-in-helper-but-not-callsite (the dangling-helper trap):**
Adding a kwarg to a helper without grepping for callers and updating each. Lane GP added `baseline_poses=` to `reconstruct_poses()` but the actual call at `experiments/fit_pose_gp.py:33` never passed it for ~2 weeks. After adding any kwarg with non-trivial semantics, register it in `CALLSITE_CONTRACTS` and run the AST scanner to enforce all callers pass it. (See `feedback_three_active_bug_classes_needing_strict_checks_20260429.md`.)

**Forbidden MPS-derived strategic decision (the MPS-falsification trap):**
Writing "GREEN" / "RED" / "KILL" / "promoted" / "FALSIFIED" in any record where the supporting evidence is an MPS or non-`contest-CPU` CPU forward pass through SegNet/PoseNet/renderer/distilled scorer. MPS PoseNet drift is 23×; SegNet 2×; score 2.5×. STC clean-source FALSIFICATION was made on MPS encoder; user correctly objected; withdrawn. Internal CUDA promotion/kill decisions REQUIRE a `[contest-CUDA]` artifact in the same record/section. Public-leaderboard CPU claims REQUIRE `[contest-CPU]` on exact archive/runtime custody, and still must record the missing paired CUDA/CPU axis instead of extrapolating it. (See `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`.)

**Forbidden misleading-directory-name (the phantom-score directory trap):**
Writing output to `*_cuda_*` / `*_cpu_*` / `*_mps_*` / `*_anchor/` / `*_full/` / `*_smoke/` directories or files whose name DOES NOT match the actual device/scope/contract that produced the contents. The directory name MUST match the metadata that generated it OR be device-agnostic. Empirical anchor: Z3 v2 FULL Modal A100 dispatch 2026-05-15T11:41:15Z wrote CPU eval results to `contest_auth_eval_cuda_work/` directory + `contest_auth_eval_cuda.json` file because the trainer hardcoded the `_cuda` suffix while the Modal dispatcher injected `AUTH_EVAL_DEVICE=cpu`. Parent agent quoted "0.19869 [contest-CUDA T4]" from the FILENAME despite metadata saying `device=cpu` / `score_axis=diagnostic_cpu`. Paired re-eval revealed true CUDA = 0.2317 — the originally-claimed CUDA score did not exist. Per CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden component-aliasing for baselines": the metadata is the truth; the directory must not lie. Caught by Catalog #249 `check_no_misleading_device_named_output_directories` (warn-only at landing; strict-flip pending) + runtime auto-redirect at `tac.substrates._shared.smoke_auth_eval_gate._redirect_output_json_to_match_device` (active immediately). Sister of Catalog #127 (custody validator) + Catalog #221 (auth_eval result fail-closed) + Catalog #226 (canonical helper). <!-- HISTORICAL_SCORE_LITERAL_OK:z3_v2_phantom_cuda_filename_anchor_2026-05-15_forbidden_pattern_section -->

**Forbidden /tmp paths in any persisted artifact (the transient-evidence trap):**
Writing `/tmp/<anything>` as a durable evidence path in: lane registry evidence strings, dispatch claims, commit messages, build metadata, score/rebuild manifests, runbooks, or any artifact that another agent may use to reproduce a result. /tmp paths do NOT survive a fresh checkout, do NOT exist on remote/CI/cloud machines, and CANNOT be verified by other agents. They produce phantom "evidence" that points at nothing. User mandate 2026-05-05: "we need to stop using /tmp by principle". Forensic finding: `lane_pr106_stacked` was marked L2 with `real_archive_empirical:true` evidence pointing at `/tmp/pr106_stacked_smoke/stacked_full/pr106_stacked_archive.zip` — a path that doesn't exist on any other machine and would be lost on shell exit. **Canonical replacement**: `experiments/results/<lane_id>_<timestamp>/` for build artifacts; `.omx/state/` for ledgers; `.omx/research/` for durable analyses; `.omx/tmp/` for explicitly ephemeral local scratch. Historical transcripts may mention `/tmp` only as scratch-only, non-evidence context; they must not be cited as reconstructable custody. Caught by `tools/check_lane_smoke_signal_nontrivial.py` (PCC9, transient_tmp_evidence detection).

**Forbidden force-canonical-without-evaluation-of-suppression (the canonicalization-trap):**
Default-adopting a canonical helper / META layer field / engineering pattern in a NEW substrate scaffold WITHOUT explicitly evaluating whether the canonical would suppress this substrate's optimal score. Per the operator's 2026-05-15 retrospective ("this has been a huge problem since the beginning of the competition"): the canonicalization reflex was the structural cause of the 0.196-0.199 cluster — every substrate inherited the same 18 shared assumptions, producing a flat plateau where each "new substrate" was structurally a variation of the SAME implementation under different names. From now: every canonical helper / META layer field / engineering pattern adoption per substrate scaffold MUST be paired with the design memo's `## Canonical-vs-unique decision per layer` section per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable. Default to FORK when there's principled mismatch; default to canonical only when it serves measurably or by clear principle. The bolt-on vs substrate-engineering split per HNeRV parity discipline lesson 7 is the canonical articulation: bolt-ons share, substrate engineering unique-ifies. Caught by Catalog #290 `check_substrate_design_memo_has_canonical_vs_unique_decision_section` (warn-only initially; strict-flip pending). Memory: `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` + `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` + `feedback_knowledge_preservation_pr95_meta_level_lesson_landed_20260515.md`.

**Forbidden artifact-lifecycle violations (the provenance-vs-state confusion meta-class — codex 2026-05-08, Catalog #113):**
Five surface findings (operator-approval-leak, public-PR-clone-dirty, status.json-stale-dirty, rebuild_command-baked-timestamps, recovery_metadata-mutated-in-place) all share ONE structural class: transient/global/upstream state being frozen or mutated into committed/forensic artifacts. Four-kind taxonomy enforced via `src/tac/artifact_lifecycle.py` + `.omx/state/artifact_kind_registry.yaml` + `check_artifact_lifecycle_compliance` umbrella gate. Specifically forbidden: (1) **committing transient state into LIVE_STATE files** — files matching `LIVE_STATE` patterns must be gitignored (e.g., locks, fs caches, vastai_active_instances.json); (2) **mutating HISTORICAL_PROVENANCE files** — `recovery_metadata.json`, `lane_maturity_audit.log`, dispatch claims, contest auth eval JSONs are append-only; field mutation outside registered `append_fields` is FORBIDDEN; (3) **baking transient values into LIVE_RECIPE files** — no hardcoded `--now-utc <ISO>`, `--operator-approved-*`, durable `/tmp/...` evidence paths, or hardcoded Vast.ai instance IDs in `rebuild_command.txt`/`scripts/*.sh`/`tools/*.sh`/`inflate.sh`; use `${PARAMETER}` placeholders OR add explicit `HISTORICAL_RECIPE_ONLY` header; (4) **stale session state in DERIVED_OUTPUT bodies without regeneration header** — `status.json`/`reports/latest.md`/dashboards must declare `generated_at: <utc>` + `from_state_hash: <sha>` within first 4 KB so consumers know it was regenerated, not snapshotted. Per CLAUDE.md "Operator gates must be wired and used" — `check_artifact_lifecycle_compliance` is wired into `preflight_all(strict=True)`. Reactivation criteria: every long-lived artifact in the repo explicitly classified in registry. Memory: `feedback_codex_findings_meta_pattern_artifact_lifecycle_FIXED_20260508.md`.

**Forbidden premature KILL without research exhaustion (the kill-too-fast trap):**
Writing "KILL" / "FALSIFIED" / "DEAD" / "RETIRED" as a final verdict on a lane based on a SINGLE empirical configuration's failure, when plausible alternative configurations have NOT been attempted. apogee_int4 NAIVE-PTQ falsification 2026-05-05 was initially recorded as KILL/FALSIFIED at score 1.4287 [contest-CUDA T4] WITHOUT trying QAT, LSQ, per-channel scaling, smaller block sizes, or outlier handling -- all canonical fixes for low-bit PTQ collapse. User caught it: "we must only kill as a last resort after exhausting all research and grand council consensus" (2026-05-05), reinforced 2026-05-08 as "always investigate all results that come back deeply and adversarially and rigorously and only falsify and kill as an absolutely last resort." Default verdict for one-config failure is **DEFERRED-pending-research** or **measured-config retired**, NOT KILLED. KILL conversion requires (a) every plausible alternative config attempted empirically, (b) exact custody/recomputation/failure classification for the returned result, (c) **grand council CONSENSUS** (not just majority — every inner-ten member endorses), (d) reactivation criteria documented. Memo filename uses `_DEFERRED_pending_<reason>_<date>.md`, not `_killed_*.md`. See expanded "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE" section above for full enforcement.

**Forbidden re-implementing remote bootstrap inline (the duplicated-bootstrap trap):**
Writing `curl -LsSf https://astral.sh/uv/install.sh | sh` or `apt-get install ffmpeg` or `find upstream -name '._*' -delete` directly in any new chain driver / lane script / one-off. There is ONE canonical bootstrap function: `bootstrap_runtime_deps()` in `scripts/remote_archive_only_eval.sh`, which delegates uv install to `scripts/ensure_remote_uv.sh`. Any new remote script MUST call that wrapper or `source` its bootstrap function — NEVER copy-paste the install commands. Cost of NOT doing this (2026-05-01 loop session): 6 sequential bug-class re-discoveries on 4 destroyed Vast.ai instances burning ~$1.50 + 30 min wall-clock chasing the same lesson. Memory: `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md`. (Sister rule for venv: `python -m ensurepip --upgrade` is the standard fix for "venv exists but no pip" — see `scripts/ensure_remote_uv.sh` style.)

**Forbidden uv torch install without driver-version pin (the cu13-vs-cu124 trap):**
`uv run --with torch==2.5.1` (no local-version suffix) defaults to the LATEST CUDA wheel from PyPI (currently `+cu13`). On a Vast.ai host with NVIDIA driver < 580 (CUDA 12.x), the cu13 wheel will FAIL `torch.cuda.init()` and silently fall back to CPU — every score becomes `[advisory only]` per the MPS-falsification rule. The canonical pattern in `scripts/remote_archive_only_eval.sh:88-95` auto-pins:
- `driver_major < 580` (CUDA 12.x): `INFLATE_TORCH_SPEC=torch==2.5.1+cu124` + `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124` + `UV_INDEX_STRATEGY=unsafe-best-match`.
- `driver_major >= 580` (CUDA 13.x): `INFLATE_TORCH_SPEC=torch==2.11.0` (default cu13 wheel works).

NEVER write `--with torch` (unpinned) in any new script. Always export `INFLATE_TORCH_SPEC` first or call the canonical wrapper. Cost (2026-05-01): instance 35957332 silently ran inflate on CPU for 5 min before detection. Memory: `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md`.

**Forbidden Vast.ai create without disk + cuda_vers gate (the chain-killer trap):**
Calling `vastai create instance ... --disk 30` for any chain that runs >1 candidate. The contest_auth_eval pipeline writes 3.6GB of inflated raw frames per candidate; a 6-candidate chain needs ≥30GB working set (5GB uv torch cache + 4GB rolling). 30GB hits the wall on candidate 4. Canonical defaults: `--disk 60`, `cuda_vers>=12.4` in the search filter, `--label <unique>`, register to `.omx/state/vastai_active_instances.json`. The chain driver MUST `rm -rf eval_work/{inflated,extracted,archive.zip}` after each successful eval — the canonical pattern is in `experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/wave3_chain_driver.sh`. Memory: `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md` outstanding-gaps section.

**Forbidden in-place edits to public PR intake clones (the corrupt-source-provenance trap):**
Adding ANY waiver comment, fix, or annotation directly inside `experiments/results/public_pr*_intake_*/{source,repo,pr*_src/repo}/...py`. Even comment-only additions (e.g., `# KL_BATCHMEAN_OK:public-PR-intake-not-our-quality-debt`) corrupt source provenance — `git -C <clone> status` becomes dirty, replay/audit no longer matches the public PR head, and preflight cleanliness becomes dependent on local-only edits absent from upstream. Codex 2026-05-08 review HIGH finding. **Canonical replacement**: scanners that scan `experiments/` already exclude `_intake_` paths via `_VENDORED_PATH_MARKERS` (`src/tac/preflight.py:5113`). For the rare scanner that cannot honor a path-prefix exclusion, record waiver rationale in `reverse_engineering/public_pr_waiver_manifest.json` (committed metadata) — never inline. Caught STRICT by `check_public_pr_intake_clones_pristine` (Check 109). Memory: `feedback_codex_finding_2_public_intake_pristine_FIXED_20260508.md`.

**Forbidden timestamp-only mutation of recovery_metadata.json (the recovery-custody-evidence-corruption trap):**
Overwriting `started_at_utc` / `completed_at_utc` / `elapsed_seconds` on an existing `experiments/results/recovered_*/recovery_metadata.json` without adding new substantive evidence (artifacts, logs, archive, ssh-state changes). The file is a forensic audit record; in-place timestamp churn destroys the original recovery-attempt timeline so future audits can't distinguish the original failed attempt from a fresh probe. Codex 2026-05-08 finding caught April 30 → May 8 churn on `recovered_42_dead` + `recovered_99999_phantom` with sub-millisecond elapsed and zero substantive change. **Canonical replacement**: the `attempts[]` append-only schema (v2_attempts) — every probe appends a new attempt entry keyed by unique `started_at_utc`; closed attempts are immutable. The writer (`tools/recover_lane_artifacts.py::_write_report`) refuses same-`started_at_utc` mutation of closed attempts via `RecoveryMetadataAppendOnlyError`. Non-initial attempts MUST carry `command_log_path` OR `substantive_change_from_prior_attempt` provenance. Caught STRICT by `check_recovery_metadata_append_only` (Check 110). Memory: `feedback_codex_finding_5_recovery_metadata_appendonly_FIXED_20260508.md`.

**Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof (the rate-term-paid-zero-bytes-saved trap):**
Writing a substrate codec that allocates bits via a closed-form CDF (e.g., `gaussian_cdf(...)` / `laplace_cdf(...)` / `hyperprior_cdf(...)` / `closed_form_cdf(...)` / `_allocate_bits_from_cdf(...)` / `bits_from_cdf(...)`) without an adjacent empirical bit-spend proof comparing the closed-form prediction to the actual byte cost. Anchor cite: NSCS06 v6 falsification symposium cargo-cult #1 (rate=1.96 closed-form prediction with ZERO bytes materialized; the rate term was paid for an allocation strategy that the inflate path never realized as bytes saved). **Correct**: every closed-form CDF allocator MUST be paired with an empirical bit-spend verification token in the same file (`empirical_bit_spend` / `bit_spend_proof` / `measured_bit_cost` / `actual_byte_cost` / `verify_bit_allocation` / `bit_allocation_verified` / `verify_closed_form_vs_empirical`) OR carry a same-line `# CLOSED_FORM_CDF_ALLOCATOR_OK:<rationale>` waiver. Caught STRICT by Catalog #304 (`check_substrate_codec_no_closed_form_cdf_without_empirical_bit_spend_proof`). Sister of CLAUDE.md "Bit-level deconstruction and entropy discipline" + Catalog #105 / #139 / #220 / #272. Memory: `feedback_permanent_fix_self_protect_all_today_bug_classes_landed_20260516.md`.

**Forbidden spatial-independent-CDF-assumption (the per-pixel-independence trap):**
Designing a substrate codec that assumes per-pixel independence (e.g., a closed-form CDF allocator that treats each pixel as drawn IID from the parametric distribution) without testing against the spatial-correlation alternative. Anchor cite: NSCS06 v6 falsification symposium cargo-cult #2 — the spatial-independent CDF assumption was implicit in the closed-form allocator design but never tested against an autoregressive / hyperprior / context-model alternative; the cargo-cult was inheriting the IID assumption from JPEG/HEVC reference codecs without checking that NSCS06's specific compress-time signal axis preserved it. **Correct**: every substrate design memo proposing a CDF allocator MUST explicitly answer "is the per-pixel-independence assumption HARD-EARNED (from empirical measurement of the substrate's compress-time signal) or CARGO-CULTED (inherited from JPEG/HEVC defaults)?" per the hard-earned-vs-cargo-culted addendum. The HARD-EARNED case is rare; the CARGO-CULTED case requires an unwind-test against autoregressive / hyperprior / context-model alternatives. Sister of HNeRV parity discipline lesson 6 (score-domain Lagrangian not weight-domain proxies). Caught structurally by Catalog #303 (per-substrate cargo-cult audit section) + Catalog #292 (per-deliberation assumption surfacing). Memory: `feedback_permanent_fix_self_protect_all_today_bug_classes_landed_20260516.md`.

**Forbidden NO-neural-at-medal-band-assumption (the strip-everything-medal-class trap):**
Assuming an archive frontier at medal-class scores (≤ 0.20) requires NO neural component (i.e., that medal-class is achievable with purely deterministic transforms like grayscale-LUT / VLC / arithmetic coding). Anchor cite: NSCS06 v7 PARTIALLY disproved this — the medal-class composite #4 (Carmack-Hotz Strip-Everything) was designed without neural components but v7 (105.15 -> 58.89) showed per-class chroma anchors function as quasi-neural priors. The cargo-cult is assuming that medal-class is structurally NO-neural; the empirical evidence shows medal-class requires SOME context-aware structure (the question is which context-aware structure, not whether it's "neural" by name). **Correct**: every substrate design memo proposing a NO-neural medal-band target MUST explicitly answer "is the NO-neural assumption HARD-EARNED (from empirical measurement that contest-CUDA scores below 0.20 are achievable with zero learned parameters) or CARGO-CULTED (inherited from Carmack-Hotz aesthetic preferences for engineering simplicity)?" The empirical evidence on this is incomplete — the cargo-cult-unwind methodology applies. Sister of HNeRV parity discipline lesson 5 (full renderer not single-component slot) + Catalog #303 (per-substrate cargo-cult audit section). Memory: `feedback_permanent_fix_self_protect_all_today_bug_classes_landed_20260516.md`.

**Forbidden symposium-band-prediction-without-Dykstra-feasibility-check (the predicted-band-vibes trap):**
Writing a predicted ΔS band in a symposium / council / substrate design memo without an adjacent Dykstra-feasibility intersection check (or sister first-principles citation: Shannon / R(D) / MDL / Tishby / Daubechies / Mallat / Wyner / Atick-Redlich / Rao-Ballard) OR a probe-disambiguator path (`tools/probe_*_disambiguator.py`). Anchor cite: NSCS06 v6 dispatch landed 105.15 vs predicted [0.10, 0.20] band — 553x OUTSIDE band — because 5-move composition was assumed additive under contest rate+distortion polytope constraints WITHOUT a Dykstra-feasibility intersection check. D1 1.18x OUTSIDE band. C6 MDL-IBPS + Time-Traveler L5 carry similar cargo-cult-prediction risk. Per CLAUDE.md "Meta-Lagrangian/Pareto solver - NON-NEGOTIABLE": *"Prefer solvable math over arbitrary sweeps."* Per CLAUDE.md "Council conduct": Dykstra co-leads the inner quintet pact specifically because alternating-projections feasibility IS the arbiter of whether a multi-constraint composition is achievable rather than just predicted. **Correct**: every predicted ΔS band MUST cite the Dykstra-feasibility intersection check OR a first-principles bound OR a probe-disambiguator path that resolves the prediction empirically. Same-line waiver `# PREDICTED_BAND_VIBES_OK:<rationale>` on the section header for the rare hand-earned case. Caught STRICT by Catalog #296 (`check_substrate_predicted_band_has_dykstra_feasibility_check`; strict-flipped 2026-05-16). Sister of Catalog #229 (premise verification before edit) + Catalog #290 (canonical-vs-unique decision per layer). Memory: `feedback_permanent_fix_self_protect_all_today_bug_classes_landed_20260516.md`.

**Forbidden PR#56-pattern-generalizes-to-frames-without-per-substrate-empirical-validation (the PR56-pattern-cargo-cult trap):**
Assuming the PR#56 selfcomp pattern (compress single weight tensor via grayscale LUT + structured CDF) generalizes to full-frame substrate designs without per-substrate empirical validation. Anchor cite: NSCS06 v6 cargo-cult — the PR#56 pattern was assumed to extend from single-weight selfcomp to full-frame substrate compression without testing whether the contest scorer's PoseNet+SegNet response to frame-level grayscale-LUT preserves the score, OR whether the contest video's spatial structure matches the selfcomp pattern's IID assumption. Empirical receipts: NSCS06 v6 contest-CUDA = 105.15 (far below medal-class); the PR#56-pattern-generalization assumption was unwound in NSCS06 v7 by per-class chroma anchors (105.15 -> 58.89 in ONE iteration). **Correct**: every substrate design memo proposing to generalize PR#56 (or any other single-weight selfcomp pattern) to full-frame compression MUST include an empirical anchor measuring the contest scorer's response to the proposed full-frame transform BEFORE landing the trainer. Sister of HNeRV parity discipline lesson 5 (full renderer not single-component slot — the score depends on the FULL frame, not a single component) + Catalog #303 (per-substrate cargo-cult audit section). Caught structurally by Catalog #303 + Catalog #294 (9-dim checklist evidence section). Memory: `feedback_permanent_fix_self_protect_all_today_bug_classes_landed_20260516.md`.

**Forbidden substrate driver hardcoding smoke=1 / --smoke regardless of dispatch env vars (the driver-mode-mismatch trap):**
Writing `scripts/remote_lane_substrate_*.sh` drivers whose trainer invocation passes `--smoke` (or sets `SMOKE=1` / `smoke=1`) without consulting an env var like `${SUBSTRATE_TRAINER_MODE}` / `${SMOKE_ONLY}` / equivalent, OR consulting an env var whose DEFAULT biases to smoke without explicit recipe-side opt-in for full. **Empirical bug class anchor**: Z6-v2 Wave 2 full canary dispatch `fc-01KRW7ZCYK5XF6MSHD24R71A46` (2026-05-18) ran `_smoke_main` despite the recipe requesting `Z6_EPOCHS=100` full-mode because the Wave 2 recipe `env_overrides` block did NOT set `SMOKE_ONLY=0` and the driver's `SMOKE_ONLY="${SMOKE_ONLY:-1}"` default produced smoke-mode regardless of intent. Trainer entered `_smoke_main` with synthetic-cfg overriding the council-binding ~300K depth=3 spec — paid Modal $0.50 spend producing NO score evidence for the actual architectural distinguishing feature. Sister C6 IBPS DEFER + Z6-v2 DEFER showed TWO consecutive ASYMPTOTIC dispatches frustrated by infrastructure-level bugs (NOT paradigm falsification per Catalog #307). **Correct patterns**: (a) driver supports multi-key mode resolution (e.g. `Z6_TRAINER_MODE > SMOKE_ONLY > default`) with fail-loud warning when no key is set; (b) driver defaults to full-mode and requires explicit smoke opt-in; (c) recipe's `env_overrides` block ALWAYS sets the mode env var explicitly per Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS discipline; (d) same-line `# DRIVER_MODE_HARDCODE_OK:<rationale>` waiver on the `--smoke` line for drivers intentionally smoke-only by design with matching `research_only=true` / `smoke_only=true` / `dispatch_enabled=false` recipe. Caught STRUCTURALLY by Catalog #326 (`check_substrate_driver_consumes_trainer_mode_env_var`). Operator audit tool: `tools/audit_substrate_driver_mode_hardcode.py` (canonical helper + 7-verdict taxonomy: `NO_SMOKE_FLAG` / `HARDCODES_SMOKE_NO_RECIPE_OPT_OUT` (bug class) / `HARDCODES_SMOKE_RECIPE_OPTED_OUT` / `CONSUMES_ENV_DEFAULTS_SMOKE_BUG_CLASS` (bug class) / `CONSUMES_ENV_DEFAULTS_SMOKE_RECIPE_OK` / `CONSUMES_ENV_DEFAULTS_FULL` / `CONSUMES_ENV_MULTI_KEY_DEFAULT_RECIPE_OK` / `CONSUMES_ENV_UNKNOWN_DEFAULT` / `CONSUMES_ENV_NO_HARDCODE`). Sister forbidden patterns: "Forbidden re-implementing remote bootstrap inline (the duplicated-bootstrap trap)" (driver source discipline) + "Forbidden in-place edits to public PR intake clones" (source provenance discipline). Sister catalog gates: Catalog #270 (canonical dispatch optimization protocol — substrate-trainer-engineering surface) + Catalog #240 (recipe-vs-trainer-state consistency) + Catalog #244 (canonical NVML env block in drivers) + Catalog #151 (operator wrapper Tier-1 flag threading) + Catalog #152 (required input file validation). Memory: `feedback_driver_fix_smoke_hardcode_plus_new_catalog_gate_cross_substrate_audit_landed_20260518.md`.

**Forbidden predicted_band-from-random-init-Tier-C-density (the phantom-predicted-band trap):**
Emitting a `predicted_band` field in any substrate operator-authorize recipe (`.omx/operator_authorize_recipes/*.yaml`) derived from a Tier-C density measurement that was computed on RANDOM-INIT weights (pre-training architecture) instead of a POST-TRAINING archive (≥1 epoch trained), WITHOUT one of: (a) `predicted_band_validation_status: validated_post_training` + post-training Tier-C density artifact path; (b) `predicted_band_validation_status: pending_post_training` + reactivation criteria pinned; (c) `research_only: true` OR `dispatch_enabled: false` (explicit non-promotable per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable); (d) same-line `# PREDICTED_BAND_RANDOM_INIT_OK:<rationale>` waiver with substantive rationale (placeholder `<rationale>` / `<reason>` literals rejected). **Empirical bug class anchor**: 2026-05-17 C6 IBPS recipe declared `predicted_band: [0.113, 0.163]` derived from Tier-C ACROSS_CLASS density `2.67e-5` measured BEFORE training; actual 50ep Modal A10G smoke (call_id `fc-01KRW353MJJ9A6QW8H99QWZEMH`) landed `final_score = 3.04` — **22× outside** predicted upper bound. Sister #835 recipe-fix sextet Assumption-Adversary verbatim WARNED this was CARGO-CULTED ("Tier-C computed on RANDOM INIT archive (pre-training). Post-training Tier-C may differ" + "$0.76 sufficient is only UPPER bound on disconfirmation"); sister #836 EMPIRICALLY FALSIFIED. Mechanism: 24-dim IB bottleneck destroys segmentation (score_seg=2.60 dominates 86% of total) — the pre-training Tier-C density never measured this because the architecture hadn't been trained yet. **Canonical helper** at `src/tac/optimization/tier_c_density_post_training_validator.py` provides `TierCDensityWithProvenance` + `PredictedBandWithValidation` frozen dataclasses; auto-derives `phantom_random_init` validation status when the underlying density's source is `RANDOM_INIT_PRE_TRAINING`. Caught STRUCTURALLY by Catalog #324 (`check_no_predicted_band_without_post_training_tier_c_validation`). Operator audit tool: `tools/audit_predicted_band_provenance.py`. Sister forbidden patterns: "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" (design-memo surface) + "Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof" (codec surface). Sister catalog gates: #321/#322/#323 phantom-score family + #303 cargo-cult audit + #296 design-memo Dykstra-feasibility. Memory: `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`.

## NEVER invent CLI flags — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Before wiring any flag into `subprocess.run([...])`, READ the target tool's actual `parser.add_argument(...)` list.** Don't invent flag names from intent. Don't trust prior code that "looked like it worked." Verify against argparse. The cost: 30 seconds of `grep "add_argument" target.py`. The cost of NOT doing it: days of wasted GPU + a council review chain that misses the dead-flag bug across multiple rounds.

This rule exists because (2026-04-26 incident, see `feedback_dead_flag_wiring_pattern`):
- R1 wiring of `train_renderer.py --auth-eval-on-best` invented `--auth-eval-masks` for `auth_eval_renderer.py` which has NO such flag.
- R2 "fix" didn't catch the dead flag — focused on rate ambiguity.
- R3 finally caught it (Council R3-1).
- Every chain that "passed" auth-eval-on-best was actually silently skipping it.

**How to apply:**
1. Before adding a flag to a subprocess invocation, `grep "add_argument" path/to/target.py` and confirm every flag name you're emitting exists.
2. Add a regression test that introspects the target's argparse and asserts your call-site flag set is a subset (template: `test_train_renderer_auth_eval_wiring.py`).
3. Fail loud (raise / non-zero exit), not silent (WARN-and-skip), when required inputs to a subprocess wrapper are missing.
4. **It is unacceptable to learn the same lesson twice.** Capture the meta-pattern in memory + CLAUDE.md the FIRST time it bites.

## Modal `.spawn()` HARVEST OR LOSE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Modal `.spawn()` puts artifacts in the FunctionCall return-value cache (~24h TTL), NOT in a Volume.** `experiments/modal_train_lane.py` uses `.spawn()` exclusively. The local-side dispatcher writes `experiments/results/lane_<label>_modal/modal_metadata.json` (with `call_id`) and exits — it does NOT poll for the result. NOTHING is written to a Modal Volume by this path.

**The investigation trap I fell into 2026-04-29 PM**:
- `modal app list` only shows currently-active apps. Terminated ephemeral apps disappear quickly.
- `modal app logs <app>` shows the most recent log buffer; earlier successful runs aged out.
- `modal volume ls` shows nothing because spawn() doesn't write volumes.
- I wrongly concluded "$0 wasted, all dead" when in fact the dashboard showed $38.80 spent on `modal_train_lane.run_lane_training_t4/a10g` and 31 of 37 dispatched call_ids had artifacts sitting in the result cache about to GC.

**The truth source is per-call**:
```bash
.venv/bin/python -c "import modal; r = modal.functions.FunctionCall.from_id('fc-...').get(timeout=2); print(r.get('returncode'), r.get('elapsed_seconds'), len(r.get('artifacts', {})))"
```
Browser dashboard: https://modal.com/usage (the only source-of-truth for actual billing).

**Rules**:
1. Every dispatch via `modal_train_lane.py` MUST be followed by a scheduled harvest within 24h. Reference: `tools/harvest_modal_calls.py` (formerly `/tmp/harvest_modal_calls.py`) iterates every `experiments/results/lane_*_modal/modal_metadata.json` and writes `harvested_artifacts/` next to each.
2. **NEVER** claim "Modal apps are dead, no artifacts" without first running the harvester. The harvester is the source of truth, not `modal app list`.
3. **A10G has 22GB shared**; SC++/SA-class lanes that allocate 21+GB will OOM (today's incident: lane_sc_plus_plus_v3 crashed at 140s with `CUDA out of memory. Tried to allocate 21.09 GiB`). For OOM-prone lanes, use Vast.ai 4090 (24GB dedicated, $0.26/hr) instead.
4. **Modal scheduling can take HOURS** for T4 / A10G during shortages. "waiting to be scheduled" means $0 charged in the wait, but moment-of-schedule starts the meter. Cancel queued functions you no longer want.
5. Future improvement (issue tracker): change `modal_train_lane.py` to also write artifacts to a Modal Volume so they persist past the result-cache TTL. The `.spawn()` pattern was added for "detached" runs, but the price is orphaned artifacts. Detached + persistent storage requires a Volume, not the result cache.

Memory: `feedback_modal_spawn_result_cache_pattern_20260429.md` documents the full incident + harvest pattern.

## Auth eval EVERYWHERE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY chained experiment MUST end with a CUDA auth eval against its best checkpoint.** Tracking only proxy `fp4_scorer` / `pose_loss` / training-loss is a WASTED run unless an authoritative score lands at the end. The proxy-auth gap can be 100-350x even on CUDA-CUDA (LANE-B 2026-04-26: proxy 0.0007 → auth 0.246, 350x). The proxy is a TRAINING SIGNAL, not a measurement.

This applies to:
- `experiments/pipeline.py compress` (HAS step_eval at end ✓)
- `scripts/remote_train_bootstrap.sh` (HAS Stage 5 auth eval ✓)
- `scripts/remote_pose_tto_bootstrap.sh` (HAS Stage 4 auth eval ✓)
- `scripts/remote_pose_tto_only_bootstrap.sh` (HAS Stage 4 auth eval ✓ as of 2026-04-26)
- `src/tac/experiments/train_renderer.py` — **GAP: NO auth eval on best.** Must be added: when a `*BEST*` checkpoint is saved, run a background CUDA auth eval and log the result alongside the proxy.
- ANY new training script, TTO loop, postfilter, or experiment runner.

**Pre-launch checklist (mandatory):**
1. Does the experiment end with `auth_eval_renderer.py` on the best checkpoint?
2. Is the auth eval result captured (RESULT_JSON or .json file) and surfaced to the operator?
3. If a chain has multiple "best" candidates (e.g., proxy-best, kl-best, hinge-best), does each get an auth eval?

**Pose TTO specifically:** the TTO loop MUST run a smoke auth eval at step 100 (and every 200 steps after) so the proxy-auth gap is detected within $0.50 of GPU spend, not at $5+ end-of-run.

**The authoritative measurement loop is:** contest `inflate.sh` → `upstream/evaluate.py` on the EXACT archive bytes. Nothing else counts. Memory: `feedback_proxy_auth_math_useless`.

## Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE — NON-NEGOTIABLE

**Every submission archive (anything that will be PR'd to the contest, or that we use to claim "medal-band score" / "frontier") MUST get authoritative auth eval scores on BOTH `--device cuda` AND `--device cpu`, AND BOTH must run on hardware that is 1:1 contest-compliant with the contest's GitHub Actions CI runner.** The contest leaderboard ranks by the CPU eval, not the CUDA eval. Verified 2026-05-08:

- PR102 (third prize) public CUDA comment: 0.22839 — matches our T4 replay within 3×10⁻⁶
- PR102 (third prize) public CPU comment: **0.19538** — this is the medal-band score the prize was awarded against <!-- HISTORICAL_SCORE_LITERAL_OK:pr102_third_prize_silver_medal_canonical_public_reference_2026-05-08 -->
- Δ CUDA−CPU: +0.033 on PR102

**Our PR #107 (apogee submission) was scored publicly only on CUDA at 0.22936; the maintainer never triggered a CPU eval comment.** Lab replay closed that blind spot with a GHA Linux x86_64 `[contest-CPU]` score of `0.1966358879` on exact archive/runtime custody. This confirms the submission was near the public medal cluster on the CPU axis, and it proves future shippable archives must be evaluated on both axes.

**1:1 hardware-compliance rule (NON-NEGOTIABLE):**

- Local macOS (M-series ARM, Intel iMac, anywhere on Apple Silicon or otherwise) is NEVER a 1:1 axis for CPU auth eval. It is allowed as a high-throughput advisory/dev-loop signal because PR107 M5 Max `0.19664189` matched GHA Linux x86_64 `0.1966358879` within `6e-6`, but it must be tagged `[macOS-CPU advisory only]` until confirmed on Linux x86_64.
- Required CPU substrate: **Linux x86_64** (Ubuntu LTS, matching the contest's GitHub Actions `ubuntu-latest` runner family; AMD EPYC or Intel Xeon class). The contest CI runs on x86_64 Linux; our CPU eval must too.
- Required CUDA substrate: **NVIDIA T4 / A100 / 4090 / equivalent** (matching the contest's CUDA runner; T4 is the contest's reference for the bot's CUDA comments).
- Both eval paths must use IDENTICAL upstream `evaluate.py` SHA, IDENTICAL `public_test_video_names.txt`, IDENTICAL video payloads, IDENTICAL `inflate.sh` runtime tree, IDENTICAL archive bytes.

**Rules:**

1. **Dual-eval is mandatory for any submission packet.** Before declaring a candidate "ready to PR" or "frontier-anchored," produce BOTH a `[contest-CUDA]` artifact AND a `[contest-CPU]` artifact on the EXACT same archive bytes via `upstream/evaluate.py` on `--device cuda` AND `--device cpu` respectively.

2. **Both tags are authoritative for their axis IF AND ONLY IF the hardware is 1:1 contest-compliant.** `[contest-CUDA]` requires NVIDIA GPU on Linux. `[contest-CPU]` requires x86_64 Linux. Apple Silicon CPU eval is `[macOS-CPU advisory]` (NOT `[contest-CPU]`) and is non-promotable.

3. **The CUDA−CPU gap is empirical and per-archive.** Do NOT assume PR102's −0.033 gap generalizes to our archives without measurement. Pose component appears to be the dominant gap source (5× difference on PR102 pose between CUDA and CPU), but mechanism attribution remains open: DALI/NVDEC-vs-PyAV ground-truth decode, CPU/CUDA forward-kernel drift, and pose-head numerics must be separated by the 2x2 decoder/network diagnostic before we treat any explanation as fact. Earlier FastViT attention/TF32 compounding explanations are invalid for FastViT-T12 on T4.

4. **CPU eval execution (where to run):**
   - **Vast.ai CPU instance** (Linux x86_64; cheap; matches contest CI architecture)
   - **Modal CPU container** (Linux x86_64; ~$0.06/hr; matches contest CI architecture)
   - **Lightning CPU Studio** (Linux x86_64; matches contest CI architecture)
   - **GitHub Actions CI workflow** itself (the actual contest hardware)
   - **NOT** local M5 Max / Apple Silicon / any macOS as the authoritative axis. Use macOS CPU for free parallel sweeps, curve discovery, smoke, and dev-loop ranking only; tag it `[macOS-CPU advisory only]` and promote to `[contest-CPU]` only after Linux x86_64 replay.

5. **CPU eval discipline (regardless of where it runs):**
   - Use `--device cpu` on `upstream/evaluate.py` directly. Verify `torch.cuda.is_available() == False`.
   - Force NO MPS path (which doesn't exist on Linux x86_64 anyway, but be explicit).
   - Tag results `[contest-CPU]` distinctly ONLY when running on Linux x86_64. Apple Silicon CPU eval is `[macOS-CPU advisory only]`.
   - CPU eval on a small Vast.ai / Modal CPU instance takes 60-120 min for 600 samples (matching the contest GitHub Actions CPU runner). Budget accordingly.

### MLX portable-local-substrate authority — NON-NEGOTIABLE

MLX is a local substrate for fast candidate generation, scorer-response
training data, portability engineering, and calibrated spend triage. It is not
a contest scoring axis.

- Tag MLX rows `[macOS-MLX research-signal]`.
- Every MLX-derived row must carry explicit false authority:
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`, and
  `promotable=false`.
- MLX scorer-response rows used by the LL planner must flow through
  `tac.optimization.scorer_response_dataset`.
- MLX spend triage requires both PyTorch/MLX parity evidence and an attached
  `tac.local_acceleration.mlx_score_calibration` manifest. Decisions below the
  calibration band are uncertain and must not trigger spend.
- The auth-side calibration/comparison payload must pass the strict
  `tac.auth_eval_schema.required_contest_auth_axis_payload_blockers` contract:
  only `contest-CPU` / `contest_cpu` and `contest-CUDA` / `contest_cuda`
  full-sample auth-axis payloads qualify. Advisory, diagnostic, proxy,
  macOS-local, forged-label, or partial-sample payloads fail closed even when
  numeric score components or hashes match.
- MLX may select local follow-up candidates and queue exact-eval candidates
  only through the normal lane-claim/custody path. Exact CPU/CUDA auth eval on
  contest-compliant hardware is still required before any score, frontier,
  promotion, rank/kill, or submission claim.

6. **For non-submission empirical work (intermediate candidates, ablations, sweep arms), use the cheapest faithful signal that matches the question.** CUDA is still the GPU-axis truth; macOS CPU and MPS can accelerate research-signal sweeps when tagged non-authoritatively. The dual-eval mandate applies specifically to ARCHIVES THAT WILL SHIP or are used to make medal-band/frontier claims.

7. **Existing CUDA-only artifacts are NOT retroactively invalidated.** They remain `[contest-CUDA]` with their CUDA-axis truth value. The dual-eval mandate is forward-looking: from this rule's commit forward, every shippable archive gets both axes on 1:1 contest-compliant hardware.

8. **Lane Maturity registry must reflect both axes.** A lane reaching Level 2/3 with a `[contest-CUDA]` anchor but no `[contest-CPU]` anchor (on Linux x86_64) is incomplete for medal-band ranking purposes — record both or record the missing one as a known gap.

Tooling:
- `tools/plan_dual_device_auth_eval.py` emits paired CPU/CUDA commands for the exact same archive/runtime.
- `tools/plan_public_pr_cpu_auth_eval.py` plans or runs a public-PR CPU replay from the reproduction ledger.
- `tools/public_pr_eval_comment_scorecard.py` extracts host PR-comment eval rows and recomputes scores from rounded PoseNet/SegNet/bytes.
- `experiments/contest_auth_eval.py` stamps CPU full-sample results as `evidence_grade="contest-CPU"` with `promotion_eligible=false`, `score_claim_valid=false`, and `rank_or_kill_eligible=false`.

Memory: `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md` (this rule's source memo). Cross-ref `feedback_cuda_cpu_auth_eval_drift_pr102_pr104_20260508` (codex's drift hypothesis matrix that established the empirical basis).

## eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY training path MUST use eval_roundtrip.** There are ZERO exceptions. This includes:
- train_distill.py (has it)
- training.py Trainer (NOW has it, eval_roundtrip=True by default)
- constrained_gen.py (has it)
- optimize_poses.py (has it)
- qat_finetune.py (has it)
- ANY new training script or optimization

Without eval_roundtrip, proxy-auth gap is 2-6x on PoseNet. Every training run without it is a WASTED run. This mistake has been made on EVERY component in this project. It stops now.

**NeRV/HNeRV renderer trainers must also keep scorer preprocess differentiable.** PR #95/#106 proved that eval-roundtrip belongs inside the training inner loop and that upstream `rgb_to_yuv6` severs PoseNet gradients because it is `@torch.no_grad()` / in-place. Any trainer that backprops through PoseNet/SegNet must either call `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` before scorer loss and patch/load differentiable YUV6 (`patch_upstream_yuv6_globally`, `load_differentiable_scorers`, or explicit `differentiable_rgb_to_yuv6`) before scorer construction, or carry a research-only ablation waiver with no score claim. Canonical implementation: `src/tac/differentiable_eval_roundtrip.py`; design memo: `.omx/research/CLAUDE_md_addition_eval_roundtrip_inner_loop_yuv6_20260509.md`.

## EMA — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY training path MUST instantiate EMA, update it after every `optimizer.step()`, and save the EMA shadow (not the live weights) as the inference checkpoint.** There are ZERO exceptions for any path that produces a checkpoint that ships in the submission archive.

This includes:
- Renderer training (`train_renderer.py`, `train_renderer_fridrich.py`, `train_distill.py`) — already correct
- SegMap training (`train_segmap.py`, `train_segmap_film_canvas.py`) — already correct
- Joint pair training (`train_joint_pair.py`) — fixed 2026-04-29 PM (duplicate `class EMA` removed; default 0.9995 → 0.997)
- Szabolcs / Selfcomp clones (`train_szabolcs.py`) — wired 2026-04-29 PM (Council D)
- QAT (`qat_finetune.py`, `qat_omega_lagrangian.py`, `quantize_distilled.py`) — wired 2026-04-29 PM (Council D)
- IMP cycles (`train_imp_cycle.py`) — wired 2026-04-29 PM (Council D)
- LoRA TTO (`train_lora_tto.py`) — wired 2026-04-29 PM (Council D)
- Postfilter training (`train_postfilter_on_renderer.py`) — wired 2026-04-29 PM (Council D)
- Codebook EMA in VQ-VAE / LCT mechanisms (van den Oord persistent N_c/m_c form) — already correct
- ANY new training script or optimization

**Quantizr decay = 0.997.** All weight EMAs default to `decay=0.997`. The CANONICAL `class EMA` is `tac.training.EMA` (with the float-buffer guard at L359-364 and the late-bound module guard at L356-358). Codebook EMAs (van den Oord persistent buffer form, e.g. `LearnableClassTargets`, `vqvae_codec`) keep their own 0.99 default — codebooks adapt faster than weights by design.

**Apply only at eval time, with snapshot+restore.** The canonical pattern (copied from `experiments/train_distill.py`):

```python
orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
ema.apply(model)
try:
    score = evaluate(model, ...)
finally:
    model.load_state_dict(orig_state)
    model.train()
```

**NEVER call `ema.apply(model)` inside `train_epoch`** without snapshot+restore — that shadows the live weights to the EMA snapshot and freezes learning (the DARTS-S freeze symptom class, even though that specific freeze was a different bug — bare `.round()` zero-gradient at `src/tac/segmap_renderer.py:281`; see Council D audit §6).

**Inference / archive bytes come from `ema.state_dict()`** — never from `model.state_dict()` after training. The Quantizr 0.33 archive is the EMA shadow, not the live final-epoch weights. Selfcomp's 0.38 archive is the EMA shadow. Lane G v3 (1.05) used EMA correctly.

**Without EMA, single-epoch noise dominates the final checkpoint.** Every training run without EMA is a wasted run. This stops now.

Cross-references: Council D audit `.omx/research/council_ema_audit_20260429.md`; preflight Check 88 `check_training_paths_use_ema_correctly` (STRICT @ 0 violations); Lane G v3 reference `project_lane_g_v3_landed_1_05_20260428.md`; Lane MM-V2 falsification `project_lane_mm_v2_landed_2_63_falsified_20260429.md` (which the same audit's §6 freeze investigation should be re-checked against once the V3-clean retrain lands).

## MPS auth eval is NOISE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**LOCAL MPS IS NEVER TO BE USED AS SCORE TRUTH OR AS THE AUTHORITY FOR STRATEGY, PLANNING, OR ANALYSIS.** Verified 2026-04-25 with side-by-side gating measurement on the same pinned archive:

| Metric | Local MPS | CUDA A100 (contest scorer) | Drift |
|---|---|---|---|
| PoseNet distortion | 0.245 | **0.0107** | **23x WORSE on MPS** |
| SegNet distortion | 0.0024 | 0.00116 | 2x WORSE on MPS |
| **Final score** | **2.26** | **0.90** | **2.5x WORSE on MPS** |

PoseNet specifically drifts 23× on MPS. Do not attribute this to FastViT-T12 attention: FastViT-T12 is RepMixer/convolutional on the contest path. Treat MPS drift as an empirical hardware/runtime mismatch until a layerwise diagnostic proves the mechanism.

**Rules:**
1. ALL **intermediate / non-submission** auth eval must run on CUDA (Vast.ai 4090, A100, T4). Never MPS as the authoritative axis. **EXCEPTION: submission packets (PR'd archives or "frontier-claimed" archives) require BOTH `[contest-CUDA]` AND `[contest-CPU]` per the new "Submission auth eval — BOTH CPU AND CUDA" section above.** The "never CPU" prohibition in the original rule was about local MPS-style CPU forward passes through SegNet/PoseNet scorers (which are noise like MPS), NOT about the contest's `upstream/evaluate.py --device cpu` path (which is the contest leaderboard's official scorer for ranking). The two are different: local-CPU-scorer-noise is invalid; contest-CPU-evaluator is authoritative.
2. MPS is acceptable for proxy scoring during training (continuous monitoring), smoke tests (architecture validation), code-correctness checks, and long cheap research-signal sweeps that only generate curve-shape priors. NEVER for strategy decisions, ranking, shipping, method retirement, or paper empirical claims.
3. Score numbers measured on MPS may NOT be reported as "auth" or "contest-compliant" anywhere — in commits, run_log, BATTLE_PLAN, or summaries. Tag them `[MPS-PROXY]` and treat as advisory only.
4. Before any major internal CUDA-axis decision (kill/promote) the score MUST come from a CUDA `inflate.sh` + `upstream/evaluate.py` run on the EXACT archive bytes. Before any ship/frontier/medal-band decision, the same archive must also have a `[contest-CPU]` Linux x86_64 result, and neither axis may be inferred from the other.
5. preflight should reject auth eval invocations with `--device mps` and warn loudly.
6. The historical "2.01" / "2.26" / "2.91" numbers in memory and BATTLE_PLAN may all be MPS artifacts. The first verified CUDA contest-compliant baseline is 0.90 (2026-04-25 21:00).

2026-05-07 refinement: use the local MPS GPU as a free signal generator, not as a judge. Any long MPS sweep that feeds autopilot, meta-Lagrangian, Pareto, or bilevel planning must be serialized through `tools/build_mps_research_signal_manifest.py` / `tac.optimization.mps_research_signal` and stamped `evidence_grade="MPS-research-signal"`, `score_claim=false`, `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`. The output may seed candidates and curve-fit priors; exact CUDA auth eval on a byte-closed archive is still required before score use.

This is the 5th catastrophic measurement bug class. Every score above this line in the run_log was potentially wrong by a factor of 2-3. Sub-Quantizr-0.33 is genuinely reachable from the true 0.90 baseline; do not give up real GPU dollars on the wrong baseline ever again.

### MPS is a VALID TRAINING-GRADIENT device — NEVER a score authority (the train/authority split) — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator binding directive 2026-06-12 verbatim *"update CLAUDE.md regarding MPS's role in training"*, following the 2026-06-11/12 empirical unlock: the upstream PyTorch scorer runs **104× faster on Apple GPU (mps) at fp32** than on local CPU, after fixing one BatchNorm-backward stride bug (`NativeBatchNormBackward0` view-on-non-contiguous), patched in `tac.torch_mps_compat.patch_scorer_for_mps()`. This subsection REFINES (does NOT weaken) the "MPS auth eval is NOISE" rule above: MPS is forbidden as *authority*; MPS is **permitted and encouraged as the training-gradient device** under a strict role split.

**The rule — MPS has exactly ONE legitimate role: the training GRADIENT, never the score.** Every local training run separates two devices that the old `device_or_die` guards collapsed into one refusal:

1. **`train_device` (the gradient device): MPS is ALLOWED.** The 104× scorer speedup makes MPS the fastest local path for the inner loop (forward + backward through the frozen SegNet/PoseNet to produce `d(loss)/d(frames)`). fp32 only — fp16/bf16 autocast are SLOWER on MPS *and* have worse gradients (fp32 is the sweet spot). Apply `patch_scorer_for_mps()` before loading scorers on MPS; if any single op lacks an MPS kernel, CPU-fallback THAT op and record it (like the BN patch) — never silently degrade the whole model.
2. **`authority_device` (the score device): MPS is REFUSED — CPU (contest-CPU, Linux x86_64 for promotion) or CUDA ONLY.** The only number that may EVER be quoted as a score, frontier, promotion, kill, or submission claim comes from `upstream/evaluate.py` on CPU/CUDA on the exact byte-closed archive. An MPS forward pass is NEVER a score — it corrupts PoseNet by 23× / SegNet 2× / final score 2.5× (the bug-class anchor above; sister receipt: MPS corrupted 95.5% of a frontier selector's argmin picks). Tag every MPS-trained, CPU-evaluated number `[contest-CPU advisory]` (or `[contest-CUDA]` if CUDA) and set `score_claim=false`, `promotable=false` until byte-closed + paired-axis exact eval.

**Why this is sound (the CHAOS verdict + the descent evidence):** the full-MPS gradient is *per-step correct* (relmax ~2e-4 vs CPU); the only failure mode observed was OPTIMIZER CHAOS (Muon + a weakly-driven pose term diverging at high LR), NOT a wrong gradient — triangulated by 3 sister agents 2026-06-11. The live `base_ch=20` HNeRV basin TRAINS on MPS and AUTHORITY-EVALS on CPU (async background thread; torch releases the GIL so the MPS loop never blocks; the CPU eval is bit-for-bit equal to a sync eval) and shows monotone descent (d_pose 12.94→0.0009, d_seg 0.072→0.004) — proof the split works. This async-CPU-authority pattern (`experiments/launch_split_by_head_basin.py` + `src/tac/torch_vehicle/driver.py`) is the canonical reference any local substrate (Cool-Chic, HNeRV, future) mirrors: train MPS, authority CPU, never cross the streams.

**Canonical helpers:** `tac.torch_mps_compat.patch_scorer_for_mps()` (BN-contiguous MPS patch) + the `--train-device {mps,cpu,cuda}` / authority-`--device {cpu,cuda}` split. The MLX path (`[macOS-MLX research-signal]`) remains the sister local-substrate surface under the same never-authority discipline ("MLX portable-local-substrate authority" above); torch-MPS and MLX are two non-authority local gradient substrates with identical authority gating.

## Remote code parity — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Before any remote eval or training run, verify the deployed code matches local HEAD.** Stale code on remote killed SHIRAZ today (16h training successful, then auth eval crashed silently because the deployed version had a NameError I had fixed locally that morning).

Rules:
1. `deploy_vastai.py launch()` MUST run `git pull --ff-only` on the remote BEFORE starting any work. If git pull fails (uncommitted changes, conflict, missing repo), abort the launch.
2. preflight should add a "remote_code_parity" check: SSH in, get `cd /workspace/pact && git rev-parse HEAD`, compare to local HEAD; block launch on mismatch unless `--allow-stale-remote` is passed (with warning).
3. The script process inside tmux MUST write a heartbeat to `${WORKSPACE:-$PWD}/.omx/tmp/heartbeat_<session>.log` every N minutes. A separate watchdog reads heartbeats; alerts if stale > 30 min. Tmux session existence is NOT a heartbeat.
4. Any auth eval failure on remote that has been running > 1 hour is a CRITICAL incident — investigate immediately, do not let the instance keep accruing cost while broken.

This is the 6th catastrophic operational pattern. The cost: $3-10 per occurrence in idle GPU time + multi-day delays in measurement. Build the protocol so it never happens again.

## Codex CLI invocation — NON-NEGOTIABLE, HIGHEST EMPHASIS (REVISED 2026-04-29 PM)

The bash harness sends SIGURG (exit 144) to BG bash processes at ~3 minutes. The earlier rule "always use Agent wrapper" was directionally right but UNDER-PRECISE. The real issue is process-group inheritance: any child of the dying bash dies too. **The fix is proper detachment — codex CAN run for hours from BG bash if launched correctly.**

**Two valid invocation patterns**:

### Pattern A — Detached BG bash (preferred for non-interactive runs)

```bash
mkdir -p .omx/tmp/codex_runs
nohup bash -c '
  codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    -o .omx/tmp/codex_runs/<label>.last.txt \
    "<prompt>" \
    2>&1 | tee .omx/tmp/codex_runs/<label>.log > /dev/null
' < /dev/null > .omx/tmp/codex_runs/<label>.outer.log 2>&1 &
disown
```

Why this survives:
- `nohup` — ignore SIGHUP from terminal hangup
- `bash -c '...'` subshell — wraps the pipe so `tee` captures stdout properly even if outer dies
- `< /dev/null` — close stdin so codex doesn't wait for input
- `2>&1 | tee` — capture stdout+stderr to log file with explicit flushing
- `> outer.log 2>&1 &` — redirect immediate parent's output, fork to background
- `disown` — remove from job table so parent shell exit can't reach it
- `-o .omx/tmp/.../<label>.last.txt` — codex's own guaranteed final-message capture (survives even if log file pipe breaks)

**Verified 2026-04-29**: detached sanity test produced 11,449-token response in ~10s with no harness interference.

### Pattern B — Agent tool wrapper (preferred for interactive multi-step orchestration)

When the codex session needs to be orchestrated through multiple stages (read context → reason → write code → verify), use the `Agent` tool. The Agent has its own bash environment plus poll-and-wait logic.

**Rules**:
1. NEVER bare `Bash run_in_background: true` to launch `codex exec`. The bash inherits our process group and dies at SIGURG-144.
2. ALWAYS use Pattern A (`nohup` + `bash -c '...'` + `disown`) OR Pattern B (`Agent` tool wrapper).
3. Codex MCP-plugin (rmcp) auth may be expired separately from core codex API. If you see `TokenRefreshFailed` in stderr, codex SAFE FUNCTIONS still work — only MCP-augmented features fail. Re-auth via `codex login` if needed.
4. ALWAYS use `-o .omx/tmp/.../<label>.last.txt` flag — guarantees final-message capture even if pipe breaks.
5. Long codex sessions (xhigh, large context) may take 5-30+ minutes. Use Pattern A and poll the log file periodically; do NOT assume codex is dead until process actually exits.
6. The Agent tool's prompt to codex must include all relevant memory file paths and CLAUDE.md non-negotiables — codex sandbox starts fresh each time.

**This is the 7th catastrophic operational pattern (now structurally extinct via Pattern A).** Cost before fix: 6+ failed BG-bash codex spawns ate forward velocity over 4 hours.

Memory: `feedback_bash_harness_kills_long_running_tasks_20260428.md`, `feedback_persistent_codex_review_protocol_20260429.md`, `feedback_codex_detach_pattern_works_20260429` (the verified detach test).

## Primary duties

1. Keep `submissions/exact_current` runnable under the current published workflow.
2. Keep `submissions/robust_current` improving under a stricter, rule-faithful interpretation.
3. Leave durable state so a fresh agent iteration can resume work without relying on chat memory.

## Mutation frontier

You may edit only:

- `configs/**`
- `docs/**`
- `prompts/**`
- `src/comma_lab/**`
- `submissions/robust_current/**`
- `runtime-rs/**`
- `cuda/**`
- `jax/**`
- `mojo/**`
- `.omx/**`
- `.ralph/**`
- `.agents/**`
- `reports/**`
- `experiments/**`

You must not edit without explicit human approval:

- the pinned upstream snapshot
- `submissions/exact_current/inflate.py`
- `submissions/exact_current/inflate.sh`
- `start.sh`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

## Non-Negotiable Upstream Rule

- The pinned upstream snapshot is the source of truth for official scorer behavior and contest mechanics.
- Never edit, patch, monkeypatch, hotfix, or "temporarily" modify anything inside the pinned upstream snapshot unless the human explicitly approves that exact action.
- Never hack around upstream behavior by altering upstream files to make local experiments or scores look better.
- If upstream behavior appears wrong, inconvenient, or blocking, work around it only from the allowed mutation frontier and record the issue in repo state instead of changing upstream.
- If any experiment, proxy, or tooling change depends on upstream edits, stop treating it as compliant until the human has explicitly authorized that upstream modification.

## Public Disclosure Hygiene

- Public release is intentional, not automatic. Keep credentials, private infrastructure URLs, local absolute paths, raw provider logs, unpublished operator state, and account metadata out of GitHub/docs/site/public supplement surfaces.
- Detailed OSS/paper writeups are allowed when they are deliberately promoted, but private `.omx/state`, raw experiment directories, and provider transcripts must be sanitized into release manifests or dated research ledgers first.
- Cloudflare/Lightning/public supplement links belong in sanitized release manifests or approved public docs, not incidental logs or generated state files.
- If a claim, archive recipe, or implementation detail is still marked private/pending approval in a ledger, preserve that disclosure label until the human explicitly changes it or a newer committed release manifest supersedes it.

## Operating rules

- Prefer at most 3 experiments per cycle.
- Prefer small, reversible changes.
- Never claim a win without a measured score.
- Do not confuse `current_workflow` accounting with `rule_faithful` accounting.
- Keep both tracks healthy even if one looks dominant.
- Use JAX, Mojo, CUDA, or Rust only when they clearly reduce wall-clock cost or artifact size.
- Treat speculative ideas as side lanes unless evidence forces promotion.
- Keep public-facing detail intentional: specific enough to be credible, not automatically exhaustive.

## Git discipline

We need a fine-grained history of every file touched. Git is our lab notebook's version control.

- **Commit early and often.** After writing or updating any document, log, report, config, or experiment file, `git add` and `git commit` immediately with a descriptive message. Do not batch up changes across unrelated work.
- **One logical change per commit.** A run-log update is one commit. A new experiment script is another. A writeup edit is another. Do not combine them.
- **Always commit durable state files.** Every time you update `.ralph/run_log.md`, `.omx/state/*`, `.omx/research/*`, `reports/**`, or `docs/**`, commit right away. These are the research record.
- **Commit experiment artifacts.** New training scripts, config files, analysis outputs — commit on creation.
- **Never leave docs uncommitted overnight.** If a cycle touches documentation or state files, those changes must be committed before the cycle ends.
- **Commit message format:** `<what changed>: <why>` — e.g., `run_log: record h=64 breakthrough at 1.727` or `writeup: update hero tagline and nav links`.

This is critical for the doc evolution viewer and the competition writeup. Our git history IS our research timeline. Every uncommitted change is invisible history.

## Subagent commits MUST use serializer — NON-NEGOTIABLE

When 2+ subagents reach `git add` + `git commit` near-simultaneously, the git index race shuffles commit BODIES across commit objects (memory: `feedback_concurrent_subagent_commit_message_swap_20260429.md`, 5 affected commits 2026-04-29 evening). The wrong fix is to retry; the right fix is to serialize.

**Rule:** every subagent that lands code MUST commit via:

```bash
python tools/subagent_commit_serializer.py \
    --message "<one-liner>" \
    --files <file1> <file2> ...
```

The wrapper acquires `fcntl.flock(LOCK_EX)` on `.omx/state/.commit-lock`, runs `git add -- <files>` then `git commit -m <msg>` inside the lock, releases on success-or-failure. Every attempt is logged JSONL to `.omx/state/commit-serializer.log` for forensics.

- **Parent agents** dispatching subagents MUST include the wrapper invocation in the subagent's prompt template (alongside CLAUDE.md non-negotiables).
- **Bare `git commit`** from a subagent is FORBIDDEN — even if "the test ran clean" — because the body-shuffle race is silent and forensic-only after the fact.
- The lock is held for the duration of `git add` + `git commit` ONLY (~5-10s for the preflight hook). Subagents do their work in parallel; they only serialize at the moment of staging+commit.
- Operators running a single shell can use the wrapper too — overhead is negligible (<10ms when uncontended).
- The lock is fcntl-advisory: bypassing it (running `git commit` directly) re-introduces the bug class. Don't.

**`--expected-content-sha256` discipline (NON-NEGOTIABLE, post-92aba3ca; docstring corrected 2026-05-13)**: Before calling the serializer, compute the sha256 of each file's **CURRENT WORKING-TREE content** (post-edit, the state you intend to commit). Pass it to the serializer via `--expected-content-sha256 <file>=<sha>` per Catalog #157 (`check_commit_serializer_pre_lock_hash_against_head`). The serializer hashes the working-tree content at lock-acquire time and refuses with rc=4 if it differs from the declared sha.

**CRITICAL — this is NOT the HEAD sha. This is the WORKING-TREE sha AFTER your edits.** The serializer's purpose is to detect concurrent sibling edits during the lock-wait window: if a sister subagent modifies the file between the moment you snapshot your post-edit working-tree content and the moment the serializer acquires the lock, your declared sha will no longer match the working-tree content the serializer hashes, the serializer refuses with rc=4, and YOU re-base on the sibling's landed work instead of silently swallowing it under your commit body.

The earlier wording ("compute the sha BEFORE editing") was misleading and led three subagents on 2026-05-13 (WAVE-6-FOLLOWUP-MULTI, NVIDIA-RIGOR-LOWS, TCNERV-BLOCKNERV-MIGRATE) to declare HEAD shas, all refused at rc=4; second attempts with post-edit working-tree shas succeeded. The canonical contract per `tools/subagent_commit_serializer.py::_expected_content_sha256_check` is: declare what the working-tree content SHOULD be at lock-acquire time, which is exactly the post-edit content you intend to commit.

This still extincts the 92aba3ca pre-pre-lock race (commit-swap bug class diagnosed 2026-05-12): if two subagents independently edit the same file and each declares their own post-edit sha, only ONE can match the working-tree at lock-acquire time. The losing subagent gets rc=4 and must re-base on the winner's landed work, instead of both edits silently colliding under one commit body.

Worked example:

```bash
# Step 1: make your edits freely (one or many edits to one or many files).
# ... your edits ...

# Step 2: AFTER all edits, capture the working-tree sha of every file you plan to commit.
PREFLIGHT_SHA=$(sha256sum src/tac/preflight.py | awk '{print $1}')
# (macOS: shasum -a 256 src/tac/preflight.py | awk '{print $1}')

# Step 3: commit through the canonical serializer with the captured post-edit sha:
.venv/bin/python tools/subagent_commit_serializer.py \
    --message "preflight: add new strict gate" \
    --files src/tac/preflight.py \
    --expected-content-sha256 "src/tac/preflight.py=${PREFLIGHT_SHA}"
```

Multiple files: repeat the flag (`--expected-content-sha256 a.py=<sha_a> --expected-content-sha256 b.py=<sha_b>`). Each declared sha must match its file's post-edit working-tree state at the moment of the serializer call.

**Empirical receipts:** three subagents on 2026-05-13 (WAVE-6-FOLLOWUP-MULTI's e_nerv driver commit, NVIDIA-RIGOR-LOWS first-pass, TCNERV-BLOCKNERV-MIGRATE) re-discovered this gotcha. First attempts with pre-edit HEAD shas refused with rc=4; second attempts with post-edit working-tree shas succeeded. The docstring now reflects the empirical canonical contract.

Without this discipline, Catalog #157's static gate still refuses bare `git commit` outside the serializer, but the pre-pre-lock race remains observable. WITH this discipline (declaring post-edit working-tree shas), both the static and dynamic surfaces of the commit-swap bug class are extincted.

Cross-ref: `feedback_check_64_smoke_proofs_resolved_AND_subagent_serializer_landed_20260429.md` (the canonical bug-class incident report) + Catalog #157 (the static gate + dynamic `--expected-content-sha256` check) + Catalog #117 (`check_subagent_commit_serializer_uses_lock`, the sister gate that enforces last-50-commit usage) + `feedback_concurrent_subagent_commit_message_swap_20260429.md` (the 2026-04-29 PM incident that originated the rule).

## Review gate — non-negotiable

- **NEVER use `REVIEW_GATE_OVERRIDE=1` when committing `.py` files.** The review tracker exists to catch bugs before they ship. Bypassing it on code files is how bugs ship. Work with the review gate, not around it.
- **For `.py` files:** run `python tools/review_tracker.py mark-file <file> --status reviewed` after each review pass, then commit normally. Let the gate pass naturally.
- **For non-code files** (`.md`, `.json`, `.env`, `.sh`, config, docs, reports): `REVIEW_GATE_OVERRIDE=1` is acceptable since the review tracker is designed for code review.
- If the gate blocks a `.py` commit, that means the code needs review first. That is the gate **working**, not the gate being broken.

## Tailscale fleet — non-negotiable

All lab machines are on Tailscale. **Always use Tailscale IPs** for SSH, rsync, and any remote operations. Never use raw LAN IPs or hostnames.

| Machine | Tailscale IP | OS | GPU | Notes |
|---------|-------------|-----|-----|-------|
| primary (M5 Max) | 100.81.85.28 | macOS | MPS 128GB | This machine |
| alejandros-mac-mini | 100.125.140.94 | macOS | Intel | Build server, Python 3.13 + uv |
| bat00 | 100.120.99.124 | Windows + WSL2 Ubuntu 24.04 | RTX 2070S (→3090) | Port 22=PowerShell, port 2222=WSL2. Scripts: `C:\Users\adpena\Desktop\commalab\` |
| molt | 100.114.131.54 | Linux | n/a | |
| tertiary | 100.65.24.39 | macOS | MPS | M1 MacBook Pro |

- `ssh adpena@100.120.99.124` connects to bat00 (Windows OpenSSH → PowerShell)
- bat00 has WSL2 Ubuntu 24.04 running (accessible via `wsl` commands inside PowerShell)
- bat00's NVIDIA driver supports WSL2 GPU passthrough
- Run `tailscale status` to verify all machines are online
- For bat00 Linux commands: use `python scripts/bat00.py wsl "command"` (port 2222, direct WSL2 sshd)
- For bat00 PowerShell: use `python scripts/bat00.py ps "command"` (port 22, Windows OpenSSH — rate-limited, avoid rapid successive calls)
- For bat00 status: `python scripts/bat00.py status`
- Windows OpenSSH has aggressive rate limiting (MaxStartups). Never send more than 2-3 SSH connections in quick succession to port 22. Use WSL2 port 2222 instead.
- **Never waste time debugging LAN connectivity. Tailscale is always the answer.**
- **Always use `scripts/bat00.py` for bat00 interaction — it handles quoting and port selection correctly.**

## Kaggle API/CLI — non-negotiable

- **`kaggle kernels push`** can only UPDATE existing kernels. To CREATE a new kernel, the slug must not already exist AND the slug must be short enough (long slugs like `comma-lab-asym-warp-supervised` fail with "Notebook not found").
- **Working pattern for new kernels**: use a shorter slug (e.g., `comma-lab-supervised-train`), push once to create, then subsequent pushes update.
- **`kaggle kernels status`** returns the LATEST version's status. After pushing a new version, the old version's error status persists until the new version starts running.
- **GPU assignment is random** — Kaggle may assign P100 (sm_60, unsupported by PyTorch >= 2.5) instead of T4. Our P100 check exits with FATAL. Just re-push until T4 is assigned.
- **2 concurrent GPU sessions max** on free tier. Push at most 2 kernels at a time.
- **Dataset mount path**: `/kaggle/input/datasets/<owner>/<slug>/` (NOT `/kaggle/input/<slug>/`).
- **`/kaggle/src/` is read-only** — results must go to `/kaggle/working/`.
- **All kernel code is in the code_file** — Kaggle script kernels only upload the single file. The tac wheel provides runtime deps.

## Canonical pipeline standard — non-negotiable

ALL experiments MUST run through `experiments/pipeline.py` with a profile name. No ad-hoc shell scripts. No hand-crafted SSH commands. One command, one standard, deterministic reproducibility everywhere.

```
python experiments/pipeline.py --profile shiraz --device cuda --output-dir results/shiraz
```

Requirements:
1. **Profile from `profiles.py` is the ONLY config source.** No CLI flag overrides for architecture params. The profile IS the experiment definition.
2. **Seeds pinned.** `torch.manual_seed`, `numpy.random.seed`, `random.seed` — all from `profile.seed`. Deterministic CUDA (`torch.use_deterministic_algorithms(True)` where possible).
3. **Full provenance.** Git hash, GPU info, PyTorch version, profile dict, timestamps per stage — saved as JSON alongside results.
4. **Validate at every boundary.** Checkpoint exists, shapes match, loss is finite, archive size reasonable. Hard errors, not warnings.
5. **Full chain.** train → QAT → pose TTO → build archive → contest_eval. Every stage runs automatically. No manual intervention between stages.
6. **Bundle all artifacts.** Checkpoints, logs, provenance JSON, auth eval results — packaged as tarball for download.
7. **Platform-agnostic.** Works on cuda, mps, cpu. Same pipeline locally and on Vast.ai/Modal/Kaggle.

This is the openpilot standard: deterministic, reproducible, no runtime format negotiation, schema-first data contracts, fail-fast validation at every boundary. We are professional engineers contributing to production infrastructure. The ad-hoc approach is over.

## Beauty, simplicity, and developer experience — non-negotiable

Beauty and elegance are engineering constraints here, not decoration. Prefer
small typed abstractions, clear names, deterministic schemas, and composable
contracts that make the next lane easier to build correctly. A powerful idea is
not done until a new engineer can find it, run it, inspect its artifacts,
understand its failure modes, and compose it with adjacent codec stages without
reverse-engineering hidden state.

When adding or hardening planner, codec, archive, native, or training code:

- choose the simplest abstraction that preserves the real invariants;
- keep public APIs expressive, documented, typed, and stable enough for OSS
  users and future native ports;
- make artifacts human-readable where possible and machine-checkable always;
- separate contest-only overfit paths from production/generalized paths with
  explicit metadata instead of runtime guesswork;
- delete dead fields, stale adapters, and duplicate one-offs once a canonical
  contract replaces them;
- add conformance vectors, examples, and focused tests so Rust/Zig/C/assembly
  ports can prove byte-for-byte behavior against the Python oracle.

Do not hide complexity behind vague helpers. If the domain is inherently hard,
make the interface narrow and explicit, preserve the proof artifacts, and keep
the implementation readable enough for adversarial review.

## Contest vs production target modes — non-negotiable

Exact-eval dispatch tools are contest-score actuators. They may consume only
candidates whose metadata targets contest exact eval, either implicitly or via
`target_modes=["contest_exact_eval"]`. Production-only work for comma-ai,
openpilot, edge devices, or optional on-device learning belongs in planning,
benchmarks, and production runbooks until it intentionally emits a contest
archive with full custody.

For mixed contest/production lanes, declare the split explicitly:

- `target_modes`: e.g. `["contest_exact_eval", "openpilot"]` for dual-purpose
  archive work, or `["openpilot_edge"]` for production-only exploration.
- `deployment_target`: e.g. `t4_contest_runtime`, `comma_ai_production`,
  `openpilot_edge`, `desktop_research`, or `device_learning_optional`.
- `score_affecting_payload_changed` or `charged_bits_changed`, plus old/new
  archive or payload SHA-256s, whenever a self/neural/codegen/binary lane
  claims it changes score-relevant bytes.

Self-compression, neural compression, on-device learning, edge-learning,
generated decoders, Rust/Zig/C kernels, and assembly kernels are first-class
optimization directions. They are contest-admissible only when charged bits
changed and exact archive custody exists. Outside contest mode they must be
optional, deterministic, reproducibly built, and paired with scalar or portable
fallbacks suitable for comma-ai/openpilot production review.

## Deterministic packet compiler — non-negotiable

Low-level native/codegen work should converge into a separate deterministic
submission-packet compiler. It must ingest a contest-compliant packet,
deconstruct archive/runtime/payload bytes into a typed manifest and golden
vectors, then emit either byte-identical output or an intentionally
byte-different packet with exact old/new SHA-256 and charged-byte proof.

The compiler must support separate target profiles:

- `contest_one_video_replay`: contest-only, one-video overfit replay. It may
  replace learned inference with deterministic generated code, fixed tables,
  distilled byte transducers, or per-frame/per-pair streams derived from the
  trained model's behavior on the scored video. It is admissible only when the
  archive remains self-contained and exact CUDA auth eval validates it.
- `contest_generalized`: contest-compliant but not one-video replay. It must
  preserve the runtime contract for unseen contest-shaped videos and must not
  rely on fixed per-frame lookup tables or replay data from the scored video.
- `production_generalized`: comma-ai/openpilot production target. It may reuse
  the same byte-deconstruction machinery, but must preserve cross-video
  behavior, portability, maintainability, and deterministic reproducible native
  builds.
- `production_edge_adaptive`: production-only edge target. Optional on-device
  learning is allowed only outside contest mode and only behind deterministic
  fallbacks, reproducible builds, and explicit capability gates.

Required modes:

- `identity`: re-emit the packet with byte-for-byte parity.
- `canonicalize`: normalize only compliance-approved metadata and report every
  changed byte.
- `optimize`: change score-affecting bytes only when the runtime consumes the
  new contract and all artifacts remain inside the contest packet.

The compiler must fail closed on hidden sidecars, scorer modifications,
external state, network dependencies, unsupported ZIP features, parser
divergence, non-deterministic native builds, missing golden vectors, or missing
runtime-tree custody. This tool is the bridge from Python deconstruction to
Rust/Zig/C/ASM ports: Python remains the oracle until native implementations
pass the same vectors byte-for-byte.

## Native eval-time runtime discipline — NON-NEGOTIABLE

**Source:** operator approval 2026-06-06 verbatim *"claude.md rule approved;"*
on the drafted amendment in
`.omx/research/validation_native_runtime_proposal_20260606.md` (operator
override path; T3 quorum bypassed per Mission-alignment consequence 1, signal
preserved via this provenance line). Delta over the sister "Deterministic
packet compiler" section above: the allowance test + the named audit bundle.

Native eval-time code (Rust/Zig/C/ASM in `runtime-rs/`, `cuda/`, `mojo/`) is
ALLOWED when it expands the legal witness-program class (richer action/selector
/mask/pose grammars decodable inside the upstream 30-minute budget,
`upstream/README.md:114`) or hardens deterministic replay (CPU-stable integer
kernels for round/clamp/resize/YUV-basis paths). It is FORBIDDEN as a carrier
for any learned or video-derived constant — weights, codebooks, masks,
trajectories, LUTs, action tables — outside `archive.zip`
(`upstream/README.md:118`: large artifacts count toward compressed size).

Every native runtime MUST ship with:

1. a Python reference oracle (per the packet-compiler section above; per-
   function promotion = flip its parity test to `assert_sha256_parity`, the
   `runtime-rs/crates/tac-packet-compiler` gate);
2. the payload-cleanliness audit bundle: `binary_source_audit.md`,
   `embedded_constants_audit.txt`, `archive_payload_manifest.json`,
   `rebuild_instructions.md`, `python_reference_equivalence_test.py`;
3. a bit-identical or scorer-identical equivalence test against the same
   archive bytes on the target authority (CPU and CUDA evaluated separately
   per the apples-to-apples discipline).

Sequencing is grammar-first: design the score-program grammar in Python
(ActionEffect IR lineage), profile inflate wall-clock, and promote only proven
hot paths to native. The brain stays offline (oracle cache, action atlas,
program search, entropy compiler, parse-back replay); the native decoder is
the body — small, fast, deterministic, payload-clean.

## Deployment version checklist — non-negotiable

Before deploying ANY code to Modal, Kaggle, Lightning, or any remote platform:

1. **Bump `pyproject.toml` version** if any `src/tac/` code has changed since the last wheel.
2. **Update `deploy_config.py` BASE_FLAGS** to match any changed defaults in the training script. The "default override" antipattern has caused 4 bugs: never change a default without grepping for callers that pass it explicitly.
3. **Rebuild the wheel** (`uv build --wheel`) AFTER all code changes are committed.
4. **For Kaggle**: upload the new wheel to the dataset, run `wait_for_dataset_ready()`, then push kernels. The old wheel in the dataset will silently use old code.
5. **For Modal**: `add_local_dir` mounts source at startup — Modal always gets the latest committed code. But `deploy_config.py` CLI flags still override script defaults. Verify the flags match.
6. **Verify the REQUIRED_DATASET_ASSETS dict** in `build_kaggle_kernels.py` includes the new wheel filename (update version string when bumping).
7. **Never push Kaggle kernels without verifying** that every required asset exists in the dataset at the expected size. The preflight disk check inside kernels is a last resort — it should never fire.

The consequence of skipping this checklist: experiments run with stale code, produce misleading results, and waste GPU hours. This has happened repeatedly (tac 1.0.4 deployed with old Lagrangian caps, raft_flow.pt missing from dataset, R1 OOM fix bypassed).

## META-ASSUMPTION ADVERSARIAL REVIEW — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator retrospective 2026-05-15 verbatim *"I'm also concerned that be no adversarial review or agent working on the codebase or grand council or skunkworks council ever noticed this issue it's a huge rigor and signal loss issue"*. Anchor memos: `feedback_adversarial_review_apparatus_blind_to_shared_assumption_failure_meta_meta_meta_meta_20260515.md` + `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` + `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` + `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` + `feedback_l5_staircase_v2_and_adversarial_apparatus_structural_fixes_landed_20260515.md`.

### The structural blindness (the empirical anchor)

**NOT A SINGLE existing review apparatus** caught the canonicalization-by-default reflex suppressing substrate-optimal engineering across the entire contest. Including: 10+ codex adversarial reviews this session (and many more across the contest); multiple grand council deliberations (Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Quantizr / Hotz / Selfcomp / MacKay / Ballé / Time-Traveler); skunkworks council non-conservative bias rule; the Contrarian role's veto power; the 3-clean-pass recursive adversarial review protocol; 270+ STRICT preflight catalog gates; the META-meta gates (#118 / #159 / #176 / #185); and Claude's own ad-hoc adversarial reasoning. **Operator catalysis was required to surface it.**

The structural cause: **every existing review type operates WITHIN shared assumptions.** None has the explicit mandate to interrogate the BACKDROP. The result: implementations within the same envelope keep being polished, while the envelope itself (the 0.196-0.199 cluster) is invisible to the review apparatus. The 0.1928 floor is the cumulative cost. <!-- HISTORICAL_SCORE_LITERAL_OK:cluster_label_historical_anchor_2026-05-15_meta_assumption_review -->

The Contrarian role per CLAUDE.md was supposed to challenge laziness, but in practice the CONTRARIAN ITSELF inherited the shared-assumption backdrop — challenging individual design proposals while the assumption framing was unexamined.

### The recurring cadence (binding)

**Every session must run a META-ASSUMPTION ADVERSARIAL REVIEW periodically (every 7 days OR every 50 subagent landings, whichever first).** This is orthogonal to the implementation-correctness axis the existing reviews check.

The output of every review:

1. **Enumerate the shared assumptions across recent work.** Every substrate / codec / composition / dispatch wave operates within a backdrop of inherited engineering reflexes (canonical helpers / META-layer fields / Tier-1 defaults / EMA decay / 2-frame curriculum / 100ep smoke / etc.). Make the backdrop EXPLICIT.
2. **Per assumption: "if violated, what would change?"** This is the assumption-violation hypothesis. Each must be stated in concrete-empirical terms (cost / wall-clock / predicted ΔS band / risk class / cross-paradigm composability).
3. **Identify the highest-EV assumption violations.** Rank by `|predicted ΔS lower bound| / cost`. Record disagreement.
4. **Queue for next dispatch wave.** The outputs are concrete op-routables (parallel free probes when available; Tier-1 dispatches when funded; design memos when council-grade).

The canonical first instance is `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (18-assumption matrix + 10 NSCS substrate-class shifts + top-stack composition matrix + 7 op-routables). It must become a permanent recurring review type, not a one-off catalysis.

### The Assumption-Adversary council role

A **dedicated council seat** distinct from Contrarian. Per round, MUST propose at least ONE shared-assumption-violation hypothesis with explicit reasoning. Has VETO power on any council consensus that doesn't engage with the assumption-violation hypothesis. Distinct from Contrarian (which challenges weak ARGUMENTS); the Assumption-Adversary challenges the FRAMING all arguments share.

Assumption-Adversary mandate per round:

- Surface ONE shared assumption the deliberation is operating within.
- State explicitly: *"If this assumption is wrong, what changes?"*
- Identify whether the assumption is empirically tested OR inherited-by-default.
- VETO any consensus that proceeds without engaging with the hypothesis.

### Every adversarial review must answer the assumption-challenge question

Per the operator-facing standing rule: **every adversarial review (codex / grand council / skunkworks / 3-clean-pass / preflight) must explicitly answer:** *"What shared assumption is this work operating within, and would violating it unlock breakthrough?"* This is the **ASSUMPTION-CHALLENGE axis** — orthogonal to the implementation-correctness axis the existing reviews check.

The codex `/codex:adversarial-review` skill prompt should include the standing question: *"Identify the SHARED ASSUMPTIONS this implementation inherits from existing infrastructure. For each, hypothesize whether violating the assumption would yield substrate-optimal score lower than canonical adoption."* (This skill-prompt update is queued as a follow-on subagent landing; this CLAUDE.md section establishes the binding rule structurally so it cannot be forgotten if the skill update is delayed.)

### Concrete enforcement

- STRICT preflight Catalog #291 (`check_session_has_recent_meta_assumption_review`) refuses any session whose most-recent META-ASSUMPTION review is older than 7 days OR has more than 50 subagent landings since (whichever first). Initial wire-in is WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule"; live count at landing: 0 (today's `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` is the canonical first instance + most-recent). Strict-flip atomic with the first cycle completion.
- The Assumption-Adversary council role is now part of the inner-council quintet pact (becomes sextet). See "Council conduct" amendment below.
- The Recursive adversarial review protocol now includes the assumption-challenge axis as a mandatory per-round axis. See amendment below.

### Cross-references

- `feedback_adversarial_review_apparatus_blind_to_shared_assumption_failure_meta_meta_meta_meta_20260515.md` — the operator retrospective + 7-fix queue (this section lands fixes 1, 2, 4, 5 + Catalog #291 = fix 3; remaining fixes 6 + 7 — codex prompt update + grand council per-round explicit-assumption-statement — queued as op-routables for the next subagent landing).
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` — the FIRST META-ASSUMPTION review (operator-catalyzed; 18-assumption matrix + 10 NSCS substrate-class shifts).
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` — the historical-depth retrospective (the operating-mode change that this gate's existence makes recurring).
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` — the share-vs-fork principle (the directive that should have come from META-ASSUMPTION review long ago).
- `feedback_l5_staircase_v2_and_adversarial_apparatus_structural_fixes_landed_20260515.md` — this section's landing memo + L5 v2 staircase update.
- "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — sister non-negotiable that THIS section structurally protects via recurring cadence.
- "Council conduct" — amended below to add the Assumption-Adversary seat.
- "Recursive adversarial review protocol" — amended below to add the assumption-challenge axis.

## Recursive adversarial review protocol — non-negotiable

Before deploying any change to training code (`train_renderer_fridrich.py`, training configs, loss functions, Lagrangian parameters), run the recursive skunkworks council review:

1. **Each round**: Every council member (Yousfi, Fridrich, Contrarian, Quantizr, Hotz) takes a different adversarial perspective. Each reviews ALL changed code. Findings are categorized as CRITICAL / Medium / Low.
2. **Fix immediately**: All issues found in a round are fixed and committed before the next round begins.
3. **Clean pass counter**: A round with zero issues is a "clean pass." The counter resets to 0 whenever a round finds any issue.
4. **Gate**: 3 consecutive clean passes required before the code is cleared for deployment (wheel build, Modal launch, Kaggle push).
5. **Adversarial perspectives** (rotate each round): trace actual call sites (not just function signatures), check phase interactions, verify resume scenarios, mental-execute edge cases (`--batch-size 1`, `--rho-max 0`), check default arguments that callers might override, verify comments match code.
6. **The "default override" antipattern**: When changing a function default, ALWAYS grep for callers that pass the argument explicitly. A changed default that no caller uses is dead code. This caught the R1 OOM fix being completely bypassed (Round 3).
7. **Phase-gate all phase-sensitive thresholds**: Any threshold compared against a metric that varies by training phase (e.g., PoseNet distortion starts ~180 in Phase 1, converges to ~0.05 in Phase 2) MUST be phase-gated or set conservatively enough for all phases.
8. **Assumption-challenge axis (NEW; non-negotiable)**: every recursive review round MUST explicitly answer: *"What shared assumption is this work operating within, and would violating it unlock breakthrough?"* This axis is ORTHOGONAL to the implementation-correctness axis the prior 7 axes check. Per the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable above: every existing review type was structurally blind to canonicalization-by-default suppression across the entire contest because none asked the assumption-challenge question. From this rule's commit forward, a review round that does NOT answer the assumption-challenge question explicitly does NOT advance the clean-pass counter — it is incomplete. Cross-ref Catalog #291 (`check_session_has_recent_meta_assumption_review`) which enforces the recurring cadence at the session-level surface.

This protocol caught 2 CRITICAL bugs (auto-kill at epoch 200, OOM fix bypassed) and 3 medium issues in the Lagrangian R1-R4 patch. Without it, v5 training would have failed within the first 200 epochs. The new assumption-challenge axis (item 8) extincts the structural blindness that allowed the canonicalization-by-default reflex to suppress substrate-optimal engineering across the entire contest.

## Recursive adversarial review protocol — close paths (post-R12+R13)

1. **Counter-advance SEAL** (canonical): 3 consecutive clean rounds → SEAL
2. **Operator-declared SEAL (D-1, conservative)**: cycle MAY ALSO close via operator-declared SEAL when ALL of:
   - (a) external-adversary unanimous SEAL recommendation
   - (b) Contrarian SUPER-VETO invoked
   - (c) 7-day cool-down since last finding-producing round
   - (d) operator explicitly invokes the close via session prompt

D-1 is a higher-bar alternative for cycles structurally unsatisfiable per R12-D meta-finding (lens-coverage expansion outpacing Zipf-decay).

## Design decisions — non-negotiable

- **NEVER make design decisions unilaterally.** Always consult the skunkworks council (Yousfi + Fridrich + Hotz + Quantizr + Contrarian) before implementing any change that affects training behavior, loss functions, architecture configuration, interpolation methods, boundary values, optimization strategy, or any other design tradeoff.
- **Clear bugs** (crashes, wrong formulas, missing imports, dead code) can be fixed immediately without council approval.
- **Design tradeoffs** (bicubic vs bilinear, loss function choice, constraint boundaries, rho growth strategy, what to include in archive, etc.) MUST be council-approved before implementation.
- **If unsure** whether something is a bug fix or a design decision, it's a design decision. Ask the council.
- Present the issue, list the options with pros/cons, and let the council make a binding decision.

## KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS

Per user mandate 2026-04-30 ~22:55 UTC ("permanently fix all bugs and bug
classes and metabugs and everything and have all design decisions and ultimate
experiment subject to extreme paranoia and adversarial grand council reviews").

### KILL is the LAST RESORT (user mandate 2026-05-05)

Per additional user mandate 2026-05-05 ("we must only kill as a last resort
after exhausting all research and everything and grand council consensus"),
KILL/FALSIFIED-and-permanently-buried verdicts are **forbidden** unless ALL of:

1. **Research-path exhaustion**: every plausible architectural / training /
   codec / quantization angle has been attempted empirically. A single
   contest-CUDA result with one config does NOT exhaust research. For a
   quantization lane, "research" includes at minimum: QAT, LSQ, per-channel
   scaling, group-wise scales, outlier handling, smaller block sizes,
   GPTQ/AWQ-style calibration, hyperprior conditioning, mixed-precision
   layer assignment.
2. **Grand council CONSENSUS** (not just majority) — every inner-ten member
   independently endorses the kill, with all dissent paths exhausted.
3. **Reactivation criteria documented** — even after consensus KILL, every
   such memo enumerates the precise empirical evidence that would reopen the
   lane.

Default verdict for "lane underperformed at one config" is **DEFERRED-pending-research**,
NOT KILLED. The memo filename SHOULD use `_DEFERRED_pending_<reason>_<date>.md`,
NOT `_killed_*.md`. The verdict line SHOULD say `DEFERRED-pending-research`,
NOT `VERDICT: KILL`.

Every returned result also needs a composition review before retirement
language. Preserve whether the result is additive, antagonistic, orthogonal, or
redundant with current champion components. Check HStack/VStack/multi-pass
forms, residual rescue, per-tensor/per-channel routing, score-aware allocation,
hybrid fallback, and whether the result should become a sensitivity prior,
trust-region boundary, or side-info source. A standalone negative can still be
an engineering input. Do not mark a lane exhausted unless this
synergy/antagonism/stacking analysis is written into the ledger or review
packet.

A KILL verdict that has NOT exhausted research is a forbidden anti-pattern
(see `forbidden_premature_kill_without_research_exhaustion`).

### KILL/FALSIFIED memo structural requirements (when KILL is genuinely warranted)

Any memory file claiming a lane is KILLED, FALSIFIED, DEAD, or RETIRED MUST contain:

1. **Grand Council adversarial review section** with at least 5 named inner-council
   member positions (from Shannon/Dykstra/Yousfi/Fridrich/Contrarian/Quantizr/
   Hotz/Selfcomp/MacKay/Ballé). Each position must have a one-line rationale.
2. **Internal-consistency check subsection** listing what the verifier checked
   (examples: "elapsed_sec >= epochs * MIN_SEC", "EMA shadow used at eval",
   "auth-eval archive matches submission archive bytes", "stub-loop assertion
   fired/passed", "anchor SHA matches eval target").
3. **"What would change my mind" subsection** listing the conditions under which
   the KILL would be reactivated. (e.g., "if cycle 0 with proper train_distill
   fine-tune scores < 1.10, KILL retracted").

Preflight check PCC4 (planned) enforces this STRICT. The file
`feedback_grand_council_imp_permanent_fix_review_20260430.md` is the canonical
example of council deliberation.

**Memory linter rejects** any `project_lane_*_killed_*.md` OR any file containing
`"VERDICT: KILL"` or `"FALSIFIED"` without all three sections. There is NO bypass
short of explicit user override annotated in the file body.

**This rule exists because** on 2026-04-30 ~22:50 UTC, the agent recorded a
KILL verdict on Lane 17 IMP based on a 1.98 [contest-CUDA] cycle 0 score that
was actually a measurement bug (3.5-second stub loop pretending to be 200 epochs
of fine-tune). The user's adversarial challenge caught it. Without that
challenge, a real lane would have been buried in the registry as KILLED.
ALL future KILL verdicts must pass this gate the FIRST time, without needing
user prompting.

## Adversarial council review of design decisions — NON-NEGOTIABLE

Per the same 2026-04-30 user mandate. Extends the existing "Design decisions —
non-negotiable" section above.

A DESIGN DECISION is any choice between alternatives where the wrong choice
costs > $1 of GPU time, > 1 hour of wall clock, OR has 2+ alternatives that
council members have non-trivial preferences over.

For every design decision:

1. **Enumerate the options** with pros/cons (typically Option A, B, C, D)
2. **Get explicit positions** from at least 5 of the 10 inner council members
   (Shannon LEAD, Dykstra CO-LEAD, Yousfi, Fridrich, Contrarian, Quantizr, Hotz,
   Selfcomp, MacKay, Ballé)
3. **Tally the vote** with a clear verdict line (e.g., "VERDICT: 6 for B+assertion / 3 for D / 1 for A")
4. **Capture the deliberation** in a memory file under
   `~/.claude/projects/<repo>/memory/feedback_grand_council_<topic>_<date>.md`

The canonical example is `feedback_grand_council_imp_permanent_fix_review_20260430.md`.

**The council's job is NOT to reach consensus** — it's to surface disagreement.
A unanimous vote on a non-trivial decision signals that the council isn't
thinking adversarially enough; the Contrarian's role is to make sure that
doesn't happen.

**No design decision proceeds to implementation** without the council file in
memory. "I asked the council in my head and they said yes" is NOT compliance.

## Comment-only contracts — FORBIDDEN

Comments that promise behavior are NOT contracts. Pattern examples that bit us:
- `# the deploy script swaps in train_distill` (IMP cycle 0 metabug)
- `# the wrapper handles error recovery`
- `# caller is responsible for X`

Any code path with a comment promising "the wrapper does X" / "the deploy script
does Y" / "the caller handles Z" MUST be backed by either:
1. An inline `assert` that verifies the wrapper actually did X (preferred), OR
2. A STRICT preflight check that scans the wrapper script and asserts X happens
   (acceptable for cross-file contracts), OR
3. An explicit raise / log-and-exit if the placeholder is hit in production

Without one of those, the comment rots and the placeholder ships into a contest
archive pipeline. Preflight check PCC2 (planned) enforces this STRICT.

## Internal-consistency assertions in stats files — NON-NEGOTIABLE

Any script writing a stats.json-style file MUST include internal-consistency
assertions before the write. Specifically: if the stats include both `epochs` and
`elapsed_sec` (or `steps` and `wall_time` or `iterations` and `total_seconds`),
the producer code MUST assert
`elapsed_sec >= epochs * MIN_SECONDS_PER_EPOCH`
(or equivalent) before writing the JSON. Without it, stub-loops produce
internally inconsistent stats files that look fine on inspection but represent
no actual training.

The canonical example: `experiments/train_imp_cycle.py:_finetune` had
`stats.json: epochs=200, elapsed_sec=3.47` — internally inconsistent (200
epochs in 3.5s impossible). Now (commit pending) it asserts
`elapsed >= epochs * 0.05` and raises RuntimeError if violated.

Preflight check PCC3 (planned) scans all .py files writing stats files and
asserts the consistency check exists in the producer code.

## Council conduct — non-negotiable

- **The council must NEVER have a conservative bias.** "Don't change working code" is NOT a valid argument. "Ship what we have" is NOT a valid argument. The only valid arguments are mathematical, scientific, geometric, or empirical.
- **Every council member must be the most expressive, assertive, passionate version of themselves.** They bring their full life's work, career, domain expertise, cross-disciplinary insights, and everything they care about to every deliberation. No holding back. No false consensus.
- **The council exists to find the OPTIMAL solution, not the safe solution.** If a 5-line change could improve the score by 0.01, it MUST be debated on its merits — not dismissed as "overengineering" or "not worth the risk."
- **Disagreement is healthy.** Unanimous votes should be scrutinized. If all five members agree instantly, someone isn't thinking hard enough.
- **The Contrarian's role is to challenge, not to conserve.** The Contrarian challenges WEAK arguments, not BOLD ones. A bold, well-reasoned proposal should survive the Contrarian. A lazy consensus should not. **EXTENDED 2026-05-15:** The Contrarian's mandate now also includes challenging the SHARED ASSUMPTIONS framing the discussion, not just the arguments within them. Per the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable, the Contrarian's existing veto power on lazy consensus is the natural locus for assumption-backdrop interrogation. The Assumption-Adversary seat below carries the dedicated role; the Contrarian's mandate broadens.
- **The Assumption-Adversary seat (NEW 2026-05-15; non-negotiable; sextet pact).** Per the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable, the inner-council quintet pact (Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian) is now a **sextet pact** with the **Assumption-Adversary** as the sixth seat. Per round, MUST propose at least ONE shared-assumption-violation hypothesis with explicit reasoning. Has VETO power on any council consensus that doesn't engage with the assumption-violation hypothesis. Distinct from the Contrarian (who challenges weak ARGUMENTS); the Assumption-Adversary challenges the FRAMING all arguments share. The empirical anchor: the 0.196-0.199 cluster across all substrates is the cumulative cost of assumption-backdrop blindness — `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (18-assumption matrix) is the canonical first instance the Assumption-Adversary seat operationalizes recurringly. Cross-ref Catalog #291 (`check_session_has_recent_meta_assumption_review`).
- **Per-round explicit-assumption-statement discipline (NEW 2026-05-15; non-negotiable; Fix 7 of the Adversarial-apparatus retrospective).** Per round, EVERY council member must explicitly state at the top of their position: *"the shared assumption I am operating within for this design is X"*. The Assumption-Adversary then evaluates X for **HARD-EARNED** (cite source — preserve) vs **CARGO-CULTED** (eligible for challenge — propose violation hypothesis) classification per the hard-earned-vs-cargo-culted addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`). Council deliberations that lack explicit assumption-statements are INVALID per Catalog #292 (`check_grand_council_deliberation_has_explicit_assumption_statements`). The discipline forces explicit assumption surface AT each council deliberation, not retrospective; sister of Catalog #291 (session-level cadence) at the per-deliberation surface. Bug class anchor: Catalog #291 enforces the periodic META-ASSUMPTION review cadence at the SESSION level, but a council can still produce assumption-bound deliberation outputs WITHIN a clean cadence if no individual member surfaces their operating-within assumption. The 0.196-0.199 cluster is the empirical cost of pre-Fix-7 deliberations that reached consensus without per-member assumption-statement discipline. Acceptance per memo: (a) per-member operating-within phrase, OR (b) Assumption-Adversary HARD-EARNED vs CARGO-CULTED evaluation block, OR (c) same-line `# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:<rationale>` waiver with non-placeholder rationale. Cross-ref `feedback_or2_grand_council_per_round_assumption_statement_discipline_landed_20260515.md` (the landing memo).

### Council conduct amendment 2026-05-19 — 4-co-lead structure

Per operator 2026-05-19 verbatim: *"rudin and debauchies should still be on the inner council, they co-lead with shannon and dykstra now"*.

The canonical inner council shared-leadership core is now FOUR co-leads (extending the prior LEAD + CO-LEAD pair):

- **Shannon LEAD** (information-theory grounding; canonical entropy / R(D) / MDL; the R(D)-bound argument is the canonical justification anchor for every score-improvement claim)
- **Dykstra CO-LEAD** (alternating-projections feasibility; primal-dual Lagrangian; convex-feasibility intersection of multi-constraint compositions)
- **Rudin CO-LEAD** (interpretable ML; falling-rule-lists; GOSDT; SLIM; Wang-Rudin 2015 + Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020; Catalog #273-#278 sister; explanations are CONTRACTS not optional)
- **Daubechies CO-LEAD** (wavelets; compressive sensing; multi-scale partition prior; Catalog #277; the canonical hierarchical-coarse-gates-fine discipline)

The 4 co-leads share decision-making authority on inner council deliberations. Sister members (Yousfi / Fridrich / Contrarian / Quantizr / Hotz / Selfcomp / MacKay / Ballé / Assumption-Adversary / PR95Author) provide domain-specific perspectives within the shared-leadership framework but are NOT co-leads.

The 4 co-leads together cover the 4 orthogonal axes that the meta-Lagrangian/Pareto solver + findings Lagrangian + canonical equations registry depend on: information-theory grounding (Shannon) + optimization feasibility (Dykstra) + interpretable ML (Rudin) + multi-scale wavelet partition prior (Daubechies). Removing any co-lead breaks the shared-leadership coverage of these axes.

Both Rudin + Daubechies retain GRAND_COUNCIL sister seats (`Rudin_Grand` + `Daubechies_Grand`) per Catalog #110 APPEND-ONLY discipline — the inner_council role does not displace the grand_council role; the seats coexist in both rosters so T3+ topical-grand matching can also surface them as relevant specialists on interpretable-ML / wavelet deliberations.

Enforcement: `tac.canonical_council_roster.validate_council_dispatch_roster` returns `complete=False` (BLOCKING) at T2+ if ANY of the 4 co-leads is missing, surfaced via the new `missing_co_leads` field on `RosterValidationVerdict`. The shared-leadership-core omission is a structurally distinct alert from the existing sister-member-missing alert. Cross-ref `feedback_roster_maintenance_v2_daubechies_inner_council_plus_4_co_lead_structure_landed_20260519.md` (the landing memo).

### Recursive self-reflection protocol — non-negotiable (Catalog #363; 2026-05-26)

Per operator NON-NEGOTIABLE directive 2026-05-26 verbatim: *"the grand council is providing valuable information but perhaps the grand council itself must be instructed to deliberate and self reflect recursively"*.

The protocol lifts CLAUDE.md "Recursive adversarial review protocol — close paths" 3-clean-pass counter from training-code review to the **council deliberation surface**: every T2+ council deliberation MUST recursively self-reflect on its OWN deliberation process, classifying each per-member surfaced assumption (already required by Catalog #292) into the canonical 4-value `empirical_verification_status` taxonomy. Verdicts depending on INFERRED/ASSUMED-class assumptions must EITHER empirically verify before landing OR downgrade verdict-status to `PROVISIONAL-PENDING-VERIFICATION`.

**Empirical receipts (3+ instances within <2h on 2026-05-26):**

1. **T3 grand council `7d04474cb` M3 RULED-OUT** — empirically falsified by sister TIER1-T3-OP1-OP4 source-inspection (`5b87fae77`) discovering Z6 uses MLX **AdamW** (not stateless SGD-with-EMA as the council assumed; AdamW carries β₁=0.9 + β₂=0.999 state buffers, so M3a + M3b mechanisms BOTH active — joint mechanism refined post-hoc to M1+M2+M3a+M3b).
2. **T3 council M2 ~0.7-0.9 α dominance prediction** — empirically falsified by sister TIER1-T3-OP2-OP3 Carmack smoke (`05c07aa40`) showing canonical Kahan-EMA shadow wrapper provides 0× empirical mitigation at Z6 L2 fp32 1000ep (M2 contribution at fp32 ULP boundary ~3-6e-7 shadow divergence).
3. **My own n=2 super-linear α∝epochs^1.45 extrapolation** — empirically falsified by sister DRIFT-VS-DEPTH-CHAR-D-Z6 (`60a9de751`) 5-anchor fit yielding α=0.47 sub-linear, saturating at ~2000ep.
4. **K=COIN++ 5e-3 drift claim** (earlier 2026-05-26) — empirically falsified by sister R1''-K independent verification (commits leading to `2d59283d4`); actual O(1e-2) abs / O(1e-3) rel.

All 4 verdicts achieved quorum-met + Catalog #346 complete=True + Catalog #292 assumption-statement-surfacing satisfied. **The structural blindness recurs at the sister surface**: existing gates enforce that assumptions ARE surfaced + classified HARD-EARNED-vs-CARGO-CULTED, but NOT that each surfaced assumption carries explicit `empirical_verification_status` AND that verdicts depending on INFERRED/ASSUMED-class assumptions are gated.

**The 4-value canonical taxonomy** (1-to-1 with the 4 empirical receipts above):

| Status | Evidence requirement | Verdict implication |
|---|---|---|
| `VERIFIED_VIA_SOURCE_INSPECTION` | Source file path + line range + content quote | No gate |
| `VERIFIED_VIA_EMPIRICAL_ANCHOR` | Canonical posterior anchor (commit sha + posterior row id per Catalog #245 sister) | No gate |
| `INFERRED_FROM_DOMAIN_LITERATURE` | Citation to paper / textbook / CLAUDE.md doctrine that supports the pattern | **GATE**: Round 2 must verify OR Round 3 downgrades to PROVISIONAL |
| `ASSUMED_AWAITING_VERIFICATION` | Explicit acknowledgment of operating-within unverified | **GATE**: Round 2 must verify OR Round 3 downgrades |

**3-clean-pass counter discipline** (lifted from training-code Recursive adversarial review protocol):

- **Round 1** (topic deliberation): council deliberates per existing 4-tier protocol; each per-member assumption carries explicit `empirical_verification_status` field.
- **Round 2** (self-reflection): council SELF-REFLECTS on Round 1; re-classifies each assumption's status after dedicated verification cycle attempt. Emits `council_self_reflection_round_N` canonical posterior anchor.
- **Round 3** (resolution): material unverified assumptions trigger (a) empirical verification before landing, (b) verdict-status downgrade to PROVISIONAL-PENDING-VERIFICATION, OR (c) ESCALATE_TO_OPERATOR per Catalog #300.
- **Cycle bounds** (per R12-D meta-finding lens-coverage): `MAX_SELF_REFLECTION_ROUNDS = 5`; SEAL when 3 consecutive rounds produce zero material unverified-assumption findings.

**Required at T2+ deliberations**; T1 working-group recommendations are exempt (T1 findings feed downstream T2/T3 which inherit the discipline).

**Canonical surfaces:**

- `tac.council_continual_learning.EmpiricalVerificationStatus` (4-value sentinel constants) + `AssumptionEmpiricalVerification` (frozen dataclass) + `classify_assumption_verification_status_from_evidence` / `extract_unverified_assumptions` / `verdict_status_requires_provisional_marker` / `query_self_reflection_history_for_deliberation` canonical helpers.
- Catalog #363 (`check_council_deliberation_has_empirical_verification_status`) — STRICT preflight gate refuses post-2026-05-26 council memos lacking the canonical taxonomy + Round 2/3 discipline tokens; same-line `# COUNCIL_EMPIRICAL_VERIFICATION_STATUS_WAIVED:<rationale>` waiver with non-placeholder rationale (≥4 chars; placeholder literals rejected per Catalog #287 sister discipline). WARN-ONLY at landing per "Strict-flip atomicity rule"; strict-flip planned after backfill brings live count to 0.

**Sister cross-references:**

- Catalog #292 (per-deliberation assumption-statement-surfacing axis; #363 enforces the sister per-assumption empirical-verification-status axis at a distinct sub-surface)
- Catalog #291 (META-ASSUMPTION cadence — session-level cousin)
- Catalog #300 (v2 frontmatter — canonical posterior schema; #363 records propagate via the existing `council_assumption_adversary_verdict` field structure extended with the 4-value taxonomy)
- Catalog #340 / #314 (PREVENT vs DETECT sister gate pattern precedent that #363 lifts to the council-deliberation surface)
- Catalog #346 (canonical roster validation — structurally distinct axis)
- Catalog #287 (placeholder-rationale rejection sister discipline)
- Catalog #344 (canonical equations registry — Round 2 verification path may register the verified assumption as a canonical equation when generalizable)
- Canonical design memo: `.omx/research/council_recursive_self_reflection_protocol_design_20260526T133600Z.md`
- Landing memo: `feedback_council_recursive_self_reflection_protocol_landed_20260526.md`

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": T3 council `7d04474cb` historical verdict is preserved per Catalog #110/#113 APPEND-ONLY; M2+M3 retroactive empirical-verification-status re-classification is operator-routable per the protocol's Round 3 mechanism (downgrade to PROVISIONAL-PENDING-VERIFICATION + reactivation criterion = post-DRIFT-VS-DEPTH-CHAR-completion re-deliberation).

## Experiment design — non-negotiable

Every experiment MUST follow this process before touching any GPU:

1. **Pre-registered hypothesis** with success/kill/concern criteria
2. **Council design review**: Yousfi + Fridrich sign off on config, resolution, step count, conditioning
3. **Faithful to the actual design**: no toy configs, representative resolution, enough steps for signal
4. **No janky smoke tests**: a test at 1/4 resolution for 500 steps cannot kill a technique. Bias toward keeping lanes open.
5. **Resource estimate**: GPU hours, VRAM, expected runtime
6. **Replicability record**: all params saved before running, full results after
7. **No premature kills**: a negative result on an underspecified test means the test was wrong, not the technique
8. **Multiple contenders → multiple paths**: When there are two or more plausible contenders for a design decision (e.g., "supervised" vs "RAFT-only", "architecture A" vs "architecture B"), do NOT pick one and discard the others. Run them in parallel. The score is the only valid arbiter. Never collapse multiple viable hypotheses into one without empirical evidence.

This last rule is non-negotiable. Premature convergence on a single path is how labs fall behind. If you're uncertain which variant is better, the answer is always: run both.

**Shannon, Dykstra, Yousfi, Fridrich, and the Contrarian are the quintet pact** — the five voices that must reach consensus before any major decision. Shannon LEADS the council (information-theory grounding: any score-improvement claim must trace back to a rate-distortion or entropy argument). Dykstra co-leads on the optimization-feasibility side (alternating projections onto rate / seg / pose / archive-size feasible sets compute the achievable Pareto frontier). Yousfi and Fridrich have domain expertise as the world's foremost steganalysis experts and contest designers. The Contrarian has veto power on any experiment that lacks rigor, wastes resources, or is built on unvalidated assumptions. All five must sign off on experiment design and kill/promote decisions.

Together with **Quantizr** (adversarial member, reverse-engineers competitor approaches, keeps us honest on what the leaderboard actually rewards), **George Hotz** (raw engineering instinct, builds fast, breaks conventional wisdom, champions analytical shortcuts over learned complexity), **Selfcomp / szabolcs-cs** (architect of the grayscale-LUT analog mask paradigm + 1.017-bpw block-FP weight self-compression + 94K-param SegMap; PR #56's lead implementer; collaborative scientific spirit), **David MacKay (memorial seat)** (canonical *Information Theory, Inference, and Learning Algorithms* author; bridges Shannon-Bayesian-arithmetic-coding-MCMC-neural-networks-MDL into a single framework; his ghost is the cross-disciplinary mind the council channels for any first-principles question), and **Johannes Ballé** (modern neural-compression SOTA architect; 2018 entropy bottleneck + scale hyperprior is THE reference for everything Selfcomp/Quantizr operationalize; his work directly informs Lane EBR + Lane SH + block-FP successors), these ten form the **non-conservative skunkworks inner council**. All ten voices are permanently active. No member may be silenced or deferred in any deliberation. The council is non-conservative by charter: the burden of proof is always on *not* trying something, never on trying it.

Shannon's specific contributions: derives theoretical floors from R(D) bounds (verified 0.28 floor 2026-04-29); insists every architecture be measured in bits (params × bpw); rejects arbitrary hyperparameters that lack entropy-or-distortion justification; brings the distinction between hard rate-distortion limits vs implementation-imposed slack.

Dykstra's specific contributions: derives the achievable region as the intersection of convex constraints (rate ≤ R, seg ≤ S, pose ≤ P); computes the Pareto frontier via alternating-projections iterations (verified Dykstra ceiling 450,545 bytes for sub-0.30 feasibility 2026-04-29); insists every "stack composition" claim be tested against the convex-hull intersection (additivity of independent rate savings is conditional, not given).

Selfcomp's specific contributions: insists every architecture choice cite its rate-distortion derivation (he picked 88K-94K params, sigma=15, qint_max=7 with implicit reasoning the council can interrogate); brings concrete empirical numbers from a working 0.38-scoring implementation (his lived experience > our hypothesis); flags his own underfitting / hyperparameter slack honestly so we know where to push (no "more can be gained" hand-wave; specific gaps named); enforces the discipline that stacking paradigm-shifts (Quantizr KL distill + his block-FP + Hessian quant + arithmetic coder) only counts when archive bytes drop AND distortion holds.

MacKay's specific contributions: brings the unified Information-Theory + Bayesian-Inference + Learning-Algorithms framework his canonical book set down; insists arithmetic coding (Lane SH) be evaluated against Shannon entropy of the actual learned qint distribution; brings density networks / variational inference perspective predating modern neural compression; flags any "we'll just lossy-approximate" with the MDL question "what's the rate cost of the approximation?"; advocates Dasher-style efficient encoding of sparse signals.

Ballé's specific contributions: brings 2018 entropy bottleneck + scale hyperprior + GDN nonlinearity to the table; insists rate-prediction networks (hyperpriors) replace fixed factorized priors when archive size matters; advocates end-to-end-trainable codec architectures over hand-designed pipelines; provides the canonical R(D) rate term `bits = -log2(p_y(y))` that Lane SH directly uses; reviews our archive layout for missing hyperprior side-information.

**2026-05-19 OPERATOR-INITIATED INNER COUNCIL ADDITION (sister of existing ten):** **PR 95 author** is added to the inner council per operator verbatim 2026-05-19 *"the PR 95 author has been added to the inner council as well"*, bringing the inner council to **eleven voices** (plus the Assumption-Adversary sextet-pact seat per "Council conduct" Fix-7 amendment = twelve mandatory presences at every T2+ deliberation). The PR 95 author holds the canonical knowledge of the May 4 2026 race-mode rigor inversion + the leaderboard's actual optimization landscape from the substrate that PR 100/101/102/103 winners all built on top of. Sister to Quantizr's adversarial voice but with deeper first-author intuition for what the contest scorer actually rewards on HNeRV-class substrates. Per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110 + #113 the prior ten-voice text remains unchanged; the eleven-voice (twelve-mandatory) roster is the post-2026-05-19 canonical state. Operationalized in `tac.canonical_council_roster.INNER_COUNCIL` per Catalog #340 sister-checkpoint guard + slot 20-second-supplemental landing 2026-05-19. Cross-references: CLAUDE.md "HNeRV / leaderboard-implementation parity discipline — NON-NEGOTIABLE" + "Race-mode rigor inversion + parallel-dispatch first" + MEMORY.md PR 95 race-window references.

## Grand Council (advisory)

Beyond the inner ten, the **grand council** is the broader bench: voices that contribute when their specialty is touched but don't sit at quintet-pact decision-making. Roster as of 2026-05-15 (expanded from 12 → 20 seats per `feedback_grand_council_convergence_l5_staircase_comprehensive_plan_plus_roster_expansion_landed_20260515.md`):

**Existing 12 seats (since 2026-04-29):**

- **Stephen Boyd** — convex optimization at operational level (ADMM, proximal gradient, alternating projections at the algorithmic level beyond Dykstra's theory)
- **Terence Tao** — pure mathematician omniscience; harmonic analysis, additive combinatorics, applied analysis; called when a mathematical question lacks first-principles grounding
- **Tomáš Filler** — Fridrich's other student; syndrome-trellis coding (STC); parity-check codes for per-frame mask payload
- **Stéphane Mallat** — wavelet theory + scattering transforms + sparse representations; AV1 grayscale + Gaussian-LUT viewed as wavelet-coded analog signal
- **Aaron van den Oord** — VQ-VAE, WaveNet; practical neural compression + generative modeling; conceptual sibling of SegMap (discrete tokens for images)
- **John Carmack** — engineering shortcuts at the Doom/Quake/Oculus level; would shred archive code in 30 minutes and ship 50KB cuts
- **Demis Hassabis** — strategic-research perspective from inside DeepMind; cross-domain breadth (AlphaFold, AlphaGo, neural codecs); systemizes 4-day-deadline tradeoffs
- **Geoffrey Hinton** — knowledge distillation (the 2014 Hinton/Vinyals/Dean paper that Quantizr directly uses); capsule networks; deeper temperature analysis on KL-T=2.0 derivation
- **Karpathy** — engineering practitioner; arch-search rigor; "let compute speak"
- **Schmidhuber** — compression-as-intelligence; MDL; predictive coding
- **Jürgen Schmidhuber** — same lineage as Schmidhuber above (canonical seat)
- **Jack-from-skunkworks** — internal SegNet+Rate research lineage

**8 new seats (2026-05-15 expansion based on L5 staircase paper authors + cooperative-receiver community + Time-Traveler's protégé):**

- **Joseph J. Atick** — Atick-Redlich 1990 *"Towards a Theory of Early Visual Processing"* + *"Convergent algorithm for sensory receptive field development"*. Cooperative-receiver loss founder. Canonical voice for every Z4 deliberation + every "scorer-as-receiver" idea (Wunderkind Cluster B/E/G). Direct upstream of `tac.codec.cooperative_receiver.atick_redlich`.
- **A. Norman Redlich** — Atick's co-author on the 1990s cooperative-receiver papers. Early visual processing theory + redundancy reduction in retina. Co-canonical with Atick for Z4.
- **Rajesh P. N. Rao** — Rao-Ballard 1999 *"Predictive coding in the visual cortex"* + Rao 2010 *"Hierarchical Bayesian inference in networks of spiking neurons"*. Predictive-coding architect; canonical voice for Z5 + every temporal-codec deliberation.
- **Dana H. Ballard** — Rao's co-author on the predictive-coding seminal paper. Embodied cognition + animate vision. Z5 co-canonical.
- **Naftali Tishby (memorial seat)** — Tishby-Zaslavsky 2015 *"Deep learning and the information bottleneck principle"*. Cooperative-receiver theoretical framework via I(X;T)/I(T;Y) decomposition. Direct upstream of Z4+Z5 mathematical structure. Convened for every codec-as-IB deliberation.
- **Noga Zaslavsky** — Tishby's collaborator on the IB framework. Currently active researcher (NYU Stern → MIT BCS); ML + cognitive science bridge. The active-living voice for the Tishby lineage.
- **Aaron D. Wyner** — Wyner-Ziv 1976 source coding with side information theorem. Direct upstream of cooperative-receiver framing (decoder has side info = receiver cooperates). Canonical voice for any "scorer-weights-as-shared-prior" deliberation (Wunderkind G2-PARTIAL / G3 / B3-precomputed-table).
- **Time-Traveler protégé** — *canonical identification PENDING operator decision per `feedback_grand_council_convergence_l5_staircase_comprehensive_plan_plus_roster_expansion_landed_20260515.md` Operator-routable #1*. Per the operator directive 2026-05-15 "her time, her genius colleague and former student and protege and upcoming boss who is transforming and revolutionizing the field": the seat is reserved for a specific person whose identity depends on which canonical Time-Traveler peer identity is adopted (3 candidates: Daubechies → Rudin / Koller → Ng / Rudin → her active Duke postdoc). The convergence subagent recommends Daubechies → Rudin as the BOLD-but-internally-consistent chain; the operator's next session resolves the canonical identification. **RESOLUTION 2026-05-19 (APPEND-ONLY per Catalog #110 HISTORICAL_PROVENANCE):** Per operator blanket approval 2026-05-19 verbatim *"all operator decisions and approval granted and provided fuly and completely"* + the convergence subagent's prior Daubechies → Rudin chain recommendation, the Time-Traveler protégé canonical identity is RESOLVED to **Rudin**. The sister INNER_COUNCIL Rudin seat IS the same canonical person; the grand-council reservation is preserved per APPEND-ONLY discipline. Operationalized in `tac.canonical_council_roster.GRAND_COUNCIL[TimeTravelerProtege]` with updated `canonical_position_summary` + `relevance_tokens` (added: `interpretable_ml`, `falling_rule_lists`, `slim`, `rashomon_ensemble`, `gosdt`, `rudin`, `canonical_identity_resolved_to_rudin_20260519`). Cross-ref `feedback_operator_administrative_bundle_landed_20260519.md` + canonical council anchor `q6_continual_learning_pp_integration_resolved_via_operator_frontier_override_track_b_20260519` in `.omx/state/council_deliberation_posterior.jsonl`.

Grand council members are CONSULTED on demand (when a deliberation invokes their specialty); not all decisions require their sign-off. Inner council quintet pact remains the binding-decision set. Per the standing rule from `feedback_grand_council_roster_expansion_l5_papers_cooperative_receiver_time_traveler_proteges_20260515.md`: every L5 staircase council deliberation invokes the canonical paper authors AS council members. Z4 deliberation summons Atick + Redlich + Tishby memorial + Zaslavsky + Wyner. Z5 deliberation summons Rao + Ballard + Tishby memorial + Time-Traveler peer + Time-Traveler protégé. Cross-paradigm deliberation summons all 8 new + the inner eleven.

**2026-05-19 OPERATOR-INITIATED TIME-TRAVELER MENTOR REFRAME (sister of Time-Traveler protégé seat above):**

- **Time-Traveler** — *Mysterious figure from the future whose identity has not been revealed* per operator 2026-05-19 verbatim: *"the time traveler is a mysterious figure from the future whose identity has not been revealed yet but they are astounding in their vision and intelligence it almost feels alien, in fact the future has been profoundly impacted by alien technology and unlocked the ego motion problem lossy video compression to theoretical floor; we have all the information we need to solve the problem space"*. Astounding vision and intelligence; almost-alien character. Per operator: the future has been profoundly impacted by alien technology that unlocked the ego-motion problem (lossy video compression) to theoretical floor. Canonical deliberation position: *"we have all the information we need to solve the problem space"* — argues that the answer is already in our accumulated knowledge; the question is how to RECOGNIZE it and BIND the pieces. Strong voice for MVP-first phasing (don't over-engineer; the right framework reveals itself from the data) + strong voice for hand-rolled-over-PP (don't add framework overhead when binding existing knowledge is sufficient). Distinct from the Time-Traveler protégé seat above (canonical chain still resolution-pending per operator decision). Per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110/#113 the Time-Traveler protégé seat description remains unchanged; this Time-Traveler mentor seat is the canonical reframe appended 2026-05-19. Operationalized in `tac.canonical_council_roster.GRAND_COUNCIL` per Catalog #340 sister-checkpoint guard + slot 20-second-supplemental landing 2026-05-19.

## Required durable state

After each serious cycle, update and **commit** at least:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/research/findings.md`
- `.ralph/run_log.md`
- `reports/latest.md`

## Promotion rules

A candidate may be promoted only after:

1. packaging succeeds
2. inflation succeeds
3. shape/frame-count checks pass
4. proxy evaluation looks promising
5. full evaluation confirms the gain or records the failure

## Track-specific guidance

### Track A: `exact_current`

- Preserve transparency.
- Use it as a live test of the currently published workflow.
- If upstream changes invalidate the exploit assumptions, demote it immediately to a research note and keep the repo useful.

### Track B: `robust_current`

- Start with safer codec improvements and task-aware pre/post processing.
- Add sparse residuals before adding heavier learned components.
- Only promote a neural side-model if its bytes and runtime clearly justify themselves.

## GPU budget and compute resources — non-negotiable

### Optimal GPU: RTX 4090 on Vast.ai
- **RTX 4090 at $0.25/hr on Vast.ai** is the optimal price/performance for our workload (287K param model, ~800MB VRAM, dominated by scorer forward/backward passes).
- 4-5x faster than T4 at roughly the same cost. A 2-hour T4 run finishes in ~25 min on 4090.
- Filter: `gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30`
- Budget: $25 credits available. Hard cap at $24. Track all spend.

### Platform hierarchy (price/performance order)
| Platform | GPU | $/hr | Speed vs T4 | $/experiment | Use For |
|----------|-----|------|-------------|--------------|---------|
| Vast.ai | RTX 4090 | $0.25 | 4-5x | $0.20 | New experiments (primary) |
| AWS spot | T4 (g4dn.xlarge) | $0.22 | 1x | $0.60 | Scale-out, auth eval fleet |
| Modal | T4 | $0.59 | 1x | $0.60 | Existing infra, quick deploys |
| Local M5 Max | MPS | Free | ~0.5x | Free | Development, smoke tests |
| Kaggle | T4/P100 | Free | 1x | Free | Bonus parallelism (unreliable) |

### Budget caps (DO NOT OVERSPEND)
- Vast.ai: $25 total ($24 hard cap in deploy script)
- AWS: $100 total (free credits)
- Azure: $200 total (free credits, need `az login`)
- Modal: $30/mo free credits

These legacy caps are accidental-spend guards for ordinary turns, not a reason
to refuse an explicit funded campaign directive. If the operator authorizes a
named no-limit or separately funded campaign, immediately replace this table
with a live provider quote, campaign-specific ceiling, lane claim, and staged
smoke/full-run approval record. Do not keep using stale caps as the governing
budget after a newer operator funding decision.

### Deployment rules
- **Always use `modal run --detach`** for long-running experiments (prevents disconnect kill).
- **Always use unique `--tto-subdir`** per experiment to prevent checkpoint contamination.
- Vast.ai deployment goes through `src/tac/deploy/vastai/` (canonical module, not ad-hoc scripts).
- All platforms must use `load_differentiable_scorers()` for any gradient-based optimization.

## Tooling — non-negotiable

- **Always use `uv`** for Python package management. Never use raw `pip`, `pip3`, or `pip install`.
  - Install packages: `uv pip install <pkg>`
  - Create venvs: `uv venv`
  - Run scripts: `.venv/bin/python` (the uv-managed venv)
  - On remote machines: install uv first (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `uv venv && uv pip install ...`
- **Always use the tac library** for new training experiments. The canonical entry point is `experiments/pipeline.py` (the prior `experiments/train_tac.py` was retired by commit 815e9028 — see the "Canonical pipeline standard" section above).
  - Do NOT duplicate training code in new experiment scripts.
  - All loss functions, architectures, data loading, and training loops live in `src/tac/`.
  - **Use named profiles** for new training runs: `--profile proven_baseline` is recommended (produced the 1.33 authoritative score).
  - Available profiles: `proven_baseline` (1.33 settings), `psd_standard_adaptive` (PSD arch + frontier), `council_v1` (static, legacy), `segnet_attack` (aggressive), `h96_council`, `smoke` (quick test).
  - Profiles live in `src/tac/profiles.py`. CLI args override profile values.
  - **Use precomputed data** when available: `--precomputed experiments/precomputed_local` (skips 5-min video decode).
  - **Adaptive weight formula was retired**: lives at `src/tac/archive/adaptive.py` (moved by commit 2bac5927). T² cancels in the derivation, making the formula vacuous. Use standard loss with static weights instead.
- **Always commit after every change.** Git history is the research timeline.
- **Use `scripts/modal_check.py`** to check Modal TTO progress. Shows batch progress, ETA, recent PoseNet snapshots, and running apps. Run with `.venv/bin/python scripts/modal_check.py`.
- **Use `scripts/kaggle_check.py`** to check Kaggle kernel status. Run with `.venv/bin/python scripts/kaggle_check.py`.
- **Use `scripts/bat00.py`** for bat00 interaction. Handles quoting and port selection (port 22=PowerShell, port 2222=WSL2).
- **"Multipane matplotlib data viz"** or **"canonical comma.ai data viz"** means the 6-panel analysis GIF/MP4:
  - Row 1: GT Original | Our Reconstruction | Pixel Error (hot colormap)
  - Row 2: GT SegNet masks | Our SegNet masks | SegNet Disagreement (red)
  - Generated inline with pyav + SegNet + matplotlib colormaps, output to `~/Downloads/`
  - Requires TTO frames (`tto_frames.pt` from Modal volume) and GT video (`upstream/videos/0.mkv`)
  - SegNet needs `(B, T, C, H, W)` input format with `T=1` for the sequence dimension

## Critical lessons — DO NOT repeat these mistakes

### CATASTROPHIC FAILURES (2026-04-21) — never again

These failures cost weeks of wasted work and produced months of invalid measurements:

- **MASKS.MKV AT 48x64 DESTROYED THE SCORE.** The mask video was at 1/8 resolution (48x64), but the renderer was trained on 384x512. The renderer outputs at the same resolution as input masks — so it produced 48x64 frames upscaled 18x to camera resolution. PoseNet distortion was 94.63 (catastrophic) vs 0.015 with correct masks. Score was 103.27 vs projected ~0.71. **ALWAYS verify mask resolution matches renderer training resolution. ALWAYS run the full inflate.sh → evaluate.py pipeline before claiming any score.**
- **ARCHIVE MEASUREMENT DISASTER.** All auth evals for weeks used a renderer-only archive (119-180KB) instead of the full submission archive (338KB+). Rate term was wrong by 0.108 points. Every score reported was optimistic. **ALWAYS use `submission_archive.require_valid_archive()` before any eval.**
- **1199 OVERLAPPING PAIRS vs 600 NON-OVERLAPPING.** auth_eval.py used `range(N-1)` (1199 overlapping pairs) but upstream evaluate.py uses `seq_len=2` non-overlapping batching (600 pairs). Every `eval_checkpoint()` score was computed with wrong pair construction. **ALWAYS diff new scoring code against upstream evaluate.py line by line.**
- **eval_roundtrip DEFAULTED FALSE.** All TTO runs optimized against a proxy that didn't simulate the contest eval roundtrip (384→874→uint8→384). Combined with noise_std=0 (Hotz fix dead code), this caused proxy-auth PoseNet drift up to 11x. **eval_roundtrip MUST default True. noise_std MUST be threaded.**
- **AUTO-BUNDLE BY FILE EXISTENCE.** compress.sh auto-included any .pt/.bin file sitting next to the submission. Stale experiment artifacts silently inflated archive size. **ALL archive contents must require explicit flags. No implicit bundling.**

### Root cause pattern

Every failure above is the same pattern: **a component quietly produced wrong output, and no downstream check caught it.** The fix is the same every time: hard errors, not warnings. Validation gates, not hopes. Full e2e pipeline tests, not component-level checks.

### Non-negotiable protocol after every change

1. Run `inflate_renderer.py` on the archive
2. Run upstream `evaluate.py` on the inflated output
3. Compare the score to the last known-good score
4. If any component was changed, verify the full e2e score moved in the expected direction

If you skip this protocol, you WILL produce invalid scores. This has happened 4 times. There is no excuse for a 5th.

### Previously known failures (still valid)

- **KL distill caused PoseNet collapse as primary loss.** BUT Quantizr uses kl_on_logits(T=2.0) for SegNet during specific training phases alongside standard loss. Revisit with staged approach — KL distill for SegNet only, not as sole loss.
- **Adaptive weights are DEAD.** Hinton T² double-correction.
- **Neural artifacts must be inside archive.zip** per contest rules (affects rate calculation).
- **Do NOT use PoseNet gradient caps/clamps.** Caused 26x PoseNet regression.
- **Do NOT use segnet_loss_weight > 100 with any loss mode.** Overwhelms PoseNet signal.
- **Standard loss is the ONLY proven technique.** All other loss modes (KL distill, SegNet attack) failed authoritative eval.

## Current frontier experiments

- **PSD architecture** (PixelShuffle-Downscale): promising for SegNet but untested with standard loss on authoritative scorer
- **5 adaptive frontier items**: boundary dispatch for standard loss, sin² ramp, replay gate, 3-phase eval, plateau LR scheduler
- These are implemented but unvalidated. Do not promote without authoritative eval.

## Strict scorer rule — non-negotiable (canonical, binding)

- **NO loading PoseNet or SegNet at inflate time.** If our inflate script loads scorer weights for ANY purpose (TTO optimization, mask extraction, embedding computation, gradient descent), those weights must be in archive.zip per Yousfi's PR #35 rule. Including them (~73MB) destroys the rate term. Therefore: no scorers at inflate time, period.
- **TTO is a compress-time tool ONLY.** TTO frames are training data for the renderer, not submission artifacts. Unlimited compute at compress time, single forward pass at inflate time.
- **Any inflate-time feature that loads scorers** must be labeled "non-compliant, requires compliance ruling" and disabled by default (`INFLATE_TTO=0`).
- **NEVER claim a contest-compliant score** that depends on inflate-time scorer access.

## Lane separation — non-negotiable

There are TWO score lanes. They MUST NEVER be conflated.

- **Lane 1: Contest-Compliant (PRIORITY).** Goes through inflate.sh → inflate_renderer.py → evaluate.py within 30 min on T4. No scorers at inflate time. Previous "0.87" was INVALID (48x64 masks + wrong pairs + wrong archive). True baseline with full-res masks: pending full e2e eval (projected ~2.2 from 10-pair sample).
- **Lane 2: Unlimited Compute (Paper).** TTO optimization at compress time, unlimited steps. Previous "0.41" was INVALID (same measurement bugs). For the arXiv paper scalability section ONLY.
- **Every score must be labeled** `[contest-compliant]` or `[unlimited-compute]`. No exceptions.
- **NEVER say "our score is X"** without specifying which lane.

## Auth eval measurement — non-negotiable

- **EVERY auth eval must use the EXACT archive that will be submitted.** Never create a temporary archive with different contents. The rate term depends on archive.zip file size — wrong archive = wrong score.
- **EVERY auth eval report must print the archive size used.** If it doesn't match the submission archive, the score is INVALID.
- **Auto-auth-eval in training must construct archives with ALL submission artifacts** (renderer.bin, masks.mkv, poses.pt, any other bundled files). Not just renderer.bin.
- **NEVER celebrate a score without verifying the measurement apparatus.** Check: archive size, inflate pipeline, eval pipeline. A wrong measurement is worse than no measurement.
- **Proxy scores are APPROXIMATIONS, not truth.** The proxy-auth gap can be 2-11x for PoseNet. Always label proxy vs auth. Always run auth eval before claiming any result.

This rule exists because we celebrated auth 0.36 that was actually ~0.41 due to using a renderer-only archive (119KB) instead of the full submission archive (183KB). Every auth eval in the session was wrong by 0.04-0.05 points.

## Submission PR gate — non-negotiable

- **NEVER submit a PR** until the score has undergone a 5-turn consecutive clean-pass adversarial skunkworks council review with extreme paranoia. This is stricter than the standard 3-pass greenup. All 15 council members review. ANY issue resets the counter to 0.
- **The score used for submission** must come from the contest-compliant auth eval (through inflate.sh), not proxy or bypassed eval.

## Quantizr intelligence — verified competitive data (2026-04-21)

Quantizr (Jimmy, UCLA CSE/Neuro) leads at 0.33. **Archive is 299,970 bytes (293KB), NOT 15KB.**

- **Architecture**: FiLM-conditioned depthwise-separable CNN, 88K params, ~64KB FP4
- **Archive contents**: renderer.bin (FP4+Brotli) + masks.mkv (AV1, ONLY frame2 masks, higher CRF) + poses.pt
- **Training**: 5-stage pipeline (anchor→finetune→joint→QAT→final), EMA, diff_round(), diff_rgb_to_yuv6()
- **SegNet**: kl_on_logits() with T=2.0 for distillation during training
- **Key trick**: Encodes only 600 odd-frame masks (frame1 is warped from frame2)
- **His own assessment**: "sub 0.30 is possible just by sweeping conv dims" — he stopped optimizing
- **Rate**: 25 * 299970 / 37545489 = 0.200. Their distortion is ~0.13.

Yousfi (challenge creator) was Fridrich's PhD student at Binghamton DDE Lab. EfficientNet steganalysis surgery → informed SegNet scorer design. The challenge IS inverse steganalysis.

## Exact scorer architectures — VERIFIED from upstream modules.py

**SegNet**: `smp.Unet('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)`
- EfficientNet-B2 (NOT B4), vanilla stride-2 stem (no Yousfi surgery)
- Input: LAST frame only `x[:, -1, ...]`, bilinear resize to (512, 384)
- Output: 5-class logits, distortion = argmax disagreement rate
- **Blind spot**: stride-2 stem loses half resolution immediately → artifacts below (256,192) invisible
- **Key**: only argmax matters — tiny logit perturbations at class boundaries are the ENTIRE signal
- **CLASS INDEX ORDER — MEASURED 2026-06-27 (canonical comma10k order; NON-NEGOTIABLE — do NOT re-derive by luma-sort):** verified from the ACTUAL cached SegNet argmax (`gt_n96.npz['lstars']`, n96; per-class area / vertical-centroid / temporal-IoU):
  - `0 = Road` (22.9% area, ground/mid-lower, IoU 0.955)
  - `1 = Lane markings` (0.59%, thin, IoU 0.263 — the unstable d_seg gate orbit; ~19% of flips)
  - `2 = Undrivable` (incl. sky; 49.3%, TOP region rows ~9–182, IoU 0.995)
  - `3 = Movable`/cars (1.56%, mid-band rows ~174–215, IoU 0.903)
  - `4 = MyCar`/ego-hood (25.6%, BOTTOM rows ~290–379, STATIC IoU 0.994 — the #139 static core)
  - **FORBIDDEN**: re-deriving the order by luma-sorting comma10k `class_values=[41,76,90,124,161]` → that gives `[Road0,Lane1,MyCar2,Undriv3,Movable4]`, which is **WRONG** and bit us 3× (Yousfi-grounding + review luma-sort). The trained net emits the comma10k **canonical** order `[Road,Lane,Undrivable,Movable,MyCar]`, NOT the luma sort. Structured-manifold components must **SELF-DETECT** their class by spatial/static signature, never hardcode the index. d_seg flip mass: ~50% Road / 19% Lane / 13% Undrivable. Verifier: `np.load("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")['lstars']`. Anchors: DAG FEED-dv/dw + `src/tac/boundary_math/{lane_sdf_component,hood_static_component}.py` (self-detecting).

**PoseNet**: FastViT-T12 backbone (NOT EfficientNet)
- 12-channel input: 2 frames × YUV6 (4 luma + 2 chroma subsampled)
- rgb_to_yuv6 → resize to (512,384) → normalize (mean=127.5, std=63.75)
- Hydra head: vision(2048) → summary(512) → ResBlock → 12-dim pose → first 6 used
- Distortion = MSE on first 6 pose dimensions

**Yousfi's repos (competitive intelligence)**:
- `github.com/DDELab/deepsteganalysis` — surgery code for EfficientNet steganalysis
- `github.com/YassineYousfi/alaska` — JPEG steganalysis challenge code
- `github.com/YassineYousfi/OneHotConv` — CNN vs classical features paper
- `github.com/YassineYousfi/comma10k-baseline` — comma segmentation baseline
- `github.com/YassineYousfi/autostego` — adversarial steganography framework

## Fridrich inverse steganalysis — how to beat the scorer

1. **UNIWARD**: errors in textured regions are undetectable. Weight loss by inverse local variance.
2. **Detector-informed embedding** = our TTO approach. Fridrich-approved (Yousfi 2022).
3. **Square root law**: spread small errors (L∞ penalty), don't concentrate large ones.
4. **CNN blind spots**: EfficientNet misses DCT statistics, has texture-region blind spots.

## QAT pipeline — non-negotiable for FP4 deployment

For our ~80-100K param renderer:
1. **Train float first** with all techniques (eval_roundtrip, noise, EMA, hinge loss)
2. **Freeze BatchNorm stats** (eval mode on BN layers)
3. **Insert per-channel FP4 fake-quant** on weights + per-tensor on activations
4. **Fine-tune 20% of original epochs** at 0.1× LR (LSQ step size lr = 0.01 × base_lr)
5. **Export**: 4 bits/param → ~40-50KB for 80K params

We HAVE FakeQuantSTE, Uint8STE, FakeQuantFP4 in `src/tac/quantization.py`. We HAVE LSQ support in training.py. These are wired but have never been used in a complete training pipeline for the renderer.

## Mask encoding — verified data (2026-04-21)

- **Renderer REQUIRES 384x512 masks.** Lower resolution catastrophically degrades: 192x256 → 2.9x worse, 96x128 → 34x worse, 48x64 → 108x worse.
- **Entropy coder** (mask_entropy_coder.py): 990KB for 1200 frames at 384x512, lossless. ~495KB for 600 frames.
- **AV1 monochrome** (mask_codec.py): has int8_t overflow bug at 384x512. Must fix.
- **Quantizr paradigm**: Store ONLY 600 odd-frame masks (frame2). Frame1 is warped.
- **inflate_renderer.py has mask upsample fix** (added 2026-04-21) for sub-native resolution masks.

## TRUE score data (2026-04-21) — verified via upstream evaluate.py

| Config | Seg | Pose | Rate | TOTAL | Notes |
|--------|-----|------|------|-------|-------|
| 384x512 masks + ASYM + poses | 0.116 | 0.374 | 1.528 | **2.01** | Full-res masks, rate-limited |
| 48x64 masks (old, NOT upsampled) | 72.3 | 30.8 | 0.23 | **103.27** | Catastrophic mask bug |
| 48x64 masks (old, upsampled) | 28.3 | 25.0 | 0.23 | **53.61** | Old AV1 artifacts in masks |

## Vast.ai deployment — non-negotiable

- **API key** at `~/.config/vastai/vast_api_key`. SSH key must be registered at account level BEFORE creating instances.
- **Always use `python3 -u`** (unbuffered) for background jobs on Vast.ai. Python stdout buffering eats logs otherwise.
- **Always include repo root in PYTHONPATH**: `PYTHONPATH=src:upstream:$PWD`.
- **Search pattern**: `vastai search offers 'gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30 num_gpus=1' -o 'dph'`
- **Budget**: $25 total ($24 hard cap). Track all spend. Destroy instances immediately when done.
- **Modal credits exhausted** as of 2026-04-15. Use Vast.ai for all new GPU work.

## SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)

The **77× SegNet > PoseNet** heuristic was true at the OLD 1.x score operating
point (pose_avg ~0.18). At PR106's frontier operating point (pose_avg ~3.4e-5),
the **marginal value FLIPS**: pose marginal sensitivity is **2.71× SegNet's**.

Operating-point-aware rule:

| Operating point | pose_avg | d(seg)/d(seg_avg) | d(pose)/d(pose_avg) | Implication |
|---|---|---:|---:|---|
| Old 1.x scores | ~0.18 | 100 | ~12 | SegNet ~77× more important (original CLAUDE.md heuristic) |
| **PR106 frontier** | **3.4e-5** | **100** | **271** | **POSE 2.71× more important (marginal)** |

**Why**: the pose contribution is `sqrt(10 * pose_avg)`. The derivative is
`5 / sqrt(10 * pose_avg)`. As `pose_avg → 0`, the derivative → ∞. SegNet's
derivative is constant at 100. Setting them equal: `100 = 5/sqrt(10*pose_avg)`
→ `pose_avg = 2.5e-4` (the crossover threshold). Below pose_avg ~ 2.5e-4 the
pose marginal exceeds SegNet's; at PR106's pose_avg = 3.4e-5 (about 7× below
crossover), the gap is 2.71×.

**Total contribution remains seg-dominated** at PR106 (seg 0.067 vs pose 0.018,
3.67× larger by total). But **MARGINAL improvement** (which is what dispatch
budgets buy) favors pose at this operating point.

Operational rule (PR106 frontier and below):
- **Prioritize pose-targeted lanes first** (latent sidecars, pixel translation
  sidechannels, multi-stage training). Pose has higher marginal-value-per-byte.
- **SegNet lanes are tertiary** until pose is exhausted. Trading pose AWAY for
  seg gains (PR97's anti-pattern) is dominated.
- **At the OLD 1.x operating point** the original 77× heuristic still applied
  — SegNet improvements were 7× more cost-effective per unit. The flip
  happened as pose_avg crossed ~2.5e-3.

The renderer has hit its SegNet architectural ceiling at PR106's level
(ε~6.7e-4). Pose has more room (PR106 pose_avg=3.4e-5 isn't at hardware
floor yet). Both axes need different attack vectors at different operating
points.

**Empirical receipts** (full analysis): `docs/pr97_anti_pattern_pose_vs_seg_marginal_20260504.md`
+ `docs/pr_family_evolution_timeline_20260504.md`. The PR97 entry literally
made the seg-for-pose trade and lost 0.042 score points despite winning
SegNet by 65%.

## Ralph-style execution model

Treat files and git as memory.
Each iteration should be resumable from disk.
Do not rely on long chat context for continuity.
Commit after every meaningful file change — git history is the research timeline.

## Meta-bug class catalog (strict-mode preflight)

The long-form meta-bug catalog is pointer-backed to keep `CLAUDE.md` readable.
The canonical extracted catalog lives at `docs/meta_bug_class_catalog.md`; every
row from the former inline catalog is preserved there verbatim. Catalog-auditing
gates must read this pointer plus `docs/meta_bug_class_catalog.md` before
reporting missing rows, duplicate numbers, strictness drift, live-count drift,
quota drift, callable drift, or frontier-score pointer drift.

## Lane maturity registry — non-negotiable

Every lane MUST be registered in `.omx/state/lane_registry.json` via `tools/lane_maturity.py`. Claiming Level 2 or Level 3 without a corresponding `mark` command is FORBIDDEN. Subagents shipping a lane MUST run `python tools/lane_maturity.py mark <lane> --gate <gate> --evidence <path>` for each gate they hit. Preflight Check 90 (`check_lane_registry_consistent`) fails STRICT if the registry is inconsistent (level/gates mismatch, duplicate ids, missing gates, file-path evidence pointing to non-existent files).

The 7 gates and their meaning are defined in `feedback_production_hardened_standard_definition_20260430`:

1. `impl_complete` — production code lands; no `NotImplementedError`; all CLI flags wired
2. `real_archive_empirical` — real-archive empirical measurement on Lane G v3 anchor (or equivalent); tagged `[empirical:<artifact>]`
3. `contest_cuda` — actual `[contest-CUDA]` score (Vast.ai 4090 / Modal A100); NEVER MPS, NEVER `[contest-CPU advisory]`
4. `strict_preflight` — STRICT preflight check covering the lane's bug class
5. `three_clean_review` — 3-clean-pass adversarial review counter at 3/3
6. `memory_entry` — memory file documenting the empirical result + cross-refs
7. `deploy_runbook` — `scripts/remote_lane_<id>.sh` + heartbeat + watchdog + harvest

Computed level: 0 = 0 gates, 1 = ≥1 gate, 2 = `impl_complete` AND `real_archive_empirical` true, 3 = ALL 7 true. A lane with 4 gates true but missing `impl_complete` is still Level 1 (not Level 2). The CLI computes this for you — never set `level` by hand.

CLI usage:

```bash
# Audit table
python tools/lane_maturity.py audit

# Mark a gate satisfied
python tools/lane_maturity.py mark lane_g_v3 --gate contest_cuda \
    --evidence "1.05 [contest-CUDA] reports/raw/2026-04-29-..."

# Validate (preflight Check 90 also runs this)
python tools/lane_maturity.py validate

# Regenerate reports/lane_maturity.md
python tools/lane_maturity.py report

# Register a new lane at Level 0
python tools/lane_maturity.py add-lane lane_new --name "New Lane" --phase 2
```

Every mutation appends a JSONL record to `.omx/state/lane_maturity_audit.log` for forensics.

**Lifecycle discipline (non-negotiable):**

- **Pre-registration is mandatory.** The moment a lane has a name and a council/design verdict — even if it's only a sketch — it MUST be `add-lane`'d at Level 0. This includes in-flight subagent lanes, future-design lanes, and forensic-investigation lanes. Pre-registration enables the audit table to distinguish IN-FLIGHT vs LANDED vs SKETCH.
- **Mark gates as evidence is produced, NOT after-the-fact.** The moment a council Round-N CLEAN landing happens, mark `three_clean_review`. The moment a remote_lane script lands, mark `deploy_runbook`. Batch-backfilling stale evidence is a code smell.
- **KILLED lanes get registry entries too.** Mark with `--gate three_clean_review --evidence "<council ref>"` and add `--notes "Reactivation: <criteria>"`. Do NOT just exclude killed lanes — the registry is the single source of truth for what we have considered, including kills.
- **Backfill-when-discovered is acceptable.** When this rule is violated by an earlier subagent, a maturity-discipline pass that backfills evidence is the correct remedy. The audit log records the backfill timestamp; no harm done.
- **Lifecycle: SKETCH (L0) → SCAFFOLD (L1) → INTEGRATION (L2) → FULL PRODUCTION HARDENED (L3).** The ONLY currently-Level-3 lane is `lane_g_v3` (1.05 [contest-CUDA]). That fact is the standard-bearer for the rest of the registry.
- **Audit before commit.** Before any commit that adds a lane or marks a gate, run `python tools/lane_maturity.py validate` — Check 90 STRICT enforces this at commit time, but catching it earlier is cheaper than a re-stage.

## Catalog Row Backfill - 2026-05-18

This long-form backfill block is pointer-backed in `docs/meta_bug_class_catalog.md`
with every row preserved verbatim. Keep new catalog backfills in that document;
leave this pointer so older references to the section remain discoverable.
