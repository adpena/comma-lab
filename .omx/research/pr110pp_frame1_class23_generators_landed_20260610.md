# pr110pp_frame1_class23_generators — Class-2 + Class-3 atom generators LANDED

UTC 2026-06-10 · claude (#50 executor, phase 1) · `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`,
non-promotable per Catalog #192/#341/#127/#323. $0 local, NO cloud, NO paid GPU, NO MPS, NO /tmp.
Binding design: `.omx/research/pr110pp_frame1_joint_methodology_v1_20260610.md` (followed exactly).

## What landed (the lab's first true frame-1 action machinery)

Frame-0 was easy (SegNet reads only frame-1, so a frame-0 perturbation has `Δd_seg ≡ 0` and reduces to
pose vs selector bytes — the matured PR110 lane). **Frame-1 is the constrained JOINT problem** (`Δd_seg ≠ 0`
AND `Δd_pose ≠ 0`). This landing builds the two frame-1 ATOM GENERATORS the design names:

1. **Class-2 — Seg-SAFE pose atoms** (`src/tac/optimization/frame1_seg_safe_pose_atoms.py`, 649 LOC,
   commit `64347d69c`). Frame-1 perturbations that help PoseNet while staying inside the source SegNet
   chamber. Support restricted to the OPEN cone (`joint_cone_radius >= open_cone_threshold`) — the fragile
   51.4% is excluded BY CONSTRUCTION. Direction = the measured pose-Jacobian, seg-flat
   (`leverage = pose_jacobian_norm * seg_margin`). Amplitude = a fraction of each pixel's OWN cone radius
   (certified seg-safe). **The falsifiable per-atom check**: exact CPU-torch `d_seg` must return UNCHANGED
   (argmax-identical => `d_seg == 0`); any seg movement disqualifies. `d_pose` advisory delta recorded;
   atom accepted iff seg-unchanged AND pose-improved.

2. **Class-3 — Seg-POSITIVE repair atoms** (`src/tac/optimization/frame1_seg_repair_atoms.py`, 686 LOC,
   commit `9d9f2a027`). Targeted color/luma corrections that push the rendered frame-1's SegNet argmax BACK
   toward the GT class at small-but-recoverable margin pixels. GENERATED FROM the margin field + fragile
   masks + thin-class supports (boundary-tube ∪ thin-class ∪ fragile region; NEVER global 8×8 tiles).
   Margin-normal correction toward the GT appearance gap, ranked by recoverable margin. Admission by **THE
   LAW**: `100·Δd_seg + Δ√(10·d_pose) + 25·Δbytes/N < 0`. **Vehicle-agnostic** interface (operates on
   `(rendered_frame, gt, margin_field, fragile_mask)`, carries NO carrier state) — this is also SNeRV's
   nonrate lever (its seg term 0.2468 needs exactly these atoms).

3. **The $0 headline run** (`tools/run_frame1_class23_atom_headline.py`) — generates atoms for a stratified
   sample (budget clusters 426-442/577-579 + fragile clusters 510-522/133/177-178 + an even spread),
   screens every atom on the exact CPU-torch DistortionNet, applies the #49 tier-1 preimage postprocess,
   emits the design row schema.

4. **24 behavioral tests** (`src/tac/tests/test_frame1_class23_atoms.py`) — cone-constraint enforced by
   construction; seg-unchanged check fails closed; fragile exclusion; the no-fake atom-actually-perturbs
   check; vehicle-agnostic apply; THE LAW admission on real scorers; MLX/numpy leverage parity. All pass.

## Compute-substrate law (operator correction 2026-06-10, encoded throughout)

- **GENERATION + leverage search: MLX** — `seg_safe_pose_leverage_mlx` / `repair_leverage_mlx` run on the
  M5 Max unified memory; numpy reference is the canonical portability oracle (Catalog #383 Backend
  pattern). MLX/numpy agree to fp32 tol (`< 1e-3`; measured `~9e-7`). Tag `[macOS-MLX research-signal]`.
- **ADVISORY screening: local CPU-torch exact frozen scorers** — the per-atom d_seg/d_pose check.
  `[macOS-CPU advisory]`.
- **RANKING + admission: contest host ONLY** (the R1 lesson). Every row carries `authority_host=
  macos_cpu_advisory`; the run NEVER claims an on-host accept. It emits the host-ranking candidate set.
- **MPS: NEVER** — not for ranking, not for generation, not for anything.

## Honest disclosure: Class-3 render proxy (NO FAKE)

Class-3 repairs a VEHICLE's rendered frame-1 vs GT. The headline has no vehicle render yet, so it uses a
controlled lossy roundtrip of GT (2× down/up-sample) as a RENDER PROXY that produces REAL SegNet argmax
disagreement, tagged `render_proxy=degraded_gt_roundtrip` per row. The scorers, GT, and d_seg/d_pose
measurements are all REAL — only the baseline render is a stand-in (not a synthetic-fixture-instead-of-real
violation; a future agent wires a real vehicle render via the same vehicle-agnostic interface).

## Headline empirical results `[macOS-CPU advisory]` (26-pair stratified sample, real scorers)

Output: `/Volumes/VertigoDataTier/pact/frame1_class23_atom_headline_20260610T023947Z/`
(rows + summary + manifest, sha-cited, deterministically rebuildable). 26 pairs sampled = the budget
clusters {426,437-442,577-579} + fragile clusters {133,177,178,510,514-519,522} + even spread {0,100,200,
300,400,500}.

### Class-2 (Seg-SAFE pose): **0 / 52 accepted** — a REAL, falsifiable finding (not a failure)

| reason | count | meaning |
|---|---|---|
| `seg_argmax_moved` | 48 | the exact d_seg moved (1-6 argmax pixels flipped, Δd_seg ~1-3e-5) |
| `pose_not_improved` | 4 | d_seg was EXACTLY 0 but pose did not improve at that support |

**The signal**: even at amplitude = 0.5× the *certified* cone radius, perturbing ALL open-cone support
pixels simultaneously flips a few SegNet argmax pixels. The #35 cone guarantees the *linearized* margin
survives a single-pixel perturbation; it does NOT guarantee the nonlinear EfficientNet argmax survives a
*joint* half-radius perturbation over thousands of pixels. The 4 seg-EXACT atoms (`pose_not_improved`)
show the complementary truth: at amplitudes small enough to be argmax-exact, the pose improvement is below
the local-CPU noise floor. **This is exactly the kind of measured constraint the design wants** — it
routes the Class-2 next step (sparser support + much smaller amplitude fraction + carry the per-pixel
pose-gradient sign so the step is a true pose-descent, then re-screen) and it is the empirical reason an
on-host R3-style run with the noise-floor tie law is the admission authority, not this advisory screen.

### Class-3 (Seg-POSITIVE repair): **14 / 26 accepted** (net-negative LAW ΔS)

| reason | count |
|---|---|
| accepted (net-negative LAW ΔS, seg reduced) | 14 |
| `law_net_nonnegative` (seg reduced but pose+bytes cost outweighs) | 11 |
| `seg_not_reduced` (repair did not reduce d_seg) | 1 |

- **value/byte (accepted)**: median 7.74e-5, p90 3.33e-4, max 3.86e-4 `[macOS-CPU advisory]`.
- **seg reduction Δd_seg (accepted)**: median -1.14e-4, max -3.20e-4 (the dominant gain — the seg term
  is 100·Δd_seg).
- **The repair lands where the design predicted**: fragile clusters accept **6/10**, budget clusters
  **4/10**. The two best atoms are pairs **133 and 177** — exactly the design's named fragile clusters to
  PROTECT/repair (dS = -0.114 / -0.105; support entirely boundary-tube ∪ fragile, with thin-class overlap
  109/56 px). The binding-constraint set IS the repair target, confirming the cone/atlas geometry.
- **#49 preimage proof carries correctly**: all Class-3 `preimage_max_abs_residual == 0.0` (certified zero
  scorer change). Bytes-freed = 0 because the atoms are screened at scorer res (384×512) where the
  camera-res zero-weight lattice does not apply — the proof reports the certified-exact path with no false
  byte claim. (At a real vehicle's camera-res emit, the tier-1 fill frees the certified ~22.7% lattice.)

### Contest-host ranking packet readiness (what an R3-style on-host run would admit)

The accepted Class-3 atoms (14 candidates, each with `support_or_cone_id` + `selector_bits_est` +
`value_per_byte`) ARE the host-ranking candidate set: an R3-style on-host run replays each candidate's
repaired pair through the exact contest DistortionNet with the measured noise floor and admits those whose
on-host ΔS clears the floor (the off-host advisory ordering is a prior, not the verdict, per R1). The
Class-2 set currently yields ZERO host candidates — the advisory screen correctly REFUSES to forward
seg-moving atoms, so no off-host Class-2 row would waste an on-host eval. The packet is therefore
Class-3-only at this round; Class-2 re-enters after the sparser/smaller-amplitude + pose-sign-carrying
revision the headline empirically motivates.

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map**: ACTIVE — the seg-safe pose leverage (`pose_jacobian_norm * seg_margin`) and repair
   leverage (recoverable margin over boundary/thin/fragile) ARE per-pixel sensitivity contributions.
2. **Pareto constraint**: ACTIVE — THE LAW admission (`100·Δd_seg + Δ√(10·d_pose) + 25·Δbytes/N < 0`) is
   the per-atom Pareto-feasibility test on the seg/pose/rate axes.
3. **Bit-allocator hook**: ACTIVE — `value_per_byte` (advisory ΔS per encoded byte) + `selector_bits_est`
   per atom feed the EV-weighted menu selection (not pair-count).
4. **Cathedral autopilot dispatch**: N/A — these are advisory generators, not archive-deployable; the
   host-ranking packet is the dispatch surface (deferred to the on-host R3-style run).
5. **Continual-learning posterior**: N/A — no contest-CUDA anchor (advisory only); rows are the side
   information the on-host run reseeds from.
6. **Probe-disambiguator**: ACTIVE — the +sign/-sign Class-2 atoms (`generate_signed_atoms`) + the
   exact-scorer screen IS the disambiguation of the pose-improving direction.

## Cross-references

`pr110pp_frame1_joint_methodology_v1_20260610.md` (design) · `frame1_joint_safe_cone_landed_20260609.md`
(#35) · `evaluator_response_atlas_engine_landed_20260609.md` (#36) ·
`resize_null_preimage_compiler_landed_20260610.md` (#49) · `pr110pp_r1_paired_eval_verdict_20260609.md`
(the substrate falsification + the off-host-ranking lesson) ·
`snerv_rate_attack_round2_directive_20260610.md` (the SNeRV nonrate reality check Class-3 serves).
