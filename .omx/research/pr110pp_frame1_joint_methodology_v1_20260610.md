# pr110pp_frame1_joint_methodology.v1 — the missing frame-1 methodology (operator-bound design)

UTC 2026-06-10 · claude · design artifact per operator directive (verbatim: "PR110++ is not optimal
yet. Frame-1 methodology is still missing."). Status: DESIGN → build dispatches when an agent slot
frees (task #50). `[macOS-CPU advisory]` design; every emitted row is exact-authority-gated.

## Why this exists (the honest assessment)

PR110/FEC6 was a first-generation FRAME-0 pose-only selector: 31 candidates, K=16 global palette,
~3.24 bits/pair Huffman, exploiting ΔS_seg ≡ 0 for frame-0 actions (SegNet reads only frame-1).
Frame-1 was never solved — it was swept symmetrically and lost every slot (public PR110 record).
R1 (2026-06-09) further proved the chooser substrate was fragile: off-host ordering does not
transfer at the 1e-5 pose scale. The public PR110 notes themselves name the missing work: distinct
frame-1 methodology, per-tier/per-region menus, EV-weighted selection, both-frame composites,
PoseNet-null subsets, SegNet-class-region waterfilling, time-coherent coding, gradient probes.

Frame-0 was easy: ΔS(a) reduces to the pose term vs selector bytes. Frame-1 is the constrained
JOINT problem: Δd_seg ≠ 0 AND Δd_pose ≠ 0 — improve either without crossing SegNet class walls,
while paying action+selector bytes. That requires the cone/margin geometry we now have.

## The measured artifacts this methodology consumes (all on disk, sha-cited at build time)

- **#35 frame1 joint-safe cone** (`frame1_joint_safe_cone.py` + 600-pair cone maps): per-pixel
  seg-margin budget ∧ pose-Jacobian; usable 48.6% / fragile 51.4% / pose-binds 73%.
- **#36 response atlas** (600-pair index): per-pair budgets, fragile clusters (protect 510-522,
  133/177-178), spend-first clusters (426-442, 577-579).
- **#47 invisibility basis**: tier-1a 22.7% zero-weight pixels; tier-1b 80.67% resize nullity;
  the S12 preimage theorem.
- **R2/R3 mode tables**: the exact-CPU per-mode pose table (R2) superseded by the ON-HOST table
  with noise floor (R3, in flight) — the only admissible ranking substrate.

## The five action classes

### Class 1 — frame-0 pose-only menu (the matured PR110 lane)
On-host exact table ONLY (R3); **noise-floor tie law: |Δ| ≤ measured floor → keep incumbent**.
Per-pair, never global-argmin-on-foreign-host. This class is already in execution (R3).

### Class 2 — frame-1 Seg-SAFE pose actions
Frame-1 perturbations helping PoseNet while staying inside the source SegNet chamber.
Admission: per-pixel margin survival m_p + ∇m_p·δ > τ on protected pixels — operationally,
support masks restricted to the OPEN cone (joint_cone_radius ≥ threshold); fragile 51.4% excluded
by construction. Direction prior: the pose-Jacobian rows (move along measured pose-sensitive,
seg-flat directions). Exact d_seg must come back UNCHANGED (the falsifiable per-action check) —
any seg movement disqualifies the action class instance.

### Class 3 — frame-1 Seg-POSITIVE correction atoms (repair, not tricks)
Direct d_seg reduction: class-boundary luma shifts, thin-class support nudges, margin-normal color
corrections, local palette/boundary-tube/class-region fills. GENERATED FROM the margin field +
fragile masks + thin-class supports (never global 8×8 tiles). Admission by THE LAW:
100·Δd_seg + Δ√(10·d_pose) + 25·Δbytes/N < 0. This is also SNeRV's nonrate lever (its seg term
0.2468 needs exactly these atoms).

### Class 4 — both-frame COMPOSITE actions (the most important missing class)
atom = {frame1_seg_correction, frame0_pose_compensation, support_mask, selector_cost}. The frame-1
move buys seg; the frame-0 move repays its pose cost at zero seg price (the asymmetry used
CONSTRUCTIVELY). Per-composite the COMMUTATOR is measured, never assumed:
comm(a,b) = ΔS(a∘b) − ΔS(a) − ΔS(b); composites admitted on their measured joint ΔS.

### Class 5 — resize-null preimage postprocessing (S12, universal)
Design actions in the SCORER-VISIBLE projection space (384×512), then emit the cheapest legal
uint8 camera-res preimage (Rx̃ = y proven per frame, RGB-before-YUV). Applies to every class above
AND to every other vehicle's frames. 80.67% of camera DOF never reach the evaluator — actions
designed in raw pixel space waste most of their bytes by construction.

## Selection (replaces global-K=16 + pair-count-maximization)

- **Menus, not a menu**: per-tier/per-region/per-cluster menus — frame0-only, frame1-joint,
  composite; hard-pair vs easy-pair (atlas clusters); fragile-region menus carry ONLY Class-3 atoms.
- **EV-weighted selection**: mode m with support P_m wins by
  EV(m) = −Σ_{i∈P_m} ΔS_i(m) − 25·codebytes(m)/N — value per encoded bit, never pair-count.
- **Time-coherent selector coding**: selector streams delta/run-length coded over pairs (the atlas
  shows temporal clustering — selector entropy should exploit it).
- **Code length inside ΔS**: every mode's admission includes its own selector/codebook bytes.

## The row schema (every mode/action/composite emits)

{pair_id, target_frame, support_or_cone_id, d_seg_delta, d_pose_delta, score_delta, selector_bits,
 authority_host, noise_floor, measured_commutator (class 4), preimage_proof (class 5: max|Rx̃−Rx|),
 accepted_or_rejected_reason, authority_tier, metric_family, base_archive_sha256}

No row without authority_host; no ranking off-host (the R1 lesson, structural).

## Compute-substrate law (operator correction 2026-06-10 — supersedes any "MPS can generate" framing)

**MPS: NEVER — not for ranking, not for generation, not for anything.** (The earlier "local MPS can
generate candidates" framing is operator-retracted.) The three-tier substrate law:
1. **GENERATION + search + gradients: MLX** — we have MLX ports of everything possible (the MLX
   SegNet/PoseNet pair teachers via `build_mlx_{segnet,posenet}_pair_teacher`, the differentiable
   YUV6 path, the canonical_kernels Backend dispatch, the mlx_score_aware harness). Candidate
   atoms, cone-conditioned direction search, margin-field gradients, composite enumeration — all
   MLX on the M5 Max 128GB unified memory, saturated freely. Tag `[macOS-MLX research-signal]`.
2. **ADVISORY verification: local CPU-torch exact frozen scorers** (the #35/#36 pattern) — cheap
   exact d_seg/d_pose screening of MLX-generated candidates before any paid dispatch. Tag
   `[macOS-CPU advisory]`.
3. **RANKING + admission: contest host ONLY** (the R1 lesson) — micro-pose orderings and final
   accept/reject come from on-host exact tables with noise floors. Tag `[contest-CPU]`.
MLX-first → numpy reference → torch parity per the standing doctrine; any artifact with an MPS
ancestor is contamination requiring rebuild.

## Build routing (task #50)

Dispatch when a slot frees. Sequencing: Class 2+3 generators first (they consume only landed
artifacts: cone + margin field + atlas), Class 4 composer second (needs Class 2/3 instances),
Class 5 integrates the #49 preimage compiler when it lands. Branch decision per operator: if R3
positive → harden corrected selector first, then this; if R3 flat → the frame-0 catalog is
exhausted on-host and THIS becomes the primary PR110++ lane immediately.

## Cross-refs
`pr110pp_r1_paired_eval_verdict_20260609.md` (the substrate falsification) ·
`frame1_joint_safe_cone_landed_20260609.md` (#35) · `evaluator_invisibility_basis_landed_20260610.md`
(#47/S12) · `evaluator_response_atlas_engine_landed_20260609.md` (#36) ·
`snerv_rate_attack_round2_directive_20260610.md` (S12 + the nonrate reality check this serves).
