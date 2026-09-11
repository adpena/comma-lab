> **RETIRED BEFORE SPAWN (MAIN, 2026-09-11 ~17:00Z) — successor check failed.** pc3's memo §4 shows the continuous-coefficient
> optimum is a BOUND on the whole lattice family (payable 495.6 B at n=135; the global rungs cost more than it is worth), and
> coefficient/rank refit as a family was already CLOSED by ra3/rr1/es1 (a re-open needs a successor check — memory
> `a_reopen_must_check_whether_the_successor_already_fired_20260829`). The only open, built, cheap piece is pc3's own owed unit:
> finish the ceiling at n600 and run `mode=project` for the per-dimension rungs. That unit was assigned to pc3 by resume; no pc4 arm.
> The charter below is kept as the record of what was NOT launched and why.

# ddm_pc4 — price the pose-carrier CONTINUOUS-COEFFICIENT re-solve on move 45 at n600, under the pose re-solve law (charter, MAIN 2026-09-11; operator full-authority GO)

## Where this door comes from (MEASURED by pc3, provisional at n=160/600)
pc3's curve on move 44 (`.omx/research/ddm_pc3_*20260911.md`; retained `/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/{ceiling,rate,refit}/`)
closed every CAPACITY rung by arithmetic (dim ÷2 … global ÷8, rank ×2) and landed the RATE cut as move 45 (predictor refit at
bit-identical codes, −160 B, commit 01f2b66ad). One bound stayed open: **continuous coefficients — payable 453.8 B (95 % hi 632.6 B)
at n=160/600, drifting up (1.15e-7 → 4.36e-7)** — the amount the carrier's stored coefficient/lattice point is over-paying relative to
a re-solved point in the basin of the shipped lattice, at ZERO archive-byte change in the schema. It is the only priced door left on
the carrier and it is a DISTORTION-side re-solve, so the pose re-solve law binds (operator 2026-09-10: resolved pose only; frame 0 is
repair inside the admission; memory `pose_resolve_is_mandatory_after_every_field_change_20260910`; `pose_base_must_be_measured_on_the_
pointers_own_configuration_on_the_arms_instrument_20260909`).

## Pointer (binding; re-derive nothing from memory)
Move 45: S 0.1371383667388406 @ 180,246 B [contest-CUDA T4 n600]; archive 145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a;
runtime tree 2d4dd13352b50a27f3d1b423261665c0f90e250781f07c2fefe86c46656dc967; promoted tree
`/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/candidate/candidate_runtime` (SEALED — read-only; copy). Components
d_seg 0.00010345, d_pose 4.59e-6 (T4 print; pc3's cold-decode pose base 4.58676e-6 on its instrument). Admit bar −2e-5 vs 180,246 B.
Pose budget: any rung must keep 100·Δd_seg + Δsqrt(10·d_pose) + 25·ΔB/37,545,489 < −2e-5; at d_pose 4.59e-6 the pose term's slope is
5/sqrt(10·d_pose) ≈ 738 S per unit d_pose — quote the sqrt EXACTLY at every rung (rbf1's linearisation overstated 7×).

## Deliverable (Opus; $0 until a seal; MAIN fires)
1. Finish pc3's ceiling at n600 (its 9 resumable shards; command in pc3's memo §9) on MOVE 45's carrier — the bound is provisional
   at n=160 and the charter's first number is the n600 value of the payable coefficient slack (B_payable, with its interval).
2. Re-solve the carrier's coefficients/lattice point on move 45 at n600 through the REAL receiver (`decode_cap1`) — the shipped
   schema, shipped rank/precision/width (receiver changes are OUT of scope: they need the first-measurement chain), the resolved pose
   read on the pointer's own configuration on your instrument; frame 0 handled as the admission's repair, never a separate lever.
   Encode with the REAL coder, twin encodes, exact bytes. Report per rung: ΔB (exact), d_pose (through R, n600), d_seg (must be
   UNCHANGED — the carrier does not touch the token plane; prove it by argmax identity), ΔS exact.
3. If a rung nets < −2e-5 at exact bytes: cold n600 public parse-back (raw identity is NOT expected here — the pose carrier changes
   the rendered frames; instead retain the raw and its sha, and the advisory frozen-CPU verdict on all 600), regenerate MANIFEST
   from outside the tree (pr18 validates it independently now), smokes, normal seal with `--inherit-decode-wall-clock` from move 45's
   leg (custody `/Volumes/APDataStore/pact/ddm_pc3_cap1_predictor_refit_move44_contest_cuda_20260911/custody_pointer45`) — receiver
   unchanged ⇒ behavior digest equal; STOP; MAIN fires. If nothing nets: the closure memo with the n600 bound and why the payable slack
   does not convert (which term ate it).
4. Memo `.omx/research/ddm_pc4_pose_carrier_coefficient_resolve_on_move45_20260911.md` (`# FORMALIZATION_PENDING:<rationale>` or an
   equation cite); checkpoint as `ddm_pc4`; serializer commits, two review passes per .py, no co-author trailer, tags `[no-triality] [p0-ledger-ok]`.

## Boundaries
No Modal, no contest eval, no receiver change, no candidate CLAIM (advisory rows `[macOS-CPU advisory]`, score_claim=false); never edit
`upstream/`, the PR tree, sealed trees, or other arms' directories (ddm_ntb2_*, ddm_sr5, ddm_pc3_pose_carrier_curve is pc3's — COPY what
you need into `/Volumes/APDataStore/pact/ddm_pc4/` (Vertigo is at its reserve; route bulk to APDataStore and certify-move nothing);
heavy steps through `tools/launch_detached_process.py --output-dir … --nice 0 --done-receipt … -- <cmd>`; keep every payload with sha.

## OPTIMAL FORM
- Reference form: pc3's landed curve (real coder, real receiver, n600 parse-back, byte-exact positive controls, six pre-registered
  falsifiers) and rp1 r2's pose-driven move 42 (the last carrier re-solve that moved the pointer, −1.62e-4). Every delta vs pc3 is
  SCOPE (one rung family: coefficients in the shipped schema) — no mechanism reduction; n600 or it is not evidence.
- Provenance pins: move 45 (above); pc3 memo + retained dirs (record shas); `src/tac/...` receiver modules pc3 pinned; pr18 commits
  5d2632ee4 + f1b9a0dbb (manifest validation now independent).

## Prior negatives accounted (operator 2026-08-15)
- pc2: carrier bytes buy a LATTICE POINT — rank cut free in span, 4–2,733× on the lattice; you are re-solving the point, not the span.
- mc1: motion-compensated previous plane CLOSED at formulation scope — do not re-open.
- ra3: carrier CLOSED as a byte lever (rate side) — this charter is the distortion-side slack pc3 measured, not ra3's question.
- rp1 r2 flag-vs-constant silent disagreement — bind base archive + tree by sha in every receipt; sj1's silent revert — prove the
  carrier is the ONLY member that changed (member-level diff of the ZIP).
- rbf1's 7× linearisation error — exact sqrt per rung.
