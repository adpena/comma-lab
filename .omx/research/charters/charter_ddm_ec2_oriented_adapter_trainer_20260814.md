# CHARTER — ddm_ec2_oriented_adapter_trainer (2026-08-14, THE EC1 TRAINER)

CONTEXT (recall, do not re-derive). EC1 landed the implicit edge-conditioning
DESIGN at fa29eb9ea1: oriented decoded-token conditioning (AUROC 0.99566,
8,380-error targeting mass = 6.25× the 1,340-flip break-even at +1,707 B),
injected BEFORE CP135's nonlinear TokenBlocks; zero-adapter = bit-identical;
nonzero control moves 589,814 pre-R values (not SA1-inert). The
MAIN_CUDA_FIRE_ORDER (sha 0d403be3…) names ONE blocker:
`true_cuda_trainer_implemented_by_producer: false` — a resumable
scorer-in-loop trainer on contest-CUDA T4 must exist before the package
command becomes fireable. Local CPU is advisory only (js1c measured the
local/T4 axis mismatch at 15,431 flips; MPS is never authority).

## THE WORK (arm builds + seals; MAIN fires all Modal dispatches)

1. **Recall first**: `.omx/research/ddm_ec1_implicit_edge_conditioning_20260814.md`
   + `experiments/ddm_ec1_implicit_edge_conditioning.py` +
   `experiments/ddm_ec1_runtime/ec1_latent_conditioner.py` + the fire order's
   `training_requirements` block at
   `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/MAIN_CUDA_FIRE_ORDER.json`
   + the proven re1t/js1b Modal worker family (dispatcher
   `experiments/ddm_re1t_modal_t4_sign_gate.py`) for image/volume/claim
   patterns. Modal module-identity law: BARE-first imports when reusing
   another module's remote function; pre-create `modal volume get` dest dirs.
2. **Build the resumable T4 scorer-in-loop trainer** as a NEW Modal app
   (entry file owns its remote function): oriented adapter params trained
   through the SAME CP135 receiver object → composite R → uint8 → frozen
   SegNet on T4, realized-flip objective vs the 34,970-flip base. Per the
   fire order: resume-from-disk on the Modal volume; DISTINCT live + EMA
   stage checkpoints + periodic intra-stage saves; full payload retention
   (model, fields, camera, scorer inputs, archives, repeat archive).
3. **SCOPE reductions (legal, declared)**: ORIENTED family ONLY in dispatch 1
   — the equal-parameter controls (class_only, undirected) are a SECOND
   sealed dispatch that fires only if oriented clears break-even (saves ~2/3
   of the spend; the capacity-vs-orientation question waits on a live win).
   Stratified pair sampling for inner training steps is legal (m88/m96:
   never prefix); the endpoint verdict field is FULL n600.
4. **Derive the wall-clock BEFORE sealing**: anchor s/step on the measured
   worker timing (~13-15 min per n600 field pass, batch 16) → derive step
   budget + hard cap ≤ 3 T4-hours (≈$1.8; #381 envelope ≈$15 left). The
   schedule is DERIVED, not guessed; record the derivation in the seal.
5. **Seal**: hash-sealed request + SEALED_FIRE_ORDER with exact
   `modal run --detach` argv, storage preflight, single-flight claim wiring,
   toy-gate-clean store under
   `pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/main_cuda/`.
   MAIN fires + polls + recovers; at endpoint the EC1 `package` command
   compiles the candidate archive and the re1t/js1b measurement buys the
   realized row.

## OPTIMAL FORM

Family reference PINS (receipts): ec1 design landing commit
fa29eb9ea17d3bfd5138478470600f322050634d · fire-order receipt
MAIN_CUDA_FIRE_ORDER.json sha
0d403be3b5af461c9e6e8c9caf77066b126f22be853c51d85509d0bcc8a6185c ·
FINAL_RESULT.json sha
bb0a6582745492dc77e4dc8a6556248bea5cc4084b06de028a4b1aa2aec76bd3 · base
instrument (34,970 flips · d_pose 6.885642960696714e-6 · 186,252 B) ·
break-even 0.785 flips/B ⇒ 1,340 flips at +1,707 B. MECHANISM
reductions = TOY-BRACKET: local-CPU or MLX scorer as training authority ·
proxy realized-flip counts · prefix-sampled verdicts · seeded design modules
scored as trained candidates (ec1's own dead-end list). Payload law DEF CON
1000. Arms cannot reach Metal or fire Modal — every dispatch is a sealed
request + fire order for MAIN.

## OUTPUT

`.omx/research/ddm_ec2_oriented_adapter_trainer_20260814.md` + trainer code +
tests + sealed fire order. Commit via `tools/subagent_commit_serializer.py`
(post-edit shas, `[no-triality] [p0-ledger-ok]`, no co-author trailer;
git-blocked → declare memo SHA for MAIN handoff). End with NEXT_IF_RESUMED +
LIVE-HYPOTHESES + DEAD-ENDS.
