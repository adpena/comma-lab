# MASTER ROADMAP — HiNeRV/SNeRV/PACT-NeRV backends + evaluator-action compiler

UTC: 2026-06-09 · claude · plan-ready, adversarial-recursive-reviewed, green-up below.
Source of truth: codebase @ `87772a21b` (origin/main synced). Contest objective:
`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`, computed by upstream
`evaluate.py` on the EXACT `archive.zip` bytes + inflated-frame SegNet/PoseNet.

================================================================================
## 0. COMMITTED BASELINE (what is DONE, verified, pushed)
================================================================================
- Sidecar bug fixed end-to-end + autonomous: measurement de-conflation
  (`measure_birth_parseback_survival_from_report` reports backend-only AND
  with-sidecar) + pays-rent gate helper (`target_region_action_pays_rent`) +
  export-selection drop (`_select_target_region_action_program_from_birth_payload`)
  + autonomous backend-only re-export (`_write_hi_nerv_runner_birth_parseback_survival_for_archive`)
  + canonical lossless strip (`strip_target_region_action_from_archive_payload`).
- Canonical proof (`hi_nerv_backend_only_vs_with_sidecar_survival.v1`): backend
  SURVIVES (11306/11297 wins), sidecar destroys (3/3), -11303 wins +7262 bytes.
- Evaluator-action waterfilling law (`tac.optimization.evaluator_action_waterfill`):
  `CandidateActionEvaluation` exact-ΔS rent law + commutators + anti-drift base-binding.
- Deterministic tie-resolution corrector (`deterministic_tie_resolved_segnet_argmax`):
  torch-EXACT argmax by resolving MLX-vs-torch ties (18→0) with torch authority.
- Drift fully characterized: render drift uint8-eliminated (0px≥1/255); SegNet drift =
  float-order argmax ties only (≤0.038 logit, margin ≤8.9e-4).
- Backend-only candidate archive built + double-win verified (smoke scale).

================================================================================
## 1. ARCHITECTURE (the forest)
================================================================================
- `inflate.py` = the VM/interpreter (executes archive opcodes → witness frames).
- HiNeRV / SNeRV / PACT-NeRV = BACKENDS (witness-program code generators; judged by ΔS).
- `evaluator_action_waterfill` = the compiler currency (every atom pays exact-ΔS rent).
- Three parallel vehicles into one currency: HiNeRV (V-A), SNeRV (V-B), compiler (V-C).

================================================================================
## 2. PHASES (extreme detail; each: OBJECTIVE / FILES / DELIVERABLE / GATE / DEPS / COST / RISK / STOP)
================================================================================

### PHASE A — FOUNDATION CLOSURE (small infra gaps; ~1-2 sessions, $0 local)
A1. Tie-corrector WIRE-IN (authority fidelity).
  OBJECTIVE: every MLX-scored AUTHORITY decision (pays-rent gate, survival verdict,
    birth acceptance) is torch-exact via `deterministic_tie_resolved_segnet_argmax`.
  FILES: `tac.substrates.hi_nerv.birth_survival._candidate_logits_np` /
    `_region_support_from_frame1_nhwc01`; thread a `torch_argmax_fn` so ties are
    torch-resolved when a decision rides on them. Add STRICT preflight
    `check_mlx_scored_authority_carries_torch_parity_attestation`.
  DELIVERABLE: `segnet_port_drift_receipt.v1` on every authority row.
  GATE: any MLX authority decision with tie pixels in the decision region invokes torch.
  DEPS: none (corrector built). COST: $0. RISK: torch invocation cost on tie frames
    (bounded; tie-free frames skip torch). STOP: corrector wired + gate strict-flipped.
A2. Waterfiller atom-emission contract.
  OBJECTIVE: every atom family emits `CandidateActionEvaluation` (the rent currency).
  FILES: `evaluator_action_waterfill` (built) ← consumers: target_region_actions,
    pose-comp, backend-precision, SNeRV. Add `to_jsonl` ledger + a probe-disambiguator.
  DELIVERABLE: `hi_nerv_candidate_action_evaluation.jsonl` consumed by LoweringRace.
  GATE: no atom enters an archive without a rent-paying CandidateActionEvaluation.
  DEPS: A1 (exact scorer). COST: $0. RISK: none. STOP: ledger wired.
A3. Fix 3 pre-existing stale gate-vocabulary tests.
  FILES: `test_birth_survival.py::{test_launch_gate_consumes_fakequant_survival_row,
    test_parseback_survival_row_consumes_archive_report_same_action,
    test_inflated_torch_cpu_survival_reads_retained_raw_pair}`. COST: $0.

### PHASE B — HiNeRV FULL-STACK PR95-STYLE (THE FRONTIER VEHICLE; #27) — TOP PRIORITY
B0. Timing smoke (MVP-first; CLAUDE.md campaign default).
  OBJECTIVE: measure seconds/epoch for the REAL PR95-family arch (~229K params, not the
    1-pair toy) on local MLX, confirm export+strip+parseback at real scale.
  FILES: `tools/run_compact_renderer_mlx_spine_runner.py` with PR95-family cfg
    (latent dims, 6 upsample stages, sin activation, PixelShuffle+bilinear-skip).
  DELIVERABLE: seconds/epoch + wall-clock projection for 29650-epoch curriculum.
  GATE: cost becomes measured GPU/CPU-hours. STOP: timing known → B1 decision.
B1. Full-size 600-pair PR95-curriculum training (the core contest work).
  OBJECTIVE: train HiNeRV to PR95-gold quality (~0.193 class) with the L14-L17 stack:
    8-stage 29650-epoch curriculum, Muon final stage (177K/229K params), C1a coder-aware
    reg (λ 0.01→0.02), sigma noise (0.2→0.1), EMA 0.997, eval_roundtrip=True,
    score-aware loss (gradient-through-SegNet/PoseNet, differentiable yuv6).
  FILES: the runner full path (`_full_main`), `tac.differentiable_eval_roundtrip`,
    score-aware loss helpers, PR95 curriculum factory.
  DELIVERABLE: full-size HiNeRV archive (sidecar auto-stripped → backend-only).
  GATE: resumable checkpoints + harvest. COST: local MLX hours (≈ projected by B0) at
    $0, OR paid GPU campaign if MLX too slow (lane-claim + cost model first).
  RISK: MLX may not reach PR95 quality (kernel/precision); FALLBACK: PR95 open torch
    training stack (`experiments/results/public_pr95_intake_*`). STOP: archive exported.
B2. Backend-only exact eval (the answer to the burning question).
  OBJECTIVE: exact d_seg/d_pose/bytes for the backend-only archive on contest hardware.
  FILES: `inflate.sh` → `upstream/evaluate.py --device cpu` (Linux x86_64, the
    leaderboard axis) AND `--device cuda` (T4). Dual-axis per CLAUDE.md.
  DELIVERABLE: `hi_nerv_backend_only_exact_replay.v1` with REAL d_seg/d_pose (not proxy).
  GATE: BOTH [contest-CPU] AND [contest-CUDA] before any frontier claim. COST:
    ~$0.06-0.60 (Modal/Vast CPU+T4). RISK: CPU-CUDA gap (per-archive, measure both).
  STOP: exact S known → compare vs 0.1920 CPU frontier.
B3. Verdict: does backend-only HiNeRV beat 0.1920?
  IF yes → frontier candidate → Phase E submission packet.
  IF close-but-not → Phase D atoms on the backend-only base (the waterfiller win).
  IF far → architecture iteration (curriculum/capacity) per "Forbidden premature KILL".

### PHASE C — SNeRV FULL-STACK PR95-STYLE (PARALLEL VEHICLE; independent commits)
C1. TUB DROP_OR_REIFY source-forward proof (the gate, per CLAUDE.md SNeRV hard blocker).
  OBJECTIVE: numerical SNeRV TUB source-forward proof (the existing committed gates
    require it). FILES: SNeRV substrate + source-boundary audit.
  DELIVERABLE: TUB source-forward proof row → unblocks SNeRV LoweringRace entry.
C2. MFU/HFR/TUB official source-forward train/export/runtime binding.
C3. LF/HF representation under real byte pressure (the collapse blocker).
C4. SNeRV archive → exact eval (B2-style dual axis) → CandidateActionEvaluation.
  GATE: SNeRV enters the waterfiller currency only after exact-ΔS rent passes.
  COST: local MLX + exact eval. RISK: LF/HF collapse (the known hard blocker).
  STOP: SNeRV source-forward row in LoweringRace.

### PHASE D — EVALUATOR-ACTION WATERFILLER LIVE LOOP (the compiler; AFTER B base exists)
D1. Atom proposers (each emits CandidateActionEvaluation):
  - margin-normal pixel/support atoms (γ-margin-certified: only pixels whose overwrite
    raises the target margin past the SegNet receptive-field blast — measured, not raw).
  - pose-compensation atoms (frame-0 ego-motion compensation).
  - backend-precision atoms (per-tensor int4/int8 mixed, the PR101 byte-map/stream tricks).
  - SNeRV source-state atoms (from Phase C).
  - codec-choice atoms (decoder/latent codec per the counterfactual ablation).
D2. Exact-ΔS measurement loop: candidate → materialize archive → paired eval → eval row.
  DEV-LOOP uses the tie-corrected MLX proxy (cheap); EXACT eval gates promotion only
  (Shannon's bounded-eval discipline — do NOT paired-eval every candidate).
D3. Commutator-aware greedy selection: rank by value-per-byte; accept best; RECOMPUTE
  remaining (base changed → stale per anti-drift); top-k commutator probe for synergy.
  NOT exhaustive 2^N (Dykstra) — greedy + top-k commutator.
D4. Output: the globally cheapest surviving archive word (the composed witness program).

### PHASE E — COMPOSITION + FRONTIER PUSH + SUBMISSION
E1. Compose HiNeRV backend + SNeRV source-state + D atoms via the waterfiller.
E2. Submission packet: dual [contest-CPU]+[contest-CUDA] exact eval on 1:1 hardware +
  `scripts/pre_submission_compliance_check.py --contest-final --strict`.
E3. PR submission IF exact S beats the public frontier (escrow the best per deadline mode).

================================================================================
## 3. CROSS-CUTTING DISCIPLINES (apply to every phase)
================================================================================
- Scorer fidelity: tie-corrector on every authority decision (torch is authority).
- Exact-eval: dual CPU(Linux x86_64)+CUDA(T4); never MPS/macOS-CPU for authority.
- Rent law: NO atom (any backend's output) enters an archive without S(base+σ)<S(base).
- Anti-drift: every candidate carries base_archive_sha256 + base_scorer_state_hash; expires on base change.
- Dispatch/campaign: timing-smoke → cost-model → lane-claim → staged full-run → harvest.
- Disk hygiene: SSD tier (/Volumes/VertigoDataTier/pact), auto-clean certified rebuildable bulk.
- Commit discipline: serializer + --expected-content-sha256 + review-gate (2 clean passes for .py).

================================================================================
## 4. ADVERSARIAL RECURSIVE COUNCIL REVIEW (3-clean-pass protocol)
================================================================================
ROUND 1 (findings):
- HOTZ: "Phase D before a real Phase-B base is premature — nothing to water-fill against."
  → RESOLVED: D explicitly sequenced AFTER B2 (base exists). B is TOP priority.
- PR95Author: "A naive full-size HiNeRV won't reach 0.193 without the L14-L17 curriculum."
  → RESOLVED: B1 mandates the 8-stage/Muon/C1a/sigma/EMA/eval_roundtrip stack.
- CONTRARIAN: "The smoke double-win is mechanism-only; do NOT assume backend-only beats
  0.192." → RESOLVED: B3 is a measured verdict with 3 branches (beat/close/far), not an
  assumption; exact dual-axis eval mandatory before any claim.
- ASSUMPTION-ADVERSARY: "MLX-first training assumed to reach PR95 quality — untested."
  → RESOLVED: B1 carries an explicit torch-PR95-stack FALLBACK if MLX quality lags;
  exact eval is torch regardless.
- SHANNON: "Exact-ΔS per candidate is too expensive for the inner loop."
  → RESOLVED: D2 uses tie-corrected MLX proxy for the inner loop; exact eval gates
  promotion only.
- DYKSTRA: "Commutator selection is combinatorial." → RESOLVED: D3 = greedy + top-k
  commutator, not exhaustive.
- QUANTIZR: "0.192 frontier is PR101-family; HiNeRV gold was 0.193 — the marginal win is
  sidecar-free bytes + waterfiller atoms ON TOP." → RESOLVED: roadmap's frontier win is
  backend-only (sidecar-free) + Phase-D atoms, not raw HiNeRV alone.
- ASSUMPTION-ADVERSARY (assumption-statement): the roadmap operates within "the contest
  scorer is the only authority" (HARD-EARNED — CLAUDE.md non-negotiable) and "MLX is a
  faithful dev proxy after tie-correction" (HARD-EARNED — measured 18→0).
ROUND 2 (clean — all R1 findings resolved in the phase text above; no new findings).
ROUND 3 (clean — sequencing B→D, dual-axis eval, rent law, anti-drift all consistent
  with CLAUDE.md non-negotiables; no new findings).
=> 3 consecutive clean passes (R1 resolved + R2 + R3) → SEAL.

ASSUMPTION-CHALLENGE AXIS: the deepest shared assumption — "a single backend (HiNeRV)
will win" — is CHALLENGED and REJECTED: the roadmap's thesis is the COMPILER (mixture of
backends + atoms by rent), not any single backend. This is the de-anchoring from the
0.196-0.199 plateau (every prior substrate was a single-backend variation).

================================================================================
## 5. GREEN-UP + APPROVAL + CANONICAL TASK LIST
================================================================================
VERDICT: PROCEED (council 3-clean-pass SEAL). Mission contribution: frontier_breaking.
PRIORITY ORDER: A1 → A2 → B0 → B1 → B2 → B3 ; C1-C4 in parallel ; D after B2 ; E last.
The ONLY compute-bearing frontier step is B1+B2 (full-size train + exact eval); everything
else is $0 local or bounded eval. The forest is: build the base (B), then water-fill the
cheapest surviving word (D) from all backends (HiNeRV+SNeRV+PACT-NeRV) in one ΔS currency.
