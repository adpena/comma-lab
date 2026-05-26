# Z7-Mamba-2 L1 EMPIRICAL Fair-Shake — Pre-Execution Gate Report 2026-05-26

**Subagent**: `z7-mamba-2-l1-empirical-mlx-fair-shake-20260526`
**Lane**: `lane_z7_mamba_2_v2_l1_empirical_mlx_fair_shake_20260526`
**Cost**: $0 (MLX-local on M5 Max)
**Verdict**: PROCEED with v2 fresh substrate as canonical promotion target

## Pre-flight verifications

- **PV-1**: CLAUDE.md + AGENTS.md NON-NEGOTIABLEs read (UNIQUE-AND-COMPLETE-PER-METHOD; MLX portable-local-substrate authority; Forbidden premature KILL; PER-SUBSTRATE OPTIMAL FORM; Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY).
- **PV-2**: Both new 2026-05-26 standing directives in scope (MLX↔CUDA bidirectional drift; pushing-the-frontier-of-research-on-optimization-algorithms).
- **PV-3**: Lane registry inspected: 7 Z7-Mamba lanes exist; `lane_path_3_b_prime_z7_mamba_2_cargo_cult_first_20260526` is the canonical L1 promotion target at L1 (impl_complete=true, no empirical anchor yet).
- **PV-4**: Substrate dir comparison: `time_traveler_l5_z7_mamba2/` (v1; 1321 LOC; bolt-on per cargo-cult audit) vs `z7_mamba2_v2_fresh_substrate/` (v2; cargo-cult-first fresh design per Phase 1+2+3 memos).
- **PV-5**: Predecessor checkpoint = none (no successor predecessor of this subagent).
- **PV-6**: Sister Z6 #1287 L1-PROMOTION pattern read in full (`.omx/research/path_3_d_z6_l1_promotion_landed_20260526.md`); reusable template.
- **PV-7**: MLX 0.31.2 importable; contest video `upstream/videos/0.mkv` 35.8 MB present.

## Canonical promotion target decision

**Target**: `z7_mamba2_v2_fresh_substrate` (Phase 3 cargo-cult-first fresh design)
**Rationale**: per operator UNIQUE-AND-COMPLETE-PER-METHOD non-negotiable + Phase 1 cargo-cult audit (8 NEW CARGO-CULTED unwinds beyond v1's existing CC-1..CC-10). The v2 substrate is designed FROM FIRST PRINCIPLES around Mamba-2 SSM math, NOT extended from the v1 Z6-decoder bolt-on pattern.

**v1 preserved** per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE; predecessor's state_dict-key-parity research is RESEARCH INPUT only (informs MLX-PyTorch equivalence at math layer).

## Sister coordination per Catalog #230/#302/#340

- Slot 1 (NSCS06 v8 + fec6 4-arm paired Modal T4): disjoint substrate scope; PAID.
- Slot 3 (BoostNeRV BPR1 Variant B codec redesign): disjoint substrate (NeRV-family).
- No other in-flight subagent touches `z7_mamba2_v2_fresh_substrate/` or `experiments/train_substrate_z7_mamba2_v2_mlx.py`.
- Catalog #340 sister-checkpoint guard PROCEED for all my files.

## Execution plan

1. NEW MLX-native trainer `experiments/train_substrate_z7_mamba2_v2_mlx.py` (~530 LOC) implementing:
   - `Z7Mamba2V2MLXRenderer` MLX-native module (Mamba-2 cell + temporal Conv1D pre-stage + spatial decoder)
   - Per CC-A unwind: distinguishing-feature Conv1D temporal pre-stage in decoder (toggle-able)
   - Per CC-B unwind: latent_dim=32 default
   - Per CC-C unwind: ego_motion_dim=16 default
   - Per CC-D unwind: A_log init scheme configurable (z_plus_1 / hippo_like / log_uniform)
   - Per CC-H unwind: ib_scale=5e-4 default
   - Per CC-J unwind: A_log procedurally regenerated (NOT serialized in archive)
   - MLX-native EMA decay=0.997 per Catalog #2
   - REAL contest video via canonical `tac.data.decode_video`
   - Canonical Provenance throughout per Catalog #287/#323
   - All MLX outputs tagged `[macOS-MLX research-signal]` per Catalog #1/#192/#317/#341

2. Smoke verification (10p × 3ep): expect monotonic loss decrease
3. Convergence smoke (50p × 30ep): mirror Z6 #1287 protocol
4. CC-A ablation at same epoch budget: A/B comparison of temporal Conv1D distinguishing feature
5. Landing memo with empirical receipts + cross-pollination findings vs NeRV-family

Issue with deferred state: NaN at ep 16-18 (Mamba-2 stability bug class per `z7_mamba_2_multi_week_path_forward_20260518.md`). Mitigation: anchor at ep 15 (pre-NaN canonical state); operator-routable for L2 stability hardening (gradient clipping + warmup-decay schedule).

## Discipline applied

- Catalog #229 PV (above)
- Catalog #206 subagent checkpoint discipline (`subagent_progress.jsonl` rows at steps 1, 2, 3, 4)
- Catalog #126 lane pre-registration (this report carries the lane intent)
- Catalog #110/#113 APPEND-ONLY (NEW lane + memos; NO mutation of v1 substrate, v2 L0 SCAFFOLD memo, lane registry L0 entries, sister subagent state)
- Catalog #230/#302/#340 sister-subagent ownership map honored (0 file overlap)
- Catalog #287 placeholder-rationale rejection (≥4 chars throughout)
- Catalog #340 sister-checkpoint guard
- CLAUDE.md "Remember all on MLX" — NO PAID DISPATCH
- CLAUDE.md "Forbidden premature KILL" — paradigm INTACT; NaN at ep 16+ is IMPLEMENTATION-level stability bug class per Catalog #307

## Cross-references

- Phase 1 cargo-cult audit: `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Phase 2 design decision: `.omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md`
- Phase 3 L0 SCAFFOLD design memo: `.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md`
- Phase 3 L0 SCAFFOLD landing memo: `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md`
- 2026-05-18 stability multi-week path forward: `.omx/research/z7_mamba_2_multi_week_path_forward_20260518.md`
- v1 mlx_native (reference): `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py`
- Sister Z6 #1287 L1 promotion pattern: `.omx/research/path_3_d_z6_l1_promotion_landed_20260526.md`
