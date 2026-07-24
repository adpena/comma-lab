# Codex findings — DDM MS2R tolerance-capped solve R2

UTC: 2026-07-24T19:10:00Z
Lane: `lane_ddm_ms2r_tolerance_capped_r2_20260724`
Evidence axis: `[macOS-CPU frozen-scorer advisory]`
Authority: `research_only=true`, `score_claim=false`, `promotion_eligible=false`

## Verdict

`MEASURED_RECEIVER_CLOSED_BINARY_Q4_Q8_CONTROL_KNEE_INSIDE_BOX; FULL_FISHER_G4_WATERFILL_AND_RD1_CELL_DUALS_STILL_BLOCKED`

Verdict scope: INSTANCE C1 exact scorer-quotient × finite per-pair q4/q8 uniform-quantum control on macOS CPU batch16; **[naive-uniform-quantum upper bound]**, not a full per-dimension Fisher/G4 or contest score verdict.

The exact finite-control knee uses 208 q4 pairs and 392 q8 pairs. It lands at exactly 136,839 errors under the 136,839 cap, with `d_seg=0.001159998575846354`, `d_pose=0.01663315449034709`, and a receiver-closed archive of 291,205,400 bytes. The registered callable returns `S=194.42556029038283`. This is a real describe-line control point, not a frontier candidate; the canonical pointer stays `0.1910828242 [contest-CPU]`.

## Evidence disposition

### MEASURED

- Fresh q1 exact control: 17,927 errors, `d_seg=0.0001519690619574653`, `d_pose=0.00010184312078531729`.
- Fresh global q4 control: 79,459 errors, inside the cap.
- Fresh global q8 control: 192,115 errors, outside the cap.
- Exact dynamic program over the complete binary q4/q8 per-pair family: 136,839 errors, 208 q4 pairs, 392 q8 pairs, 291,203,920 additive predictor-record bytes.
- Production receiver parse-back: archive 291,205,400 bytes, SHA-256 `e3d0581ff4a3f475057e77e530374dad444b640a049b058cd66b37563534773e`; deterministic archive construction passed twice.
- Fresh coder race: 50 independently checkpointed streams × RAW, Brotli q11, constriction order-1 context ANS, and trained-dictionary zstd19. RAW won 50/50; no coder row was unavailable; every row parsed back exactly.
- Registered callable `ddm_tolerance_capped_min_score_waterfill_v1`: admissible inside cap, 291,205,400 counted bytes, zero admitted coder gain, `S=194.42556029038283`.

### DERIVED

- The measured control removes 118,321,525 bytes relative to the 409,526,925-byte C1 exact control.
- The rate term alone is `193.90172278752317`; therefore this construction is a custody-valid control for the describe line but not a useful frontier candidate.
- The candidate can serve as an exact receiver-closed quotient-plane input to `incumbent_v1`; a fresh composed score row is still owed to ic2/MAIN and is not claimed here.

### BLOCKED / NULL

- Headline status is `HEADLINE_BLOCKED` on exactly:
  - `TYPED_SUBPROBLEM_ALTERNATION_NOT_ACTIVE`
  - `TYPED_BLOCK_ATLAS_NOT_ACTIVE`
  - `PER_DIMENSION_EFFECTIVE_QUANTA_NOT_ACTIVE`
- The q4/q8 menu is a deliberately naive uniform-quantum upper bound. The proper replacement is the typed per-dimension Fisher/G4 construction named by `dynamic_quantum_calibration_v1`.
- All 162 `stratum × scorer_visibility × G4` RD1 cells remain NULL. The measured aggregate binary q4/q8 edge is non-transferable because C1 predictor bytes are not assigned to those cells.
- No smaller sum from the stream-local coder race is admitted: no counted multi-stream receiver container was materialized. The raw production archive is the only complete, one-object byte custody.
- This is not a contest-CPU or contest-CUDA result and does not authorize promotion.

## Adversarial fixes made during the run

1. The first coder-stage implementation checkpointed only after all 50 streams. That violated the per-stage resumability contract. Only that stage was stopped; per-stream atomic checkpoints were added; the stage resumed and preserved all 50 results.
2. A smaller sum of isolated coded streams could have been mistaken for a counted archive. Admission now fails closed unless a complete receiver container exists; the production raw archive remains the byte authority.
3. The 162-cell dual table could have been populated from an aggregate binary edge. The supplement preserves all cells as NULL and marks the aggregate edge non-transferable.
4. Runtime paths and wall-clock receipt time made the first receipt host/time dependent. Paths were normalized, the receipt timestamp was sealed in typed config, and two full checkpoint-resume replays produced receipt SHA-256 `03cd9aabc1275c49c983631dd547e7497f8fe95804a9bfd7a24c5d61e9a81d25`.

## STORES CONSULTED

- Delegated authority file, SHA-256 `52639db184e5d89a88b30ef24b3a633e3550e7a6fb4e7e03c14e0fbee5179600`
- `CLAUDE.md` and `AGENTS.md`
- `.omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/BUNDLE-COMPLETE.json`
- `.omx/research/ddm_ms2r_min_score_waterfill_20260724T161242Z/receipt.json`
- `.omx/research/ddm_rd1_distortion_rate_20260724T121221Z/receipt.json`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- Per-arm inbox `ddm_ms2r_tolerance_capped_solve_r2.jsonl`
- Broadcast inbox `_broadcast.jsonl` through `2026-07-24T17:39:13Z`

## MAIN landing review

MAIN must review the isolated commit before landing. In particular, re-check the finite-family label, exact cap equality, receiver/coder byte admission, all-NULL RD1 disposition, callable output, and headline refusal. Do not treat this worktree or its SSD checkpoints as a main-branch source of truth until that review lands it.
