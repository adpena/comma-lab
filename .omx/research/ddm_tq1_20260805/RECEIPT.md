# ddm_tq1 RECEIPT

score_claim: false
axis: [macOS-CPU frozen-scorer advisory]
phase: A_BUILD_SCORER_LIGHT (13.185s)
phase_b: GATED_CLOSED_JD5_ENDPOINT_ABSENT
follow_on: QUEUED-WITH-FIRE-ORDER

## BUILD SITES

- Driver: `experiments/ddm_tq1_optimal_token_edit.py`
- SSD root: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form`
- Price ledger: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form/tq1_phase_a_candidate_prices.jsonl`
- Candidate menu: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form/tq1_phase_a_candidate_menu.jsonl`
- Phase A JSON: `/Users/adpena/Projects/pact/.omx/research/ddm_tq1_20260805/phase_a_receipt.json`
- Resume instructions: `/Users/adpena/Projects/pact/.omx/research/ddm_tq1_20260805/NEXT_IF_RESUMED.md`

## PHASE A RESULT

- Base archive verified: `/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip` sha256 `d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`, 357836 B.
- Candidate menu generated: 1140 moves with exact affected-pair sets from the decoded receiver token lattice.
- Candidate prefix priced: 12 moves; accepted moves: 0 (`NOT_RUN_PHASE_A_SCORER_LIGHT`).
- Smoke byte-close archives: 4; each parsed with `build_byte_ledger`.
- Global `[16,12,8,4]` is recorded only as a dominated baseline, not used as the generator.

## PHASE B

- Status: QUEUED-WITH-FIRE-ORDER.
- Gate checked: no jd5 endpoint receipt with `status=complete` was found.
- Required acceptance rule is implemented as `acceptance_verdict`: realized joint `delta_S < 0`, pose-term erosion <= 0.005, real archive bytes.

## RECALL EVIDENCE

- `.omx/tmp/codex_runs/tq1_prompt.md`: Phase A build now, Phase B only after jd5 complete endpoint; prior blanket token edits are instance negatives.
- `.omx/tmp/codex_runs/_common_contract.md`: serializer commit, two review passes for `.py`, no forbidden-file edits, no `/tmp` persisted evidence.
- `.omx/state/main_hot_state.md`: qo1 live own-vehicle row and fresh rt1/fz4/ed2 negatives; scorer boundary remains jd5/sq2 ordered.
- `.omx/research/ddm_tq1_preempted_by_rt1_and_sl2_composition_20260805.md`: earlier tq1 preemption is stale for this charter because it only replayed blanket-map negatives.
- `experiments/ddm_td1_token_drop_guided_surface.py`: reused cached scorer-instrument guard fields and IX2 staging pattern.
- `experiments/ddm_tw1_token_waterfill_state_dependence.py`: incorporated the state-dependence lesson by repricing candidates on the actual token object.
- `tools/measure_ddm_dr2b_tolerance_costate.py` and `tools/measure_ddm_rd1_lambda_continuation_frontier.py`: Phase B fire order must reuse their v19/v19b realized move-level accounting before closure.

## FOLLOW-ONS

- QUEUED-WITH-FIRE-ORDER: run Phase B after jd5 complete endpoint, with psutil RSS preflight <= 20 GB and checkpoint every about 20 scorer moves.
- FOLDED: previous rt1/fz4/ed2 blanket negatives are dominated-baseline context only; they do not close the TQ1 family.
