# ddm_rr17 ROUND17_FINDINGS

## Verdict

CLEAN. rr17 advances the clean-pass counter to 1/3.

SELECTION-POLICY HOLD RELEASED. MAIN's `mx1t_provenance_addendum.json`
honestly closes rr16 F1 for the arm-selection boundary: the mx1t K=8
tail-average row remains only an n32 advisory ARM-CAP selection signal
(`[macOS-CPU advisory torch upstream SegNet]`, `score_claim=false`), but it is
now cache-bound enough to consume for that scoped purpose.

No Metal, MPS, scorer pass, n600 run, archive replay, or `upstream/evaluate.py`
run was performed. The live run directory was not touched.

## RECALL EVIDENCE

Sources searched/read before adjudication:

- Governing files: `.omx/research/ddm_rr17_20260807/CHARTER.md`,
  `.omx/tmp/codex_runs/_common_contract.md`, `PROGRAM.md`, `CLAUDE.md` /
  `AGENTS.md` (byte-identical), `docs/operating_manual_craft_handoff.md`,
  `.omx/state/main_hot_state.md`.
- Charter-surface query over `.omx/research`, `.omx/state`, `docs`, `src`,
  and `reports`: `mx1t|mx1g|rr16|gt_seg_cache|tail-average|selection-policy|selection policy|argv_n32_arm_cap|aa1 repair`.
- Canonical-equation recall: `.venv/bin/python tools/list_canonical_equations.py --json | rg "mx1t|mx1g|rr16|gt_seg_cache|tail-average|selection-policy|selection policy|argv_n32_arm_cap|aa1 repair"`.
- Research-index/DAG recall: the same query over `.omx/research/CANONICAL_RESEARCH_INDEX*`
  and `.omx/research/sub015_DAG_*`.
- Task/queue surface recall: `find .omx -maxdepth 3` for task/ledger/queue/backlog
  files, then targeted reads/searches of the rr16/mx1g/mx1t and harvest artifacts.

Findings beyond the charter seeds that affected the plan:

- `.omx/state/active_lane_dispatch_claims.md` already recorded mx1t completion
  as local n32 torch facets and `score_claim=false`, matching the scope I kept.
- `.omx/state/canonical_equations_registry.jsonl` contains
  `jd1_plateau_tail_average_ema_v1`; that is adjacent precedent only, not a
  transfer of mx1t's simple post-hoc parameter average into a general plateau law.
- `.omx/research/ddm_rr11_20260807/ROUND11_FINDINGS.md` showed prior argv/cache
  clobber risk around the same ticket family, so I checked the live regenerated
  `argv_n32_arm_cap` contents directly rather than trusting high-level ticket text.
- `.omx/research/ddm_mx1h_20260807/MX1H_FINDINGS.md` confirmed the GT-cache
  torch-verdict pattern and step-1500 anchor context; it did not change the
  rr17 release condition.

Nothing found in the searched scope changed the rr17 charter's release test:
the decision still rests on cache SHA, live ARM-CAP argv binding, and rr16's
repeat measurements.

## F1 Release Check

The addendum under review:

- `.omx/research/ddm_mx1t_20260807/mx1t_provenance_addendum.json`
- status field: `HOLD released pending rr17 verification of this addendum`
- binding claim: input cache and target cache are both
  `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt`
- recorded bytes: `943720076`
- recorded SHA-256:
  `286fe40a2a29aa6950684f43229fce3a4a284ac7ffc65040e7e18953b95787d4`

Independent filesystem verification:

- `stat -f '%z %N' /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt`
  returned `943720076`.
- `shasum -a 256 /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt`
  returned
  `286fe40a2a29aa6950684f43229fce3a4a284ac7ffc65040e7e18953b95787d4`.

Live ARM-CAP fire argv verification:

- `.omx/research/ddm_mx1g_20260807/launch_ticket_mx1g_from_regen2.json`
  top-level `argv_n32_arm_cap` contains `--input-cache` and `--target-cache`
  both pointing to the same GT cache path above.
- The same argv carries `--pairs 32`, `--steps 6000`, `--seed 20260806`,
  `--fire-argv-key argv_n32_arm_cap`, and the live fire-guard ticket path.

rr16 reproduction match:

- rr16 default parser/cache replay measured `d_seg = 0.004567305246988933`,
  proving cache identity is load-bearing.
- rr16 repeat A under GT-to-GT measured `d_seg = 0.0010732014973958333`,
  mismatch pixels `6752`.
- rr16 repeat B under GT-to-GT measured `d_seg = 0.0010732014973958333`,
  mismatch pixels `6752`.
- The committed mx1t step-3250 row is the same
  `d_seg = 0.0010732014973958333`; the K=8 tail-average row is
  `d_seg = 0.0010673205057779949`, delta vs final
  `-0.000005880991617838397`.

Conclusion: rr16 F1's release condition is satisfied. The addendum does not
pretend to be per-row machine schema repair, but it binds the invariant cache
identity and argv for the load-bearing mx1t selection-policy use. The HOLD is
released for the n32 advisory arm-selection boundary only.

## Commit-Scope Review

No #911-genus absorption found in the two MAIN repair commits checked.

- `c70d78198f` (`ddm_aa1+rr16 arm artifact commit repair...`) contains only:
  `.omx/research/ddm_aa1_20260807/AA1_ATTACK_SHEET.jsonl`,
  `.omx/research/ddm_aa1_20260807/AA1_FINDINGS.md`,
  `.omx/research/ddm_aa1_20260807/CHARTER.md`,
  `.omx/research/ddm_aa1_20260807/aa1_pr130_payload_blind_audit.json`,
  `.omx/research/ddm_oh1_20260807/CHARTER.md`,
  `.omx/research/ddm_rr16_20260807/CHARTER.md`,
  `.omx/research/ddm_rr16_20260807/ROUND16_FINDINGS.md`,
  `.omx/research/ddm_rv2_20260807/CHARTER.md`.
- `bd031e496d` (`ddm_mx1t cache-bound provenance addendum...`) contains only
  `.omx/research/ddm_mx1t_20260807/mx1t_provenance_addendum.json`.

The earlier mx1t arm repair commit `0bc7c20966` was also checked for context and
contains only mx1t-owned artifacts plus the analyzer/tests named by that arm:
`CHARTER.md`, `MX1T_FINDINGS.md`, the two mx1t JSONL receipt files,
`experiments/ddm_mx1_pr130_semantic_renderer.py`, and
`experiments/tests/test_ddm_mx1_memory_probe.py`.

## Harvest-Table Audit

All three chartered harvest tables parse as JSONL and carry honest
non-score labels:

| table | rows | parse errors | score_claim not false | promotion_eligible not false |
| --- | ---: | ---: | ---: | ---: |
| `.omx/research/ddm_rv2_20260807/RV2_REGRADE_TABLE.jsonl` | 21 | 0 | 0 | 0 |
| `.omx/research/ddm_oh1_20260807/OH1_CONSUMPTION_PLAN.jsonl` | 43 | 0 | 0 | 0 |
| `.omx/research/ddm_aa1_20260807/AA1_ATTACK_SHEET.jsonl` | 13 | 0 | 0 | 0 |

Commit custody:

- RV2 table last commit: `b61cacae01 ddm_rv2: regrade corpus against composed vehicle`.
- OH1 table last commit: `81bb5edcb5 ddm_oh1 orphan-signal consumption plan`.
- AA1 table last commit: `c70d78198f ddm_aa1+rr16 arm artifact commit repair...`.

Spot checks, three per table:

- RV2 `rv2_r06_ix2tok01_sv2_coder_on_semantic_payload` cites HB1 live-row
  evidence. HB1's receipt/table says OUR-label HPAC is not measured, PP1 KT is
  the measured incumbent (`142001` B for tq1c labels; `173617` B for GT labels),
  and HPAC remains blocked/queued. RV2 only marks it covered by live row, not
  measured HPAC.
- RV2 `rv2_r10_869_933_adaptive_waterfill_semantic_payload` cites AA1 row
  `tz1_869_adaptive_per_cell_l`; that row exists and labels the delta as
  projected from our stream and unmeasured on PR130, with a queued fire order.
- RV2 `rv2_r13_pr130_blind_coordinate_direct_401` cites AA1's blind audit; the
  audit records `camera_resolution_storing_counted_payload_bytes = 0` and
  `direct_blind_coordinate_reclaim_bytes = 0`.
- OH1 `p1a_phi_composite_r_adjoint` cites the P1A follow-on memo; the source
  contains the phi / one-over-phi composite-R adjoint item and labels it
  unmeasured/read-not-run.
- OH1 `mh1_split_bank_gate_per_receipt` cites MH1; the source contains the
  split-bank consumption-per-receipt finding and the named A1 row.
- OH1 `mh1_recover_lane_skipband_arm_c_524` cites MH1; the source contains the
  ARM-C `LaneSkipBand` #524 entry and says it was absent from the tracked lever
  activation ledger.
- AA1 `aa1_blind_coordinate_direct_cpr1` matches its audit: direct reclaim is
  `0` B and score delta `0`.
- AA1 `cpr1_carrier_repack` matches
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/evidence/cpr1_verification.json`:
  `archive_bytes_saved = 3328`, lossless round-trip evidence present.
- AA1 `hpac_model_self_compress` matches the same CPR1 verification artifact:
  raw HPAC bytes `20179`, packed model bytes `15164`, saving `5015` B, and
  `packed_model_max_logit_diff = 0.0`.

No fabricated receipt or row/assertion mismatch was found in this bounded
light-touch sample. I did not exhaustively verify every evidence reference in
all 77 rows.

## Assumption-Challenge Axis

Shared assumption challenged: a cache-bound addendum can release the mx1t
selection-policy HOLD without rewriting every original mx1t facet row.

Verdict: accepted for this scoped boundary. The original defect was that a
consumer could not know whether the mx1t numbers were GT-to-GT or parser-default
tq1c-to-GT. The addendum, live ARM-CAP argv, independently recomputed cache SHA,
and rr16 repeats jointly close that uncertainty for the n32 arm-selection
decision. Requiring every historical JSONL row to be rewritten would be cleaner
for machine-local consumption, but would not unlock a breakthrough by itself and
is not needed to decide K=8 vs final for this ARM-CAP n32 advisory policy.

What would violate the assumption and matter: if the K=8 tail-average row were
promoted as n600, public-wire, exact, or population-selection evidence, this
addendum would be insufficient. That promotion still requires fresh n120/n600
or exact byte-closed measurement under the proper scorer/contest authority.

## Boundary

Measured in rr17: cache bytes/SHA, live JSON argv binding, JSONL parseability,
score-claim labels, commit file scopes, and bounded receipt spot checks.

Did not measure in rr17: any scorer output, n600 behavior, exact archive score,
Metal/MPS behavior, full-table receipt truth for all 77 rows, or public-wire
performance.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
Contest pointer remains borrowed/unmoved.
