# DDM CD1 working-tree debt landing — 2026-09-03

## Verdict

The stale tree is fully adjudicated, but this sandbox could not move every
adjudicated byte into `main` because Git index writes are denied. Thirty-nine
landable paths were split into eight logical serializer units. Every serializer
attempt declared every post-edit SHA-256 and ended with the required message
suffix; all eight returned the expected `rc=17` and retained an intended commit,
bundle, format patch, and machine-readable receipt on VertigoDataTier. They are
`LAND (bundle ready; not in HEAD)`, not fake merged commits.

Two 133-byte nested PQ1 hook markers were the only `REVERT` disposition. Their
owner review explicitly called them out-of-packet repo litter worth sweeping;
their exact bytes were copied to SSD and SHA-verified before removal. The QBR1
claim ledger and GC1 source/test remain `HOLD(live-arm)`. FPC3 and GF2 became
clean through concurrent MAIN landings while this audit was running.

The charter prediction did not hold: **27/39 = 69.23%** of the original
non-FPC3/non-GC1 path population matched a serializer fallback bundle byte for
byte. That is below the predicted 80%, but above the stated fewer-than-half
falsifier. The uncovered remainder was AU1 generated audit state, MAIN state,
one JG2 owner test, QBR1's live claim ledger, and the two stray PQ1 markers.

This handoff follows `docs/operating_manual_craft_handoff.md`: claims below name
their custody, exact boundary, consumer, and fire condition. It consumes rather
than repeats `.omx/research/ddm_dd1_drift_debt_ledger_verdict_20260901.md` and
lands the verified HT1 verdict instead of re-auditing its 84-lane conclusion.

## Serializer landing units

| Unit | Source custody | Intended commit | New bundle SHA-256 | New bundle |
|---|---|---|---|---|
| PQ12 digest repair | owner fallback `8d35566d...7d02`, commit `ed9cde0512`, 1/1 same | `312f637d63842baf944d65d5b0feb3283f1fa211` | `6bc38a7c802c55d6f48be03419935f6ad13a556de63f5feb1fc8caf65b0cef08` | `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/pq12/20260903T185549.613223Z-50772/intended-commit.bundle` |
| HT1 enforcement | owner fallback `09b100b3...a24`, commit `076402bb8b`, 14/14 same | `b0d64fdcb0a082d7ee8220ab672eefa405b15c11` | `048e553ce1a79bb61cf575c8283b87a30db84698fc76d3fc38a108fe289e7399` | `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/ht1_enforcement/20260903T185557.205960Z-50943/intended-commit.bundle` |
| JG2/JG4 checkpoint cure | HT1 coder fallback `e0a4ea...e1c0`, commit `1a42ff6f6f`, plus owner test and CD1 drift fix | `7235ad6d208e5c49c274edbb541ce984793df46e` | `258dfaee6575edee7f3229be16f64eb386a6cc6a42854f3ea1152708557bf227` | `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/jg2_jg4/20260903T185606.658948Z-51237/intended-commit.bundle` |
| HT1 verdict | owner fallback `bcddefbd...ea4a`, commit `d5a670a975`, 1/1 same | `6da8d2cfa6294db3f488bae4903cebceb3de9b8b` | `cbaa222830222cfbd1344bf6e76a7835da36ea23bb1835c7ca9cc94332525ff7` | `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/ht1_memo/20260903T185611.836766Z-51377/intended-commit.bundle` |
| LV3 residual chain | verified chain bundles `bfd48be3`, `474f3fd6`, `44e045f8`, `4839e960`; 10/10 final bytes same | `62ca8287199fcb45faf2497ef92129859270e63c` | `ee92e20ef900269120b19d9d5525dcd65716932e500d1d0710fdac63dd1a9465` | `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/lv3/20260903T185620.303442Z-51550/intended-commit.bundle` |
| AU1 rebuilt indexes | AU1 audit directory and its own two test modules; no prior fallback | `121116d5fb1816f84d905ea4721be7991fca45e0` | `2a3152f0ebcd34c98aba0da9952c19280f995c6a51dccaf2f93ec963f0daa6a9` | `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/au1/20260903T185628.375534Z-51805/intended-commit.bundle` |
| MAIN G8V1 state | current focus, harvested call ledger, latest report; no prior fallback | `ed0ec9200a184ddb9b332d526aa391fdb1352a63` | `fcd292cf2d18af90cb2b9cc071d009d69818c62862bb1d5f150b5e16a24a9e44` | `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/main_state/20260903T185634.505372Z-51987/intended-commit.bundle` |
| Test-drift repair | CD1 registry lost-update repair plus stale pointer fixture | `19f43a580e297e537ab7876e5ac8cd87e4403efe` | `9c3e3eac52cf4b3ba964374c0dc72a23ceb781103f6f1530994c797f6d12aa58` | `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/test_drift/20260903T185647.286517Z-52462/intended-commit.bundle` |

All eight receipts say `BUNDLE_READY_MAIN_MUST_LAND`. Their base was
`acce5094f52f5a5f195c44af1ec76c8934f06220`; GF2 subsequently moved `main` to
`ab93088002e7eb80d4660c6a1a999d2d957fc67a`, so a Git-writable consumer must
apply them as logical cherry-picks/format patches, not reset `main` to the bundle
parents.

## Per-path ledger

Test abbreviations: `R501` is the combined owner regression run (501 passed,
one subsequently repaired stale JG4 assertion); `J46` is the post-repair JG2/JG4
run (46/46); `C22` is the canonical-equation pair (22/22); `JSON` is strict parse
of 30,343 records/documents across 14 files; `PC18` is `py_compile` on all 18
landable Python paths; `PF` is the fast preflight invocation. `Δ` is added/deleted
lines relative to the then-current `HEAD`, or lines/bytes for untracked files.

| Path | Δ | Owner / last naming evidence | Prior bundle | Verification | Disposition / commit |
|---|---:|---|---|---|---|
| `.omx/research/ddm_au1_20260805/au1_corrections_index.jsonl` | +497/-94 | AU1 audit outputs | none | AU1 tests in R501; JSON | LAND bundle / `121116d5fb` |
| `.omx/research/ddm_au1_20260805/au1_headline_vs_body.jsonl` | +331/-21 | AU1 audit outputs | none | AU1 tests in R501; JSON | LAND bundle / `121116d5fb` |
| `.omx/research/ddm_au1_20260805/au1_if_budget_checks.jsonl` | +1/-1 | AU1 audit outputs | none | AU1 tests in R501; JSON | LAND bundle / `121116d5fb` |
| `.omx/research/ddm_au1_20260805/au1_scope_vs_object.jsonl` | +45/-2 | AU1 audit outputs | none | AU1 tests in R501; JSON | LAND bundle / `121116d5fb` |
| `.omx/research/ddm_au1_20260805/au1_summary.json` | +11/-11 | AU1 audit outputs | none | AU1 tests in R501; JSON | LAND bundle / `121116d5fb` |
| `.omx/research/ddm_pq1_submission_packet_prep_20260815/verify_files_digest.py` | +3/-3 | PQ12 packet digest repair | `8d35566d...7d02`, SAME | PC18; source review | LAND bundle / `312f637d63` |
| `.omx/research/ddm_pq1_submission_packet_prep_20260815/.omx/state/magnitude_dismissal_marker.json` | 1 line/133 B | PQ1 `REVIEW_PASS10_FRESH_EYES.md`: hook litter worth sweeping | exact SSD copy | source/retained SHA match | REVERT; custody receipt |
| `.omx/research/ddm_pq1_submission_packet_prep_20260815/.omx/state/triality_drift_marker.json` | 1 line/133 B | PQ1 `REVIEW_PASS10_FRESH_EYES.md`: hook litter worth sweeping | exact SSD copy | source/retained SHA match | REVERT; custody receipt |
| `.omx/state/active_lane_dispatch_claims.md` | +16/-0 | MAIN QBR1 six-cell burn, active scorer and Metal claims | none | live board and process state | HOLD(live QBR1) |
| `.omx/state/canonical_equations_registry.jsonl` | +7/-0 | CD1 repair of `4c77d9db6e` lost update | none | C22; 464-entry list; JSON | LAND bundle / `19f43a580e` |
| `.omx/state/current_focus.md` | +3/-3 | MAIN G8V1 harvested state | none | state cross-read; diff check | LAND bundle / `ed0ec9200a` |
| `.omx/state/lane_registry.json` | +336/-0 | HT1 red-debt hygiene | `09b100b3...a24`, SAME | R501; JSON | LAND bundle / `b0d64fdcb0` |
| `.omx/state/modal_call_id_ledger.jsonl` | +4/-0 | MAIN G8V1 harvest | none | JSON | LAND bundle / `ed0ec9200a` |
| `.omx/state/probe_outcomes.jsonl` | +60/-0 | LV3 generation 3 chain | `44e045f8...b456`, SAME | JSON | LAND bundle / `62ca828719` |
| `experiments/ddm_cpu1_gt_lineage_attribution.py` | +3/-3 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `experiments/ddm_jg2_tail_reencode.py` | +360/-46 | HT1 coder-pile checkpoint/payload cure | `e0a4ea...e1c0`, SAME | J46; PC18; two review marks | LAND bundle / `7235ad6d20` |
| `experiments/ddm_msr1_edge_contrast_probe.py` | +1/-1 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `experiments/ddm_msr1_manufactured_seg_characterize.py` | +2/-2 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `experiments/ddm_msr1_zero_byte_reach_bound.py` | +1/-1 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `experiments/ddm_rt2_mechanism_decomposition.py` | +1/-1 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `experiments/ddm_t1h_compose_pass1.py` | +3/-3 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `experiments/modal_click_polish_cpu.py` | +1/-1 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `reports/latest.md` | +2/-2 | MAIN G8V1 harvested state | none | state cross-read; diff check | LAND bundle / `ed0ec9200a` |
| `src/tac/tests/test_contest_auth_eval.py` | +2/-2 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py` | +4/-1 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `src/tac/tests/test_ddm_jg2_tail_reencode.py` | +132/-7 | RXC1/HT1 JG2 cure tests | no exact prior bundle | J46; PC18; two review marks | LAND bundle / `7235ad6d20` |
| `src/tac/tests/test_preflight_representation_integration_gates.py` | +2/-9 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `src/tac/witness_dsl/tests/test_taskspace_inverse_stack_receipt.py` | +5/-4 | CD1 stale competitive-target fixture | none | isolated pass, C22 companion, PC18; two review marks | LAND bundle / `19f43a580e` |
| `tests/test_ddm_jg4_reencoder_resume_fidelity.py` | +8/-3 | CD1 follow-up to JG2 v3 checkpoint schema | none | J46; PC18; two manual passes (tracker excludes root `tests/`) | LAND bundle / `7235ad6d20` |
| `tools/plan_decoder_q_signed_waterbucket.py` | +2/-2 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `tools/probe_defect_network_rate_code.py` | +1/-1 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `tools/register_prereg.py` | +1/-1 | HT1 enforcement | `09b100b3...a24`, SAME | R501; PC18; two review marks | LAND bundle / `b0d64fdcb0` |
| `.omx/research/ddm_ht1_red_debt_hygiene_verdict_20260901.md` | +90 lines/12,199 B | HT1 owner verdict | `bcddefbd...ea4a`, SAME | owner suite in R501; document review | LAND bundle / `6da8d2cfa6` |
| `.omx/research/ddm_lv3_recursive_leverage_20260901/equations_table_generation_001.json` | +19 lines/2,142 B | LV3 generation 1 | `bfd48be3...afb`, SAME | JSON; chain receipt | LAND bundle / `62ca828719` |
| `.omx/research/ddm_lv3_recursive_leverage_20260901/generation_001.json` | +25 lines/1,393 B | LV3 generation 1 | `bfd48be3...afb`, SAME | JSON; chain receipt | LAND bundle / `62ca828719` |
| `.omx/research/ddm_lv3_recursive_leverage_20260901/generation_002.json` | +24 lines/1,448 B | LV3 generation 2 | `474f3fd6...d1d`, SAME | JSON; chain receipt | LAND bundle / `62ca828719` |
| `.omx/research/ddm_lv3_recursive_leverage_20260901/generation_003.json` | +25 lines/1,515 B | LV3 generation 3 | `44e045f8...456`, SAME | JSON; chain receipt | LAND bundle / `62ca828719` |
| `.omx/research/ddm_lv3_recursive_leverage_20260901/recall/generation_001.json` | +23 lines/1,060 B | LV3 generation 1 | `bfd48be3...afb`, SAME | JSON; chain receipt | LAND bundle / `62ca828719` |
| `.omx/research/ddm_lv3_recursive_leverage_20260901/recall/generation_002.json` | +19 lines/1,088 B | LV3 generation 2 | `474f3fd6...d1d`, SAME | JSON; chain receipt | LAND bundle / `62ca828719` |
| `.omx/research/ddm_lv3_recursive_leverage_20260901/recall/generation_003.json` | +19 lines/1,212 B | LV3 generation 3 | `44e045f8...456`, SAME | JSON; chain receipt | LAND bundle / `62ca828719` |
| `.omx/research/ddm_lv3_recursive_leverage_20260901/solver_reprice_matrix_generation_001.json` | +93 lines/7,268 B | LV3 generation 1 | `bfd48be3...afb`, SAME | JSON; chain receipt | LAND bundle / `62ca828719` |
| `.omx/research/ddm_lv3_recursive_leverage_verdict_20260901.md` | +161 lines/10,705 B | LV3 final handoff | `4839e960...13c`, SAME | final chain receipt; document review | LAND bundle / `62ca828719` |
| `.omx/research/ddm_fpc3_chunked_n600_trainer_20260903.md` | +170/-0 | live FPC3 at census start | concurrent MAIN landing | owner reported 12 tests | LAND on HEAD / `7aada29099` |
| `experiments/semantic_joint_ctxmix_pipeline.py` | +44/-7 | live FPC3 at census start | concurrent MAIN landing | owner reported 12 tests | LAND on HEAD / `7aada29099` |
| `src/tac/semantic_pipeline/pipeline.py` | +607/-120 | live FPC3 at census start | concurrent MAIN landing | owner reported 12 tests | LAND on HEAD / `7aada29099` |
| `src/tac/semantic_pipeline/stages/train.py` | +936/-17 | live FPC3 at census start | concurrent MAIN landing | owner reported 12 tests | LAND on HEAD / `7aada29099` |
| `src/tac/tests/test_semantic_pipeline.py` | +161/-2 | live FPC3 at census start | concurrent MAIN landing | owner reported 12 tests | LAND on HEAD / `7aada29099` |
| `experiments/ddm_gc1_generator_capacity_control.py` | +1,580 lines/64,955 B | live GC1 | none | intentionally not run or reviewed by CD1 | HOLD(live GC1) |
| `tests/test_ddm_gc1_generator_capacity_control.py` | +139 lines/4,996 B | live GC1 | none | intentionally not run or reviewed by CD1 | HOLD(live GC1) |
| `experiments/ddm_gf2_static_dynamic_generator_form.py` | +841 lines/34,459 B | GF2 appeared during census | concurrent MAIN landing | owner serializer landing | LAND on HEAD / `ab93088002` |
| `experiments/tests/test_ddm_gf2_static_dynamic_generator_form.py` | +104 lines/3,958 B | GF2 appeared during census | concurrent MAIN landing | owner serializer landing | LAND on HEAD / `ab93088002` |

The marker custody receipt is
`/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/reverted_pq1_nested_markers/custody_receipt.json`.
It retains `magnitude_dismissal_marker.json` as 133 B / SHA-256
`1104184834f366e6311b55530f0eb5c50bf86a6fde0d7cb2a9d66de37fa9ea53`
and `triality_drift_marker.json` as 133 B / SHA-256
`f0a28535ebecd5588c3fffc8ee0155e5565e6751bfa0b12b3e0e7df14fa16b91`.

## Verification ledger

- Full witness-DSL census inherited from MAIN: **1,324 passed, 35 failed, 16
  errors in 1,749.98 s**. The 341,872-byte log is retained at
  `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/witness_dsl_full_suite_20260903.log`,
  SHA-256 `fe4b1ca1ac94bf11186797e3557fc5acd3ab7f5ce4a35ec14117102e9ab2b08f`.
- Combined owner suites: **501 passed, 1 failed in 305.38 s**. The only failure
  was the stale JG4 v2 schema assertion; after the test was updated to the real
  v3 schema and `encoder_sha256` ledger key, JG2/JG4 passed **46/46 in 50.93 s**.
- Canonical Bregman/shearlet modules passed **22/22** after the seven lost rows
  were restored through their canonical population functions. The registry lists
  exactly 464 current equations. The changed-pointer test passes in isolation
  and in the combined run.
- Strict JSON/JSONL parsing covered 14 structured files and 30,343 records or
  documents. All 18 landable Python files passed `py_compile`; `git diff --check`
  passed.
- Ruff inspected all 18 landable Python files. Sixteen were clean. Its three
  `I001` findings are in `src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py`
  and `tools/probe_defect_network_rate_code.py`; the same findings reproduce on
  `HEAD`, so auto-sorting them would destroy owner-bundle byte identity without
  curing a regression from this landing.
- Fast preflight returned rc=0 and explicitly examined 0 gates in
  `fast --no-codebase` mode. This is the charter's fast-pass check, not a claim
  of full developer-gate coverage.
- Review tracker marks were recorded twice (`ddm_cd1_pass1`, `ddm_cd1_pass2`)
  for every Python path in its tracked roots. It does not ingest the PQ1 nested
  `.omx` Python file or root `tests/`; those two files received two manual source
  review passes plus compile/test evidence, and the serializer correctly ran
  without a Python override.
- No scorer, Modal call, Metal run, archive mutation, or exact evaluation was
  performed. `upstream/`, `submissions/semantic_joint_ctxmix/`, the QBR1 sealed
  source, and protected `direct_description_carrier_compose.py` were untouched.

## Test-drift ledger

| Cluster | Measured root cause | Disposition |
|---|---|---|
| Three named canonical “surfaces once” failures | Seven legacy rows were present before `4c77d9db6e` and disappeared in that commit while it added two unrelated equations. This is an append-only lost-update race, not a formula regression. | FIXED: invoked the official locked population functions for the five Bregman application rows, windowed curvelet row, and compact shearlet row. Each occurs once; C22 passes; bundle `19f43a580e`. |
| Stale competitive-target assertion | Fixture changed the upstream row to `0.16`, but AFR1 at `0.147976...` remained the effective minimum, so expecting `0.16` was false. | FIXED: fixture now changes the target to `0.10` and proves recomputation/conditional-ceiling movement; bundle `19f43a580e`. |
| JG4 checkpoint schema assertion, discovered by owner-suite run | JG2 correctly advanced from v2 to v3 to bind `encoder_sha256`; the old test demanded v2 and a five-key ledger. | FIXED: require v3 and the six-key ledger; J46 passes; bundle `7235ad6d20`. |
| EP725 renderer/oracle cluster: 30 affected tests | Frozen receipt pins renderer SHA `1cecaa3e...`; current renderer closure changed after `90d5377456` altered `tools/levelset_byte_close_and_eval.py` to use `safe_extract_zip`. Source inspection places the edit outside the canonical decoder call, but no new retained bounded decode/output-identity receipt exists. | OWED `PIN-REFRESH-WITH-RECEIPT` or explicit retired-lineage quarantine. A hash-only edit is forbidden. |
| V15 producer-custody cluster: 19 affected tests | Retained compile receipt pins `direct_description_carrier_compose.py` at 156,551 B / `3e1f69bb...`; protected current source is 160,470 B / `6fef110d...`. It moved through several commits, last `36f4b29476`, which adds non-vacuous apparatus proof. | OWED fresh V15 compile/materialization receipt proving exact archive behavior. Protected source was not edited and the old pin was not blessed. |
| PBR2/V9 source-manifest cluster: one affected strict-reopen test | Teacher receipt pins source manifest `92ab2350...`; current manifest is `715e3e57...`. The first observed drift is `2ed5794ed3`; later source changes touched direct-description, predictor-upgrade, and minimizer surfaces. | OWED exact PBR2/V9 rematerialization or owner-directed retired-lineage quarantine. No blind manifest rewrite. |

The 51 full-suite failure/error outcomes are fully explained by those four
witness-DSL roots: 30 EP725 renderer pins, 19 V15 producer pins, one stale
pointer fixture, and one PBR2/V9 manifest pin. No curriculum, lever, policy, or
live semantic-pipeline test appeared in the failing set. A post-fix full
1,375-test rerun was deliberately not claimed: the two receipt-gated roots remain
unchanged and the repaired fixtures were exercised directly.

## RECALL EVIDENCE

I searched the full corpus rather than the charter alone:

- canonical registry via `tools/list_canonical_equations.py --json` and content
  queries for the seven absent equation IDs;
- `.omx/research`, the canonical research index, `sub015_DAG_*` FEED blocks,
  design docs, task rows, `.omx/state/commit-serializer.log`, and both SSD roots
  with the queries `serializer fallback bundle blocked git`, `canonical equation
  registry append only concurrent`, `EP725 source drift renderer receipt`,
  `PQ1 nested .omx marker`, `LV3 recursive leverage`, and `HT1 red debt`;
- graph-memory default recall for those three first queries. Typed topic recall
  returned zero rows in that bounded surface; default recall found the HD1/FC1X
  bundle-import precedent, DD1 drift ledger, and registry append-only law.

Beyond the charter seeds, PQ1's tenth fresh-eyes review explicitly authorized
sweeping the nested markers; LV3's four-bundle chain receipt claimed exact final
workspace identity and survived independent byte comparison; and the registry
history exposed `4c77d9db6e` as the seven-row lost update. Those findings changed
the plan: markers became a custody-backed REVERT, LV3 became a byte-verified
bundle import rather than an origin guess, and registry repair used canonical
population functions rather than hand-authored JSON. FPC3 and GF2 landing during
the audit also changed their dispositions from live HOLD to landed-on-HEAD.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `main` plus the eight receipt directories under `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/serializer_fallbacks/`; fire trigger: a Git-index-writable MAIN session. Apply in this order: PQ12, HT1 enforcement, JG2/JG4, HT1 memo, LV3, AU1, MAIN state, test drift; verify each declared file SHA against its receipt and use the serializer, never reset `main` to the bundle parent.
- **HOLD(live-arm)** — owner: QBR1/MAIN; consumer store: `.omx/state/active_lane_dispatch_claims.md`; fire trigger: the six-cell QBR1 burn reaches a terminal receipt. Reconcile the active/stale claim rows, then serialize only the stable ledger delta.
- **HOLD(live-arm)** — owner: `ddm_gc1_generator_capacity_control`; consumer store: its owner verdict and `main`; fire trigger: GC1 writes its terminal arm message. Run the owner-named tests, two review passes, and serializer byte guards before landing its source/test pair.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: retired EP725 lineage maintainer; consumer store: the EP725 bounded decode receipt set under `.omx/research/original_taskspace_inverse_witness_codec_20260725/`; fire trigger: either a retained bit-exact double-decode/output receipt against the current safe-extract renderer exists, or the owner explicitly chooses quarantine. Refresh the pin only in the first branch; xfail with owning memo and drift commit only in the second.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: retired V15 lineage maintainer; consumer store: `.omx/research/original_taskspace_inverse_witness_codec_20260725/fresh_v15_semantic_base_n600_20260726/`; fire trigger: a fresh current-source V15 compile receipt and compiled semantic archive prove deterministic equality. Then refresh downstream source custody together; otherwise quarantine the retired lineage.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: retired PBR2/V9 lineage maintainer; consumer store: `.omx/research/original_taskspace_inverse_witness_codec_20260725/g1_teacher_atom_census_n64_20260726.json`; fire trigger: exact PBR2/V9 rematerialization closes pair window, packet, and current renderer manifest. Refresh the census receipt only after all three close; otherwise quarantine with `2ed5794ed3` named.

## LIVE-HYPOTHESES

- EP725 may reproduce identical decoded outputs under the current renderer because
  `90d5377456` changed extraction hardening rather than the canonical decoder
  arithmetic. That makes a receipt-backed pin refresh plausible, but untested.
- V15 may regenerate its exact semantic archive under current source because the
  last observed source move adds validation/apparatus proof rather than an
  advertised rendering change. Only a fresh compile receipt can establish this.
- PBR2/V9 may retain its n64 packet/output identity under the current 13-source
  manifest if the intervening direct-description changes are validation-only for
  this fixture. The manifest hash alone cannot decide it.
- The eight CD1 fallback units should apply without GF2 conflicts because their
  declared paths are disjoint from `ab93088002`; this remains to be tested by a
  Git-writable serializer consumer against the now-current `main`.

## DEAD-ENDS

- Blindly replacing an old source/renderer SHA with the current SHA is closed:
  all three witness roots require output-bearing regeneration or quarantine.
- Treating serializer `rc=17` as a merged commit is closed: every CD1 intended
  commit is retained, but none is in `HEAD`.
- Auto-sorting the three inherited Ruff findings is closed for this landing:
  it would break exact owner-bundle identity without fixing introduced drift.
- Committing QBR1's active claim ledger or GC1's live source is closed until
  their owner fire triggers; doing so would absorb live-arm state.
- Deleting the PQ1 nested markers without custody is closed: exact bytes and
  hashes now live under the CD1 SSD receipt.
- The ≥80% bundle-match prediction is closed at 27/39 (69.23%); successors
  should not assume every stale artifact came through the serializer.

OWN-VEHICLE FRONTIER UNMOVED: AFR1 — S `0.14797617125559104` @ `180,002 B` `[contest-CUDA T4 n600]`, archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
