The charter result is **CEILING-PASS_GATE-BLOCKED**.

The selected warm-lineage W64 candidate is 166,465 bytes, saving 13,921 bytes versus fx5 and clearing the required 12,155-byte ceiling by 1,766 bytes. This is a necessary rate result, not a score claim: no training or scorer ran, and the frontier remains 0.14823186109359.

Under full Seg elimination, the arithmetic projection is 0.11882344, leaving only 0.00117656 score units for realized Seg/Pose degradation. Metal was unavailable, CPU fallback was forbidden, and scorer ownership remained with the active fleet lane.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_wd4_warm_lineage_width_20260821.md)
- [Builder and resumable gate](/Users/adpena/Projects/pact/experiments/ddm_wd4_warm_lineage_width.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_wd4_warm_lineage_width.py)
- [Machine-readable result](/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/RESULT.json)
- Selected archive SHA-256: `29c26e428647a8feb1a9614e55be3d12a88a66351443f7f98de23622982f7a06`
- Retained custody: 549 files, 12,941,268 logical bytes
- Commits: `9a9ef1310c43a16988281b9fbc1f66fa5e4831d1`, `69eee24142abe4fd77648386015ab9d387d60d11`
- Verification: 34 relevant tests passed; Ruff, compilation, secret scan, parse-back, deterministic-repeat, payload-hash, and two review-tracker passes succeeded. Unrelated dirty work was preserved.

## NEXT_IF_RESUMED

- **BLOCKED-PRELAUNCH** — Owner: `ddm_wd4_warm_lineage_width`; consumer store: `/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/retained/warm_gate_training/`; fire trigger: a Metal-visible host, passing storage/governor preflight, and a unique active local lane claim. Run the retained resumable 32-step W64 gate.
- **QUEUED-SCORER** — Owner: MAIN after explicit scorer ownership; consumer store: `/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/`; fire trigger: JO1 terminal, a unique scorer lane, and the first retained warm-training checkpoint. Reject early if matched stratified-n32 degradation exceeds 0.0017648420892356284.
- **CONDITIONAL-N600** — Owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wd4_warm_lineage_width/scorer_gate_n600/`; fire trigger: the n32 gate passes. Promote only if n600 realized degradation is at most 0.0011765613928237523; otherwise record `GATE-FAIL` and stop this arm.

## LIVE-HYPOTHESES

- Warm W64 may preserve the conditioned basin because it inherits the exact current SM3R state and removes complete eight-channel normalization groups instead of training a narrower vehicle from scratch.
- Group-salience selection may preserve quality better than a prefix at the same width because it retains stronger current-weight participation while keeping GroupNorm neighborhoods intact.
- A bounded low-learning-rate warm window may recover slicing damage because it uses the exact fx5 teacher, MC36 tokens, EMA, and byte-closed checkpoints; this remains untested because Metal was unavailable.

## DEAD-ENDS

- e960 is not the semantic ancestor; it selected the HPAC probability model.
- Old WANS/e480b byte tables do not transfer to fx5, whose current semantic representation is SM3R.
- Tested widths 72, 80, and 88 cannot clear the 12,155-byte ceiling; W72 saves only 10,879 bytes.
- Fresh D56/F64 failures do not decide this warm-lineage arm, but repeating those exact fresh-start configurations is closed because they degraded score by roughly 1.1–1.5.
- Individual-channel salience reordering is closed because it disrupts inherited GroupNorm neighborhoods; its retained runtime is explicitly superseded.
- CPU fallback is closed by the charter, and scorer or paid dispatch is closed until explicit ownership and lane triggers are satisfied.