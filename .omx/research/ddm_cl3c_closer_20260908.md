# ddm_cl3c — s18 and ladder bookkeeping closer

Tokens: `[no-triality] [p0-ledger-ok]`  
Charter: `.omx/research/charters/ddm_cl3c_closer_s18_ladder_registration_20260908.md`  
Axis: `[macOS-CPU advisory / scorer-free EXACT byte measurement]`  
Score claim: `false`

Status: **COMPLETE**. The detached fixed-path receiver-copy replay exited 0 at
2026-09-08T23:42:19Z and regenerated the s18 result from the retained encodes.

## Conclusion

The s18 seed rung is admissible at **J = 12,416 + 113,483 = 125,899 B**, or **+137 B**
against the historical pc1/rc1 ladder control. It therefore cannot replace that control. The current
frontier is a later sj1 object with a different token field, so s18's 174,923-byte archive is not a score
comparison and `best_beats_live_pointer` must remain false. No scorer ran, no candidate was sealed, and
this closer moved no pointer.

## RECALL EVIDENCE

Read in full: `PROGRAM.md`; the governing `AGENTS.md`/`CLAUDE.md`; the charter and common contract;
the cl3 memo; the cl3 `RUNBOOK.md`, `price_rung.sh`, `LADDER_REPORT.json`, and
`RC1_LADDER_REPORT.json`; cl2's pricer; cl3's live-tree pricer/pin patcher; and the predecessor's final
checkpoint (`tools/subagent_checkpoint.py read --subagent-id ddm_cl3`).

The bounded corpus recall used these exact searches:

```text
rg -n --glob '*.md' --glob '*.json' --glob '*.jsonl' 'ddm_cl3|lambda_1p0_s18|coder_strength_substitutes_for_capacity_v1|hpac_prior_capacity_slope_v1' .omx/research .omx/state
rg -n --glob '*.md' --glob '*.py' 'capacity coordinate|frame_dim|receptive.field|third container|strong coder|seed spread' .omx/research src/tac/canonical_equations experiments
rg -n 'HPAC|capacity|seed|coder_strength' .omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md .omx/research/canonical_research_index.md
```

Beyond the charter seeds, this found three plan-relevant facts:

1. sj1 is a successor object with the same 12,343-byte HPAC section but a different field and a
   120,225-byte stream; archive bytes alone cannot compare cl3 rungs to the current score.
2. `ddm_ds1_cheap_to_shrink_objective` already classifies the HPAC model as a lossless rate-only object;
   it supplies no honest distortion objective that could rescue the multiplier rung.
3. The research index and DAG contain broader capacity-response hypotheses, but no current exact-object
   evidence that reopens λ=4 or seed selection. Only a different capacity coordinate remains open.

## Source-verified controls

- Historical ladder control: pc1/rc1 archive **174,786 B**, sha256
  `1de6c5d7186a0b31e5cc085bb6d2baab8275ee0d9de4d509f4d8add13695a629`; HPAC **12,343 B**;
  stream **113,419 B**; J **125,762 B**. The retained cl2 twins, receiver decode, and census supply all
  three admissibility legs.
- Current pointer: sj1 archive **181,645 B**, sha256
  `06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3`; HPAC **12,343 B**;
  stream **120,225 B**; J **132,568 B**. The canonical pointer and archive were both re-read.

## Complete RC1 ladder

| rung | model B | stream B | J B | ΔJ vs ladder control | twin | receiver-copy exact field | census only model+stream moved | admissible |
|---|---:|---:|---:|---:|:--:|:--:|:--:|:--:|
| control λ=1.0 / seed 20260716 | 12,343 | 113,419 | **125,762** | 0 | PASS | PASS | PASS | **YES** |
| λ=2.0 / seed 20260716 | 11,886 | 114,100 | 125,986 | +224 | PASS | PASS | PASS | YES |
| λ=4.0 | — | — | — | — | — | — | — | **NOT RUN; falsifier fired** |
| λ=1.0 / seed 20260717 | 12,375 | 113,416 | 125,791 | +29 | PASS | PASS | PASS | YES |
| λ=1.0 / seed 20260718 (s18) | **12,416** | **113,483** | **125,899** | **+137** | PASS | PASS | PASS | **YES** |

The s18 preregistration predicted admissibility and +137 ± 0 B. The replay measured section counts
**12,416 / 113,483 / 125,899 B**, all three legs passed, and the residual is **0 B**. Receiver-copy decode
took 1,575.82 s and returned the exact 117,964,800-byte field sha256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. The census measured semantic
30,246 B, carrier 18,568 B, and residual table 96 B byte-identical to the control.

## Payload custody

The second encode is a fresh frame-0-to-599 RC64 pass, **113,483 B**, sha256
`5f0018581ad15f6303fbfe74cb5c097cbf66684c8c3db2cc1a7256b04dfe88c8`. It is retained at
`/Volumes/VertigoDataTier/pact/ddm_cl3_hpac_smaller_prior_and_seed_selection/rungs/lambda_1p0_s18/retained/token_stream.second_encode.rc64.bin`
beside `SECOND_ENCODE_RECEIPT.json`. Its twin has the same bytes and sha.

## Ladder report and equations

`experiments/ddm_cl3_rc1_rung_price.py report` now emits RC1 report schema v2 with every memo rung,
all three admissibility flags, a source-verified historical ladder control, and a separately source-verified
current sj1 pointer. The tool emits only `RC1_LADDER_REPORT.json`, so the old Brotli-basis
`LADDER_REPORT.json` is intentionally not rewritten.

Because the receiver proof passed, s18 was appended through `update_equation_with_empirical_anchor` only.
The two source-defined equations were absent from the durable registry, so their source builders were
registered first; then the helper appended exactly one new anchor to each:

- `coder_strength_substitutes_for_capacity_v1`: the same s18 weights are +40 B on the weak Brotli model
  container versus +137 B on rc1, anchor
  `cl3c_s18_two_container_seed_price_admissible_20260908` and a second two-container observation.
- `hpac_prior_capacity_slope_v1`: fixed λ=1.0 seed scale is 137 B at n=3 with a deterministic 0 B
  same-seed noise floor, anchor `cl3c_s18_fixed_lambda_seed_scale_exact_reproof_20260908`; this is not
  a new multiplier secant.

## Canonical tasks

The exact extractor registered **five** rows from
`.omx/research/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md` under owner `ddm_cl3c`:

| task id | disposition |
|---|---|
| `ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905::ITEM_1` | COMPLETED — λ=4 correctly not fired |
| `ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905::ITEM_2` | PENDING — different capacity coordinate |
| `ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905::ITEM_3` | PENDING — substitution-law promotion anchor |
| `ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905::ITEM_4` | PENDING — pin-validation class fix |
| `ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905::ITEM_5` | COMPLETED — seed selection closed on scale |

The extractor also surfaced pre-existing bounded warnings, including two unreadable historical task IDs;
they were not created or repaired by this closer.

## Verification

- Both modified Python files passed two genuine `review_tracker.py` review/mark cycles after the final edit.
- `ruff check`, `py_compile`, and the combined pricer/equation/Catalog test selection pass: **47 passed**.
- Catalog #344 strict check passes with **0 violations** after both append-only equation updates.
- Commit custody is via `tools/subagent_commit_serializer.py` with post-edit SHA-256 for every file;
  the commit hash is reported by the final handoff rather than self-referentially embedded here.

## NEXT_IF_RESUMED

- ITEM_2 — disposition `QUEUED`; owner `ddm_cl3c`; consumer store
  `.omx/state/canonical_task_status.jsonl` plus a distinct retained rung under the cl3 SSD tree; fire only
  when one width, `frame_dim`, depth, or receptive-field coordinate has a preregistered exact-J prediction
  on the then-current receiver object. This does not reopen the rate multiplier.
- ITEM_3 — disposition `DEFERRED-PENDING-RESEARCH`; owner `ddm_cl3c`; consumer store
  `.omx/state/canonical_equations_registry.jsonl`; fire on a third model container for the same weights or
  the same two containers for a non-multiplier capacity coordinate.
- ITEM_4 — disposition `QUEUED-CLASS-FIX`; owner `ddm_cl3c`; consumer store
  `experiments/ddm_cl2_hpac_prior_capacity_ladder.py` and its tests; fire when no pricing job is active and
  a new receiver tree needs pin validation, before any encode starts.

## LIVE-HYPOTHESES

- Width, `frame_dim`, depth, or receptive-field shape may buy useful capacity even though the trainer's
  rate multiplier does not, because none of those structural coordinates was varied by cl2/cl3.
- Model-coder strength may continue to substitute for capacity across a third container or structural
  coordinate, because both measured weight changes spend the same model-redundancy pool.
- Validating archive pins before encoding should remove the delayed-failure class without changing the
  receiver proof, because the pins are static inputs already present before the first frame is encoded.

## DEAD-ENDS

- λ=4 on this formulation: λ=2 already lost +224 B, firing the preregistered stop rule.
- More seeds under the same law: n=3 spans only 137 B, the incumbent is already best, and the observed
  scale is immaterial to the rate demand.
- Treating s18's smaller archive as a frontier win: s18 restores pc1's older field, so distortion is not
  held against sj1 and no score comparison exists.

`sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]` — unchanged by this arm.
