# ddm_cl3 — the SMALLER-model direction and seed selection of the HPAC prior on the shipped object (charter, 2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-05 ~16:05Z · Axis of every number here:
`[macOS-CPU advisory / scorer-free EXACT byte measurement]`, `score_claim=false` until a T4 row exists.

## Why this arm exists (recall, not volition)
cl2 (`.omx/research/ddm_cl2_hpac_prior_capacity_ladder_on_shipped_object_20260905.md`) priced the HPAC prior-capacity ladder on the shipped
fs2 mixer with the token FIELD held bit-identical (d_seg / d_pose held by construction). Measured rows (`/Volumes/VertigoDataTier/pact/
ddm_cl2_hpac_prior_capacity_ladder/LADDER_REPORT.json`): shipped J = 126,926 (stream 113,411) · λ=1.0 re-trained control J = 126,885
(stream 113,419, archive 179,982 B = **−41 B**, twin-reproduced byte-exactly, sealing now) · λ=0.5 (BIGGER model, +350 B model) J = 127,391
(stream +156 B) → the bigger direction is FALSIFIED (secant ΔB_stream/ΔB_model = +0.446 vs the −1 the law needs). Law registered:
`hpac_prior_capacity_slope_v1` (rung pays iff ΔB_stream/ΔB_model < −1).
**Untested:** the SMALLER-model direction (λ > 1) and seed selection at λ = 1.0. Both are the same axis (`[fx1]` MODEL AXIS LIVE: coder-SEARCH
closed ≠ MODEL closed) and both are rate-only, distortion-held moves — the cheapest kind of exact row we have.

## PRIOR-LAW PREDICTION (owed before any measurement; m38)
From the measured λ 1.0→0.5 secant (+350 B of model bought +156 B of stream — capacity is already past the point where more prior helps):
- **λ = 2.0:** model ≈ −250…−400 B; stream tax +0…+200 B → **net −350…−50 B, predicted to PAY** (ΔB_stream/ΔB_model < −1).
- **λ = 4.0:** model ≈ −500…−700 B; stream tax +200…+900 B → **the falsifier boundary**; either sign admissible.
- **Seeds at λ = 1.0 (20260717, 20260718 vs the control's 20260716):** the control's −41 B is training noise of that order; min-of-3 ≈
  **−40…−90 B beyond the control**. If the three seeds land within ±20 B of one another the seed lever is at its floor.
- **FALSIFIER (whole-axis):** if λ=2.0 nets ≥ 0 B, the capacity axis is CLOSED IN BOTH DIRECTIONS on the shipped object (one instance = the
  axis on this vehicle; do not run λ=4.0 in that case — record it as not-run-because-falsified).
Write the measured numbers next to these lines in the memo; residuals go to the law's anchors.

## What to do (cl2's recipe, nothing re-implemented)
1. RECALL first: read cl2's memo + charter (`.omx/research/charters/ddm_cl2_*.md`), `RUNG_START.json` / `RUNG_RESULT.json` /
   `VERIFY_RESULT.json` of `rungs/lambda_1p0` and `lambda_0p5`, and `experiments/ddm_cl2_hpac_prior_capacity_ladder.py`
   (sha `e3153943fe34239c…`). `tools/subagent_checkpoint.py read --subagent-id ddm_cl3` before anything else.
2. Rungs, SERIAL on Metal (`device mps`, profile `cl2_shipped_ladder`, 60-epoch cosine, batch 8, QAT 0.5, warm start = the shipped ep634 EMA
   init `ff2d3e45…`, cache `f29c479a…`, field `cc10a7b0…` — the exact cl2 inputs under `…/ddm_cl2_hpac_prior_capacity_ladder/inputs/`):
   `lambda_2p0` (λ=2.0, seed 20260716) → `lambda_1p0_s17` (λ=1.0, seed 20260717) → `lambda_1p0_s18` (λ=1.0, seed 20260718) →
   `lambda_4p0` only if λ=2.0 paid. Extend `RUNG_LAMBDA` (and add a SEED check beside the `rate_lambda` check at ~line 337) so `price --rung`
   and `verify --rung` accept the new rung names; nothing else in the pricing path changes.
3. Per rung: train → `price` (pack + Brotli race + stage + encode TWICE, streams byte-identical) → `verify` (receiver-copy decode → field
   identity byte-for-byte + decode wall-clock vs the shipped 1,494.5 s). Record J, ΔB_model, ΔB_stream, ΔJ vs shipped 126,926 AND vs the
   cl2 control 126,885.
4. Winner = min J over {cl2 control, new rungs}. If the winner is NOT the cl2 control: twin it (a fresh-root retrain of the exact winner
   law; the twin's stream and archive shas must match — cl2 did this for its control) and produce the contest-CUDA seal via
   `tools/make_candidate_seal.py` (single-axis waiver reason stated). **Do NOT dispatch Modal.** MAIN fires T4 from your seal.
5. Memo `.omx/research/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md` (prediction lines with measured residuals · verdict_scope on
   every negative · frontier line last); anchors on `hpac_prior_capacity_slope_v1` (new EmpiricalAnchor rows, no new law unless the seed
   result is a distinct law); lane `lane_ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905` (L2 when byte-closed); measured peaks recorded;
   anything left owed goes into a `## ITEM n — …` section of your memo and is registered with `tools/extract_canonical_tasks_from_directive.py --directive <memo> --register-all --owner ddm_cl3`.

## OPTIMAL FORM
Reference form = cl2's ladder itself (full n600 field, 60-epoch cosine to the terminal QAT checkpoint, the exact shipped pack/stage/encode/
decode path, Brotli q0..q11 race, twin verification). Deltas: NONE at mechanism level; SCOPE = four rungs. Provenance pins: script sha
`e3153943fe34239c…`; trainer `tools/train_ddm_cl1_hpac_capacity.py`; inputs by sha above; fs2 fire tree as staged by cl2. A rung with
fewer epochs, a subset field, or a different packer is a TOY — refuse it at the typing moment.

## Compute, memory, disk, resumability (binding)
- Each cl2 rung measured 53–63 min wall at ≤ 2.4 GiB declared peak (`.omx/state/measured_peaks.jsonl`, families
  `train_ddm_cl1_hpac_capacity` / `ddm_cl2_hpac_prior_capacity_ladder`). md3's 49.6 GiB Metal cell is LIVE until ~19:00Z; a 2.4 GiB
  trainer is admitted beside it — run ONE rung at a time (cl2's overlapping peaks were graded CONFOUNDED). Launch every stage through
  `tools/launch_detached_process.py` with a distinct `--done-receipt` (`.omx/tmp/codex_runs/ddm_cl3_<stage>.done`); poll receipts with a
  background until-loop, never a foreground wait > 3 min (rc=144 reaper).
- KEEP THE PAYLOAD: every checkpoint, raw IHS1, every Brotli representation, both streams, candidate archive, decoded field, with
  sha256 + bytes. Disk: Vertigo has 29 GiB free, APDataStore 49 GiB — put new rungs under
  `/Volumes/APDataStore/pact/ddm_cl3_hpac_smaller_prior_and_seed_selection/rungs/<rung>/` (a rung retains ~130 MB; a 3.66 GB parseback
  render only for the WINNER, on APDataStore). Never delete; certify-or-block.
- Resumable from disk: per-stage checkpoints already exist in the trainer; `tools/subagent_checkpoint.py` every ~10 tool uses.
- Commits ONLY via `tools/subagent_commit_serializer.py --message "… [no-triality] [p0-ledger-ok]" --files … --expected-content-sha256
  <file>=<post-edit sha>`; `.py` files get two visible review passes (`tools/review_tracker.py mark-file`); NO co-author trailer; no
  `/tmp` paths in any artifact.
- Read `docs/operating_manual_craft_handoff.md` and CLAUDE.md before starting. Label every number MEASURED / DERIVED / PREDICTED.
  End your final message with the own-vehicle frontier line: `fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]` (plus any
  advisory candidate line, clearly labeled advisory).
