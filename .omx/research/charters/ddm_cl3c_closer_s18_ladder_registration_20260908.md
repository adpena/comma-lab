# ddm_cl3c — cl3 bookkeeping closer: s18 decode identity + section census, ladder report completeness, task registration (charter, 2026-09-08)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm · Spawned by MAIN 2026-09-08 under the operator's standing GO. Predecessor: `ddm_cl3` (Opus), stopped by the operator mid-bookkeeping on 2026-09-06 with its VERDICTS FINAL (capacity axis closed both directions; seeds at the noise floor; no candidate). This arm CLOSES the books; it does not reopen the ladder. `score_claim=false`; no scorer runs; no Metal.

## MANDATE

Operator 20260908: *"recover and respawn and continue with all, also codex is back"*. cl3's memo `.omx/research/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md` records the verdicts and ITEMs 1–5, but three closing steps were interrupted: (1) the `lambda_1p0_s18` seed rung (J +137 B, row in the memo's ladder table) lost its determinism proof to the arm's own `--reuse-encodes` defect (commits `b9f72faee`, `58bf07be0`) and its receiver-copy decode identity + section census were not re-run after the fix; (2) `RC1_LADDER_REPORT.json` on the tree carries no s18 row (`'s18' not in report` — MEASURED at spawn); (3) ITEMs 1–5 were never registered in the canonical task ledger. Close all three with receipts.

## SCOPE

1. Recall: the cl3 memo (whole), `RUNBOOK.md`, `price_rung.sh`, `LADDER_REPORT.json` + `RC1_LADDER_REPORT.json` under `/Volumes/VertigoDataTier/pact/ddm_cl3_hpac_smaller_prior_and_seed_selection/`, and cl2's pricer `experiments/ddm_cl2_hpac_prior_capacity_ladder.py` (+ cl3's live-tree pin patcher). `tools/subagent_checkpoint.py read --subagent-id ddm_cl3` for the predecessor's last `next_action`.
2. **s18 admissibility.** On the s18 rung's retained streams: (a) two encodes byte-identical (if only one exists, run the second through the FIXED path — this is a CPU pricer, ~1 h; detached if > 30 min); (b) receiver-copy decode returns the exact field; (c) section census `only_model_and_stream_moved=True` with semantic, carrier, residual table byte-identical to the control. Record J (model + stream bytes) beside the memo's +137 B. If any leg fails, the row is INADMISSIBLE — say so; the verdict does not change (it lost on bytes either way), but the ladder must not carry an unproven row.
3. **Ladder report.** Regenerate `RC1_LADDER_REPORT.json` (and the Brotli-basis `LADDER_REPORT.json` if the tool emits both) so every rung the memo tables carries a row with admissibility flags; `best_beats_live_pointer` must read false against the CURRENT pointer (181,645 B, S 0.13900437796841966 — the bar moved since cl3 ran; recompute with the live pointer's hpac container bytes, verified at source in `candidate_pass3/candidate_runtime`).
4. **Task registration.** `tools/extract_canonical_tasks_from_directive.py --directive .omx/research/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md --register-all --owner ddm_cl3c` (dry-read the sections first: ITEMs 1–5 must each become one row; report the count registered and the ids WITH the memo filename beside each — the m89 split).
5. Append a dated `## CLOSER (2026-09-08)` section to the cl3 memo (append-only; headline numbers in the body, not just the header) and write your own short memo `.omx/research/ddm_cl3c_closer_20260908.md`.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO scorer. NO Metal/MPS (nothing trains here). Never edit `submissions/semantic_joint_ctxmix/`, the live pointer tree, or `experiments/ddm_sj1_*` (ddm_sj1 owns them).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes. Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer.
- ALWAYS KEEP THE PAYLOAD: retain the second encode's stream + sha under the cl3 tree's `rungs/lambda_1p0_s18/retained/` (Vertigo has ~18 GiB free — the stream is ~120 KB; check `df` first). Both SSDs are near full: create nothing bulky; parse-back renders are NOT needed for this closer.
- VERIFIED-AT-SOURCE LAW: the +137 B row, the section byte counts (12,416 / 113,483 / 125,899), the live pointer's container bytes — mark `verified-at-source:` or re-measure.
- EQUATIONS-LEG LAW: if s18 is admissible, append it as an anchor on `coder_strength_substitutes_for_capacity_v1` / `hpac_prior_capacity_slope_v1` ONLY through `update_equation_with_empirical_anchor` (`tac.canonical_equations`), never by hand; if it is inadmissible, the memo says `# FORMALIZATION_PENDING:s18 row inadmissible — no anchor appended`. Run the Catalog #344 check before the final message.
- DETACHED >30-MIN COMPUTE: the second encode may exceed 30 min — `tools/launch_detached_process.py --output-dir <run_dir> --done-receipt ddm_cl3c_s18_encode.done --nice 10 --nice-best-effort -- <cmd>`; ≤ 2 threads (the frontier arm's shards need the CPU); artifact-bound waits ≤ 780 s; never `nohup`/`&`/clock waiters.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_cl3c …` every ~10 tool uses; `read` first.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- cl2's pin patcher hardcodes fs2's sha and an underscore byte format and refused the live tree AFTER an hour of encoding — cl3 memo "my own pin-patcher defect" (commit `6991740f3`, cure `f3eb816d4`); use cl3's live-tree patcher, and verify its pins against the CURRENT tree before encoding.
- `--reuse-encodes` guarded on `encodes` non-empty and killed the SECOND encode on fresh runs (commit `b9f72faee`) — s18's determinism proof was lost to exactly this; run the fixed path and confirm both encodes are present before comparing.
- The capacity axis is CLOSED both directions (λ 1.0→2.0 +224 B; cl2's 1.0→0.5 +506 B) and seeds sit at a 137 B spread against a 41,777 B demand — cl3 memo VERDICT + memory `coder_strength_substitutes_for_capacity_v1`. This arm does NOT run λ=4.0 or new seeds; reopening is out of scope.

## OPTIMAL FORM

- Family exemplar: cl3's admissible λ=2.0 row, commit `c963db895` (two encodes byte-identical, receiver decode returns the exact field, census proves only model+stream moved) — the reference form for an admissible ladder row; ladder report form `56d50ca99`; receipt path `.omx/research/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md`.
- SCOPE reductions declared per row: one rung (s18) is re-proved — the other rungs' admissibility already stands (SCOPE). MECHANISM reductions FORBIDDEN: no "decode identity" through a library shortcut that skips the receiver-copy path; no ladder row without all three legs.
- **PRIOR-LAW PREDICTION (falsifiable):** s18 is ADMISSIBLE (twin byte-identical, exact-field decode, census clean) and its J lands +137 ± 0 B vs the memo (the streams are retained; determinism means the number does not move). FALSIFIER: the twin differs or J moves by ≥ 1 B — count it plainly; the row becomes INADMISSIBLE and the seed-spread claim (137 B, n=3) must be re-stated at n=2.

## DELIVERABLE

`.omx/research/ddm_cl3c_closer_20260908.md` + the appended `## CLOSER` section in the cl3 memo + regenerated ladder report(s) on the tree: rows — rung · J · legs (twin / decode / census) · admissible · registered task ids with memo filename · NEXT_IF_RESUMED. Commit via the serializer. End with `sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]` (unchanged by this arm).
