# ddm_rlc3 — resume ddm_rlc2 under the FROZEN pre-fire intent contract: re-base rlc1's cure onto move 42 and produce a committed intent (charter, MAIN 2026-09-10)

## What changed since rlc2 stopped
The seal-before-fire cycle is resolved: pr12's contract is implemented and frozen (commit a47543199;
receipt `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`; 345 tests on the host).
Surfaces: `src/tac/candidate_seal.py` (`PREFIRE_INTENT_SCHEMA = "candidate_prefire_intent.v1"`,
`PREFIRE_RISK_SCHEMA = "candidate_prefire_timing_risk.v1"`, `build_prefire_intent`, `validate_prefire_intent`,
`validate_prefire_risk`, `build_first_measurement_authorization`, `complete_first_fire_intent`,
`SEAL_SCHEMA_V3`); `tools/make_candidate_seal.py --first-fire-intent … --timing-risk-evidence …` (the intent
emitter; refuses when any non-timing gate fails; the normal validator refuses the intent as not a seal);
`tools/authorize_candidate_first_measurement.py` (MAIN ONLY — you never run it); `tools/fire_modal_auth_eval.py`
first-measurement mode (MAIN ONLY); `make_candidate_seal.py --complete-first-fire-intent … --first-measurement-authorization …
--candidate-t4-receipt …` (completion → unchanged `t4_direct` → `candidate_seal.v3`; MAIN runs it after harvest).
Read `src/tac/candidate_seal.py` lines ~1600–2560 and `tools/make_candidate_seal.py` FIRST; the contract text is
`.omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md` (§"Exact normative contract",
§"Exact CLI and lifecycle", §"RLC2 decision", §"Prospective freeze and real positive control").

## Your deliverable (pr12 §RLC2 decision, verbatim scope) — NO dispatch
Everything in the rlc2 charter (`.omx/research/ddm_rlc2_rebase_rule118_cure_onto_move42_charter_20260910.md`)
except the seal step, which is replaced by the INTENT:
1. Materialize the real move-42 re-base of rlc1's counted-rider cure (exact seven-path receiver delta;
   rider rebuilt against move 42's archive; archive = move 42's bytes + rider by the REAL coder; retain BOTH
   full encodes; prove `archive_bytes ≤ 180,207` — else STOP: pr12 authorizes no fire at 180,208 B or more).
2. Full cold n600 public parse-back proving raw identity with move 42's retained raw (sha in the packet);
   regenerate `MANIFEST.sha256` from OUTSIDE the tree and validate it independently; repeat pr9's literal
   census on the new tree; capture candidate AND frontier public smokes (`experiments/ddm_rlc1_smoke.py` pattern);
   normalized receiver digest must equal the timed RLC1 reference `b06e59a67b60f577…` exactly.
3. Build the exact timing-risk receipt pr12 requires (`candidate_prefire_timing_risk.v1`): lineage to move 40's
   completed `t4_direct` leg (990.054 s; receiver 6726fd77…) with the delta = the counted maps; the measured
   local cost fraction of that delta (rlc1 g3/g4 829.0/831.5 s vs move 40's 783–797 s cold runs — DIAGNOSTICS,
   cite `.omx/research/ddm_rlc1_20260910/QUIESCED_TIMING_RECORD.json` and
   `.omx/research/ddm_dwc1_20260910/MOVE40_QUIESCED_LEG.json`); risk ceiling ≤ 1,260 s; exactly the fields
   the validator reads (`validate_prefire_risk`).
4. Emit and self-validate the intent with `tools/make_candidate_seal.py --first-fire-intent …` (all
   non-timing gates: archive bytes+digest, runtime + normalized receiver digests, receiver pins, manifest,
   current pointer (move 42, sha f111ab42…) and admit bar, both smoke groups, retained payload paths,
   falsifiers pre-registered, `score_claim=false`); prove the normal `validate_seal` and the normal
   `--seal` path both REFUSE it (record the typed refusal). Commit the intent JSON (serializer) — pr12 requires
   a COMMITTED intent; every reference it carries has `{path, bytes, sha256}`.
5. Memo `.omx/research/ddm_rlc3_resume_rebase_cure_onto_move42_prefire_intent_20260910.md`: bytes + sha,
   raw-identity proof, census, smokes, risk receipt, intent path + file_sha256 + bytes + canonical digest, the
   conditional row (`net_dS = 25·(B − 180,238)/37,545,489`; at 180,178 B: −3.995e-5 → S 0.13743655372199698),
   every boundary. Serializer commits (two review passes per .py; no co-author trailer; tags
   `[no-triality] [p0-ledger-ok]`). Checkpoint as `ddm_rlc3`.

## Boundaries (binding)
- NO Modal, NO `authorize_candidate_first_measurement.py`, NO first-measurement fire, NO completion — those are
  MAIN's (pr12: "MAIN alone may create the separate authorization and fire"). No timing windows (suspended).
- Never edit `upstream/`, the PR tree, sealed trees (`ddm_rp1_round2/candidate`, `ddm_sj1_compose39_price/candidate`),
  or `src/tac/decode_wall_clock.py`; do not change the contract code (`candidate_seal.py` intent/authorization/
  completion paths) — if it refuses your real intent for a reason you believe is a contract defect, STOP with
  the exact refusal (that IS the real-control result pr12 wants recorded), do not patch around it.
- Keep every payload (SSD tier ≤ 8 GiB per store; hardlink byte-identical raws with sha-verified certificates);
  launch heavy steps through `tools/launch_detached_process.py --done-receipt …`; do not touch sj1's live
  directories (`ddm_sj1_pass6`).
- Rule 118: the rider is the only counted content; quote the literal census.

## OPTIMAL FORM
- Reference form: rlc1's landed cure (ba0110e15) on move 42's tree + pr12's contract as landed; real coder,
  real n600 parse-back, real smokes — no ledger sums, no fixtures.
- Provenance pins: a47543199 (contract), PREFIRE_CONTRACT_FROZEN.json sha (record), pr12 memo sha 50d00e39…,
  rlc2 STOP memo sha 76ba13cf…, pointer move 42 d2803c214 / archive f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f.

## Prior negatives accounted (operator 2026-08-15)
- rlc2 STOP (cycle), ffi1/ffi2 STOPs (clauses) — all resolved above; a new refusal from the REAL producer is
  the contract's first real control, report it exactly.
- pr9 condition 1 (stale manifest) — regenerate from outside the tree; MAIN's first regeneration hashed its own
  in-progress file.
- rp1 r2 (flag vs constant) — bind base archive/tree by sha in every receipt; sj1's silent revert — your
  archive must be move 42's bytes + rider, proven by raw identity.
- dwc1 (gate with no door) — fixtures prove nothing; only your real intent + MAIN's real harvest do.

Final message: candidate bytes + sha, raw identity, census, the intent path with file_sha256/bytes/digest, the
typed normal-validator refusal, the conditional row, every boundary, and the frontier line
`composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged (MAIN fires).
