# ddm_g8r_compress_adversarial_review — fresh-eyes RECURSIVE ADVERSARIAL REVIEW of the gen-8 compress script + tree (operator binding 09-02 "Need recursive adversarial review against the compress script when it lands"): findings→fix→counter-reset, SEAL at 3 consecutive clean passes; the last gate before the operator's READY report (harness #1389 — memo anchor ddm_g8c_gen8_compress_vendoring_20260902.md)

## MANDATE

Operator 20260902: *"Need recursive adversarial review against the compress script when it lands"* + the production-hardened bar (*"This PR should basically be production hardened and optimal"*). The gen-8 tree LANDED (commit 7ba53d1b84, `submissions/semantic_joint_ctxmix/`, 40 files, tree digest 477f2a3d…789e). This arm is the fresh-eyes adversarial reviewer: attack `compress.py` (170 KB, vendored five-stage encode chain) and the whole shipping tree per the CLAUDE.md recursive-review protocol — every finding fixed same-round (compress-side) or typed-flagged to MAIN (receiver-side, since receiver edits invalidate the decode-identity proof), the clean-pass counter resets on ANY finding, and the review SEALS only at 3 consecutive clean passes. The seal is the final gate before MAIN's READY report to the operator; nothing publishes.

## SCOPE

1. RECEIPT-vs-CLAIM audit: every claim in the g8c memo (`.omx/research/ddm_g8c_gen8_compress_vendoring_20260902.md`) and the tree's `README.md` re-derived from the EXECUTED receipts at `/Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/` (`bare_proof_v3/RESULT.json` — two runs reproducing cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25 at 180,002 B from base df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080 at 180,456 B · `runtime_identity_v3/RESULT.json` — PASS over 3,662,409,600 raw bytes sha 7246a4ff8f79…f5f2de7 · `runtime_preflight/` refusal receipts · `RETENTION_MANIFEST.json`). NAME the `"status": "DISABLED"` entry inside runtime_identity_v3/RESULT.json — confirm it is the expected untested-CUDA leg and not a silently skipped required check.
2. SHA-PIN COMPLETENESS in `compress.py`: every input pinned, every pin re-derived from live custody, none carried forward from memos (the rc64 stale-pin lesson in PRIOR NEGATIVE SIGNAL).
3. IMPORT CLOSURE proven by execution: in a bare venv with ONLY the runtime's declared deps, `python -c "import compress"` (and the vendored modules) resolves from the tree + stdlib alone — zero reach-back into the research repo. Also grep-prove: no absolute paths, no /Volumes/, no network calls outside the declared fail-closed installs, no hidden state files.
4. FAIL-CLOSED REFUSALS exercised as positive controls (executed, not read): missing compiler → refuse; missing Brotli → the declared fallback or refuse per design; CUDA unavailable → the explicit CUDA-required error in `inflate.py` (never silent CPU). Each control's rc + message captured.
5. DETERMINISM + README truth: the two rebuild runs byte-agree (re-derive from receipts; re-run ONE fresh bare-checkout rebuild yourself as the review's own executed control, ~10 min); every claim in README.md matches the code exactly (no aspirational text).
6. STRICT COMPLIANCE: run `scripts/pre_submission_compliance_check.py` in its strictest applicable mode against the gen-8 tree + the pinned afr1 archive identity (grep its argparse first — never invent flags) → per-check rows; each of gen-7's 7 RED rows must be GREEN-by-construction here or carry a one-line written adjudication. Deliver the receipt; MAIN adjudicates residual reds.
7. ROUNDS: publish a per-round findings table (finding → severity → fixed-or-flagged → proof of fix). Compress-side fixes: apply + re-run the bare-checkout proof (output must still be cbb8d928…, byte-exact) + 2 review passes + serializer commit. Receiver-side findings: DO NOT edit — typed flag to MAIN with the named identity-proof cost. SEAL = 3 consecutive rounds with zero findings.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who currently holds it into a
  charter: an occupancy claim goes stale the moment that holder exits, and the arm has no way
  to learn it did (the #1210 stale-precondition genus — MEASURED 2026-08-29, when
  `ddm_bz2_bornsmall_capacity_ceiling` correctly refused to claim a capacity ceiling because
  a charter told it a since-released lane was taken). If this arm's work needs a scorer run,
  emit a typed fire order naming its trigger and let MAIN fire it; landing an honest partial
  plus a fire order is the CORRECT outcome, never a failure.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_g8r_compress_adversarial_review/`.
- DETACHED >30-MIN COMPUTE: any single compute step projected to exceed 30 minutes MUST
  launch outside the arm session with `nohup` + `disown`, a pidfile, crash-resumable stage
  checkpoints, and a durable done-receipt. The arm MONITORS that process; a successor or
  MAIN harvests the done-receipt. An in-session multi-hour compute loop is FORBIDDEN.
- CLOSED-FORM-FIRST (operator 2026-08-31 "All upstream can be closed form"): the scoring
  chain is frozen piecewise-analytic math with every non-analytic locus exactly known —
  derive/solve against the EXACT upstream operators (atlas:
  ddm_cfa1_closed_form_atlas_20260831.md) before any fit, surrogate, or sampled estimate;
  a fitted stage owes a one-line reason the closed form was not usable.
- OWNERSHIP: `submissions/semantic_joint_ctxmix/` compress-side files (compress.py, compress_vendor/) are this arm's to fix; receiver files (inflate.py, inflate.sh, runtime/, cpr1/) are FLAG-ONLY (identity-proof custody). The frozen gen-7 packet on APDataStore is READ-ONLY sealed. NO publish, no gh, no hosting — the operator's final go-ahead is the only publish gate, held by MAIN+operator.
- The FULL decode-identity re-proof (hours) is NOT this arm's to re-run — cite the existing runtime_identity_v3 receipt; only a receiver-file change would invalidate it, which is exactly why receiver findings are flag-only.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- rc64 stale-pin blocker (cure receipts beside ce1 commit 2c3a2153e4): the compress chain previously carried a stale runtime-member pin — every pin must be re-derived from live custody, never carried from memos.
- The g8c arm's own DEAD-ENDS (memo ddm_g8c_gen8_compress_vendoring_20260902.md): an 11-file helper directory violated the public shape (now embedded — verify the embedding is real and complete); modeling DX2 as another RC64 encode reproduced the WRONG archive until the real Rice-to-CABAC fold was used (verify the shipped stage is the real fold); the first identity launch refused correctly on a missing RC64 decoder binding (verify the binding ships).
- AppleDouble/ExFAT sidecar class (cure receipts in the ddm_sf2 structural-fix wave): `._*` + `__pycache__` killed compliance runs; verify the tree and any run env stay clean (PYTHONDONTWRITEBYTECODE=1).
- e4 brotli precedent (memo ddm_e4_brotli_declared_dep lineage): network installs admissible ONLY fail-closed with a bare-venv-proven bootstrap.

## OPTIMAL FORM

- Family exemplar: the candidate-seal contract landing is the reference — commit 361608c875, 36 controls ALL EXECUTED including 20 negative-direction refusals proven by real runs; that executed-control discipline (never read-the-code-and-nod) is this review's form. Sister receipt: the ce1 chain proof at commit 2c3a2153e4.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — every refusal control EXECUTED with captured rc, never inferred from reading the guard.
- **PRIOR-LAW PREDICTION (falsifiable):** fresh-eyes reviews of first-generation vendored scripts at this size (~170 KB) find ≥1 real finding in round 1 (the review corpus's base rate). FALSIFIER: rounds 1–3 all clean with every control executed → count it plainly and SEAL; a zero-finding review with executed controls is a legitimate outcome, not a failed review.

## DELIVERABLE

`.omx/research/ddm_g8r_compress_adversarial_review_20260902.md` — typed rows: (1) per-round findings table w/ severity + fix-or-flag + proof, (2) executed-control table (control → command → rc → message), (3) compliance per-check rows + the 7 gen-7-RED dispositions, (4) the DISABLED-status adjudication, (5) SEAL verdict (3 clean passes) or honest-blocked with the named residual. Commit via the
serializer. End with the own-vehicle frontier line.
