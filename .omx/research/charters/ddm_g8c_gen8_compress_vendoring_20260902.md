# ddm_g8c_gen8_compress_vendoring — build the gen-8 public-minimal submission tree `semantic_joint_ctxmix` with a SELF-CONTAINED compress.py (makes the PR-template compression answer literally true); prepare-only, nothing publishes (harness task #1389 — memo anchor ddm_yr1_yousfi_adversarial_pr_review_20260902.md fire-orders 2+3)

## MANDATE

Operator 20260902: *"what is necessary So the compression answer can be yes? Anything that needs to be consolidated?"* then *"why didn't you do the actual consolidation? wtf"* — plus the same-day steer that leading submissions are consolidated (one prose doc + at most one README beside the code).
This arm BUILDS the generation-8 public submission tree now: vendor the encode side of the five lossless stages so `compress.py` runs from a bare checkout, trim the 55-file gen-7 lab-handoff to the measured ship-set, ship exactly ONE README beside the code, and prove everything by byte-identity. The frozen gen-7 packet is never edited; publishing stays behind the operator's #1111 confirm — this arm prepares, it does not publish.

## SCOPE

1. NEW TREE: create `submissions/semantic_joint_ctxmix/` (repo-side, prepare-only). Seed it from the frozen gen-7 authority runtime (38 files enumerated by `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/MANIFEST.sha256`) MINUS `FX5_BUILD_MANIFEST.json` (yr1 F4: internal + absolute-path leak). Authority: the frozen packet READ-ONLY at that path; every copied file SHA-verified against the manifest.
2. VENDORED COMPRESS: make `compress.py` in the new tree replay the five lossless stages (fx5 integer log-odds mixer → dx2 group-conditioned re-encode → gb1 tile-conditioned re-encode → lb1 joint collect → afr1 re-encode) importing ONLY from the tree + stdlib + the runtime's already-declared deps. Source of truth for the chain: the ce1-verified `experiments/ddm_pq2_compress_e2e.py` AFR1 chain (commit 2c3a2153e4) — vendor the encode-side modules it imports (locate at source; do not guess module lists from memos). Inputs declared with SHA-256 pins inside the script; the pinned base archive (sha df7fd266…, 180,456 B, generation-6 rc2 — verify the full sha at source from the pq12/ce1 receipts) documented as an adjacent-file-or-fetch input, NEVER embedded.
3. BARE-CHECKOUT PROOF: fresh venv + clean clone of the new tree only → `python compress.py` reproduces the exact afr1 archive `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` (180,002 B) — TWO complete runs, receipts retained. If a run exceeds 30 min, detached per HARD CONSTRAINTS.
4. ONE README: place the operator-reviewed README content (Drive `SUBMISSION_PACKAGE.md` §2 — pull the live copy, don't retype from memory) as the tree's ONLY markdown. No COMPRESS.md, no accounting ledger, no review records, no AppleDouble sidecars (purge `._*` per the AppleDouble/ExFAT row in PRIOR NEGATIVE SIGNAL below).
5. FAIL-CLOSED RUNTIME EDITS (yr1 F6+F8): `inflate.sh` Brotli network-install → fail-closed preflight with a clear error; unguarded `cc` → guarded with a preflight; `inflate.py` silent CPU fallback → explicit CUDA-required error (CPU is MEASURED over-budget: killed at 1,800 s, memo ddm_afr1_cpu_axis_timeout_verdict_20260902.md). EVERY runtime edit then owes: fresh runtime-tree digest + a local decode byte-identity proof — all 600 pairs, 3,662,409,600 raw output bytes identical to the gen-7 authority raw hashes (the ux1-verified identity path; local wall-clock may exceed the CI budget, that is fine — detached + receipts). The archive bytes are UNTOUCHED; if identity fails, land honest-blocked with the diff named, never associate the afr1 score with a non-identical tree.
6. CLEAN MANIFEST: emit `MANIFEST.sha256` for the new tree + a one-page verification note inside the memo (tree file count, digests, identity receipts). Deliver a typed READY row for MAIN; MAIN owns any Modal re-buy decision and all publishing.

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
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/`.
- DETACHED >30-MIN COMPUTE: any single compute step projected to exceed 30 minutes MUST
  launch outside the arm session with `nohup` + `disown`, a pidfile, crash-resumable stage
  checkpoints, and a durable done-receipt. The arm MONITORS that process; a successor or
  MAIN harvests the done-receipt. An in-session multi-hour compute loop is FORBIDDEN.
- CLOSED-FORM-FIRST (operator 2026-08-31 "All upstream can be closed form"): the scoring
  chain is frozen piecewise-analytic math with every non-analytic locus exactly known —
  derive/solve against the EXACT upstream operators (atlas:
  ddm_cfa1_closed_form_atlas_20260831.md) before any fit, surrogate, or sampled estimate;
  a fitted stage owes a one-line reason the closed form was not usable.
- FROZEN CUSTODY: `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/` is READ-ONLY sealed; the new tree is a COPY. Do not touch the live qbr1 burn-prep files or the Drive folder (MAIN owns both). No `gh`, no hosting, no public comment, no push of anything the operator has not confirmed — the #1111 gate is MAIN+operator territory.
- Decode identity proofs run LOCALLY (no scorer needed for byte-identity); the deterministic receiver was proven CPU-runnable for identity (ux1/rr6 lineage) — wall-clock over the CI budget locally is acceptable and is NOT a compliance claim.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- yr1 F3 (ddm_yr1_yousfi_adversarial_pr_review_20260902.md): the shipped COMPRESS.md itself said the script "will not import" from a bare submission dir — the exact contradiction this arm removes; never re-ship a doc that contradicts the yes-answer.
- rc64 pin-stale blocker (cure recorded in the ce1 chain receipts beside commit 2c3a2153e4): `ddm_pq2_compress_e2e` previously carried a stale runtime-member pin — re-derive every pin from the LIVE frozen manifest, never carry pins forward from memos.
- e4 brotli precedent (memo ddm_e4_brotli_declared_dep lineage): network installs are admissible ONLY fail-closed with a proven bootstrap; the r5 lesson — prove the bootstrap in a bare venv, never assume host site-packages.
- AppleDouble/ExFAT sidecar class (cure receipts in the 08-18 session ledger + ddm_sf2 structural-fix wave): `._*` sidecars and `__pycache__` on ExFAT killed compliance runs at t=5s; purge + `PYTHONDONTWRITEBYTECODE=1` in every run env.

## OPTIMAL FORM

- Family exemplar: the ce1 compress-chain consolidation is the reference — commit 2c3a2153e4 ("ddm_ce1: reproduce AFR1 compression chain end to end") + its two byte-identical rebuild receipts, gate discharged at commit 03fb73d2a9. This arm is the same family at the next rung: same chain, imports closed over a bare checkout.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — the vendored stages must be the REAL encoders (byte-identical output), never re-implementations or stubs.
- **PRIOR-LAW PREDICTION (falsifiable):** the ce1 chain is deterministic with SHA-pinned inputs, so closing its import surface changes nothing — the bare-checkout rebuild will reproduce cbb8d928… exactly on both runs. FALSIFIER: any byte differs → an undeclared environment dependency exists in the chain; count it plainly, name the stage, and land honest-blocked with the diff.

## DELIVERABLE

`.omx/research/ddm_g8c_gen8_compress_vendoring_20260902.md` — typed rows: (1) vendored-module table (source path → tree path → sha), (2) bare-checkout rebuild receipts ×2 (env fingerprint + archive sha + wall-clock), (3) runtime-edit ledger (edit → new tree digest → decode-identity receipt), (4) ship-set table (kept/dropped per file with reason), (5) READY-FOR-MAIN row (what MAIN verifies before the tree is presented to the operator) — or honest-blocked rows with the named diff. Commit via the
serializer. End with the own-vehicle frontier line.
