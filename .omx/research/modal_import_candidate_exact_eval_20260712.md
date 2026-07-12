# Modal exact contest-CPU eval — PR128-click IMPORT CANDIDATE (NON-SUBMISSION defensive bank)

**UTC:** 2026-07-12
**Agent:** MODAL-EXACT-EVAL (modaleval)
**Operator authorization (2026-07-12, verbatim intent):** "Go on modal but we are NOT going to
submit a PR with it. Only with cgauge whether v9 or an iterated version." → Modal exact-eval GO for
the DEFENSIVE-BANK candidate; the result NEVER becomes a contest PR (borrowed substrate).

## What was measured

- **Archive:** `experiments/results/pr128_click_import_forensics_20260712/import_candidate_archive.zip`
  - sha256 `196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5`, **176,564 B** (re-verified pre-dispatch)
- **Runtime:** `experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/` (PR110-lineage
  inflate.sh + inflate.py + src + encoder). The import candidate decodes under this runtime.
- **Axis:** contest-CPU (Linux x86_64), Modal CPU container. The decisive axis — our pointer
  (0.19108282419209976) is contest-CPU; PR128's clicks were CPU-axis-selected. CUDA is a separate
  axis, NOT run (not required for a non-submission CPU bank; not inferred from CPU).
- **call_id:** `fc-01KXAGAT8JQA4BNH64FJ1SDC5N` (persisted: CALL_ID.txt / modal_call_id.txt /
  modal_auth_eval_spawn.json / dispatch.log)

## Advisory (pre-dispatch, macOS-CPU, NO score claim)

| | d_seg | d_pose | bytes | S |
|---|---|---|---|---|
| incumbent base (macOS-CPU adv) | 0.0005598958 | 2.9416e-05 | 177,169 | 0.19111029 |
| **import candidate (macOS-CPU adv)** | 0.0005336507 | 2.9372e-05 | 176,564 | **0.18806993** |

Advisory macOS-CPU reproduces PR128's published 0.187992 within 8e-5. Advisory delta import vs the
contest-CPU pointer incumbent (0.19108282) = **-0.00301**.

## Exact contest-CPU result — MEASURED (recomputed from components)

S = 100·d_seg + √(10·d_pose) + 25·bytes/37545489 (my recompute == tool == canonical_score; the
rounded `final_score` display field was NOT used).

- d_seg (avg SegNet dist) = **0.0005334**
- d_pose (avg PoseNet dist) = **2.937e-05**
- archive_bytes = **176,564** ✓ matches
- archive_sha256 (provenance) = **196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5** ✓ matches
- n = **600**
- **S [contest-CPU] = 0.1880443979880752**
- delta vs incumbent 0.19108282419209976 = **-0.003038426204024558**
- **pointer MOVED**: `our_local_frontier_contest_cpu` → 0.1880443979880752 (sha 196acd18),
  architecture_class `lane_pr128_click_import_NONSUBMISSION_defensive_bank_20260712`,
  `submitted_pr_number_for_current_frontier` = null (unchanged).
- evidence_grade = contest-CPU, score_axis = contest_cpu, score_claim = True, passed = True, returncode 0
- Exact vs advisory: exact 0.18804440 vs advisory macOS-CPU 0.18806993 — within 2.6e-5 (macOS-CPU↔
  contest-CPU drift); exact is marginally lower. Advisory→exact gap is negligible, no CPU-axis drift concern.

## borrowed_substrate_accounting (NO-FAKE #7)

- **Borrowed mechanism (structure):** PR128 `rhnerv_latent_polish` (author a12dongithub, MIT license)
  — the "click" latent-table polish. Source: their release `archive.zip` sha `cfd941de`, final latent
  table decoded byte-exact.
- **Clicks applied to OUR base:** PR112 == our-PR110 base is byte-identical (same q-grid, 0 differing
  cells). The import candidate = OUR PR110 payload with PR128-derived latent clicks; 2,656 clicks,
  598 pairs touched, 28 dims touched.
- **Ours-original in this archive:** the PR110 payload/entropy-recode substrate + q-grid + runtime.
  **Borrowed:** the click *values* (latent polish deltas) lifted from PR128's released table.
- **Classification:** NO-FAKE #7 defensive prior — borrowed mechanism structure, a search accelerator,
  NOT innovation. This is a DEFENSIVE BANK, never a submission.

## NON-SUBMISSION flag (binding)

Per operator 2026-07-12: **contest PRs are reserved exclusively for our own V9·CGauge witness**
(v9 or an iterated version). This import candidate is an INTERNAL frontier bank only.
`submitted_pr_number_for_current_frontier` stays `null`. NO PR was or will be opened for this archive.

## Modal spend

- CPU-only single-axis dispatch, 600-sample eval, ran fast (result harvested < a few minutes after
  spawn). Est. ~$0.03–0.10. Cumulative this task ≈ **≤$0.10**. Well within the #381 ≤$20 envelope.
  CUDA axis NOT dispatched (not required for a non-submission CPU bank).

## Verdict

REAL contest-CPU pointer move (MEANS→END: a lower exact score landed). Internal defensive bank only.
NO PR was or will be opened for this archive — submissions are reserved for our own V9·CGauge witness
per operator 2026-07-12. The 0.18804440 defensive frontier buys margin while the witness matures.
