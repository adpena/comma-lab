# ddm_gv1 — governor / memory guard / costate organ / controller polish (operator 2026-09-04: "saturate cpu, gpu, and ane;
# continue polishing the memory guard and governor and costate organ and controller")

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0

## Why
Today MAIN sequenced the cell queue BY HAND (seal → re-root pins → claims → authorize → launch → milestone reads →
verdict → next cell) and serialized the Metal on a governance rule (one Metal fire) that the operator has now overridden:
the memory guard admitted ng3 concurrently with ng2 at 15:32Z (reclaimable 58 GiB ≥ 42 + 16) and throughput rose
20 + 15 = 35 steps/min vs 28 serial. The apparatus should hold this: admission by MEASURED headroom, the queue driven by
the controller, the organ sensing the cells, and the ANE available as a third substrate for advisory reads.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line)
- Governor / launch surfaces: `tools/launch_detached_process.py` (`--derive-resource-budgets`, `--measured-peak-rss-gib`,
  `--nice-best-effort`, done receipts), `tools/launch_guard_hook.py` (orphan-prone-waiter refusal — keep), the memory
  preflight `tools/witness_memory_preflight.py` (peak-RSS projection; the "0.70×RAM REFUSE" law) and the admission gate
  (`[[m78]]` reclaimable-aware; `[[m79]]` ceiling 116 GiB), `tools/claim_lane_dispatch.py` (claims), the chain driver
  `experiments/ddm_qbr1_cell_chain.py` (attach/launch/wait/adjudicate for ONE sealed six-cell order), and MAIN's hand-written
  fire scripts under `/Volumes/APDataStore/pact/ddm_ng{1,2,3}_*/fire/` (the pattern to absorb: preconditions → claims →
  authorize via the chain driver's functions → sealed launcher argv → distinct done-receipt).
- Costate organ / controller: `src/tac/ddm_costate_organ.py`, `tools/costate_digest.py` (SessionStart digest),
  `tac.witness_control.*`; the activation ledger (`src/tac/witness_dsl/activation_ledger.py`).
- The pin re-root tool `experiments/ddm_reseal_pins_inside_sealed_tree.py` (ccedf3b8a) and the seal law memory
  `seal_validates_only_inside_the_tree_that_fires_it_20260904`.

## Deliver (each a reusable, tested surface — CLAUDE.md "Results must become system intelligence")
1. **Memory-guard admission for concurrent Metal cells**: one function + CLI (`tools/cell_admission.py` or inside the
   memory preflight) that reads free + inactive (reclaimable), the resident peaks of live cells (from their launch
   manifests/status files), the candidate's measured peak, and a margin; returns ADMIT/REFUSE with the arithmetic;
   pinned by tests; fire scripts consume it (replace the inline vm_stat arithmetic).
2. **Metal-contention telemetry → governor learning**: record per-cell step rate (from history.jsonl growth) under
   concurrency into a ledger the governor reads (today's row: 28 serial → 20 + 15 concurrent); admit a second/third
   cell only while total throughput ≥ serial; refuse with the measured number otherwise.
3. **Cell queue driver** (generalize the QBR1 chain driver): take an ordered list of SEALED cell configs (re-rooted pins
   verified inside their sealed trees), fresh claims per cell, authorize via the chain driver's own functions, launch with
   distinct receipts, read milestones against a named control, write per-cell verdict rows against pre-registered
   falsifiers, and hand off; concurrency governed by item 1–2; MAIN only adjudicates. Include a `--dry-run`. Run it in
   dry-run against the ng-cell store as the test.
4. **Costate organ SENSE for live cells**: the digest shows live cells, their rate, ETA, last milestone vs control, and
   admission headroom (today the digest still speaks about TR1-era runs).
5. **ANE as a substrate for advisory reads**: verify at source whether a CoreML export of the frozen SegNet/PoseNet is
   already in-tree (grep `coreml`, `ane`, `mlprogram`); if present, benchmark one n32 argmax read on ANE vs CPU torch and
   report parity (bit-identical argmax? if not, the site delta) — ANE is NEVER authority; if absent, write the smallest
   honest conversion receipt (what it would take, blockers) — no multi-hour build.
Land each with tests + a memo `.omx/research/ddm_gv1_governor_memory_guard_controller_polish_20260904.md`.

## Constraints
- $0; ng2 + ng3 are LIVE on the Metal — never touch their custody or claims; read their manifests only. `upstream/` and
  `submissions/semantic_joint_ctxmix/` read-only; no /tmp paths. OPTIMAL FORM: reference = the existing chain driver and
  launcher at `8376e5fe38f9baaeb3b725aeab6720a4e3515839`; extend, never fork a parallel launcher. Commits ONLY via `tools/subagent_commit_serializer.py
  --message … --files … --expected-content-sha256 <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO
  co-author trailer; every .py: tests + `tools/review_tracker.py mark-file` twice; never REVIEW_GATE_OVERRIDE on .py.
  EQUATIONS-LEG LAW where a measured law lands (throughput under concurrency is a candidate). Final message →
  `.omx/research/arm_final_messages/ddm_gv1_final_<utc>.md`, committed; LAST action `touch .omx/tmp/codex_runs/ddm_gv1.done`.
