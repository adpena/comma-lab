# ddm_vr7 — shuffling is exhausted (APDataStore 33 GiB, Vertigo 63 GiB): extend the retained-reproducer admission to the sj1-chain layout (vr6: "the planner refuses the candidate layout") so the cold-stored sj1 bulk (25.6 GiB on APDataStore, MOVED.json manifests) and tc3's superseded move-37/39-base traces (~10 GiB on Vertigo) can be DELETED under a full certificate; plan + apply command for MAIN (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, high) · Spawned by MAIN 2026-09-10. Sources: MAIN's certified move (`fbf6553ef`: `.omx/research/ddm_main_storage_20260910/sj1_bulk_move_log_20260910.jsonl` — 11 files, 25.58 GiB, sha per file, manifests at each source path: parse-back `0.raw` files and `odd_frames.u8` renders of `ddm_sj1_pass5_price`, `ddm_sj1_compose39_price`, `ddm_sj1_multipass_token_predistortion`), the reproducers: each sj1 candidate's SEAL + `candidate_runtime/archive.zip` (moves 35, 38, 40 are exact rows — the seals are retained on Vertigo) and `experiments/ddm_sj1_joint_admission.py parseback` / `render-edits` as the rebuild argv (state the exact argv per file and verify ONE rebuild reproduces the recorded sha — that is the certificate), vr6 (`6020c867f`: the layout refusal; 181 BLOCKED rows), vr5/cd3/vg1 (`463337c85`, `a7cb75034`, `d7bf47655`: `experiments/ddm_vr3_certified_raw_reclaim.py` plan-retained/apply; process gate command+cwd with robust cwd census), tc3 (`62451b9d3`: `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/` — the pre-rebase traces under `launch_trace600`, `launch_prepare600`, `launch_public600` etc. are on the SUPERSEDED move-37/39 bases; the move-40 `rebase_move40/` tree is the live seal's — RETAIN). Axes: bytes `[measured]`; `score_claim=false`.

## MANDATE
(1) Generalize `plan-retained`'s admission to the sj1-chain layout by DATA (a per-family reproducer descriptor: seal path, archive sha, rebuild argv, output path, expected sha), keeping vr3's two original families and vr5's families working (tests). (2) For each cold-stored sj1 file: resolve its `MOVED.json` (mv1's helper if landed — check `git log` — else read the manifest directly), verify the destination sha, and PROVE rebuildability by rebuilding ONE representative file per family from its seal's archive (a real parse-back, twin-hashed) — that rebuild is the certificate; the others inherit it only if their descriptor is identical in kind (say so). (3) tc3's superseded pre-rebase traces: classify by base (move-37 / move-39 = superseded; move-40 = live seal's — RETAIN); certify the superseded ones against tc3's retained receipts. (4) Emit the plan ledger + the exact `apply` command for MAIN; do NOT apply. Target ≥ 30 GiB certified across both drives.

## PRIOR-LAW PREDICTION (m38)
- All 11 cold-stored files certify (the parse-back rebuild reproduces the recorded sha); ≥ 8 GiB of tc3 traces certify; total ≥ 33 GiB DELETABLE.
- **FALSIFIER:** if a representative rebuild does NOT reproduce the recorded sha, that family is BLOCKED with the reason (the payload is then NOT rebuildable and must stay — the certify-or-block rule doing its job); report it first.

## SCOPE
Planner extension + one rebuild per family (≤ 2 procs, launcher `--nice-best-effort`, bulk under `/Volumes/VertigoDataTier/pact/ddm_vr7_certify/` ≤ 8 GiB, deleted by you after the sha comparison — your OWN scratch only) + plan ledger + apply command. No deletion of any inventory row (MAIN applies).

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; never touch seals, archives, candidate runtimes, receipts, ledgers, the pointer tree, rp1's round-2 dir (LIVE), eb2's dir (LIVE), or tc3's `rebase_move40/` tree. `.py` = 2 visible review passes + ruff; serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, `landing.patch` after ONE attempt. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_vr7`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).

## PRIOR NEGATIVE SIGNAL
- vr6: layout refusal; vr4: "absent" ≠ certificate; vr5 attempt 1: custody artifacts are not consumers; cd3/vg1: gate precision — all are constraints on this planner, not things to re-derive.

## OPTIMAL FORM
- Reference form: vr5 (`463337c85`) for the plan/apply split and journal; vr3 (`ff1095d49`) for the certificate schema. SCOPE reductions: one rebuild per family (declared; inheritance stated per file). MECHANISM reductions FORBIDDEN: no "rebuildable" without a rebuild; no size-only admission.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
Planner change + tests, the plan ledger `.omx/research/ddm_vr7_reclaim_plan_20260910.jsonl`, the rebuild certificates, the apply command, memo `.omx/research/ddm_vr7_certified_deletion_20260910.md`. Commit via the serializer. End with the live frontier line.
