# ddm_vr6 — Vertigo is at 45 GiB (the serializer fallback reserve is 40 GiB): certify-or-block reclaim of TONIGHT's superseded candidates' bulk (parse-back trees, renders, twin encodes of moves 38/39-class candidates now superseded by move 40, and closed arms' work dirs) on the vr3/vr5 machinery generalized by cd3/vg1; ≥ 40 GiB back on Vertigo, nothing uncertified (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, high) · Spawned by MAIN 2026-09-10. Sources: vr5 (`463337c85`, apply `484971b9c` + bz2d `919dcd145`: 17 raws, 58.6 GiB; the retained-reproducer admission in `experiments/ddm_vr3_certified_raw_reclaim.py plan-retained`), cd3/vg1 (`a7cb75034`, `d7bf47655`, `f0539bdc3`: the process gate narrowed to command+cwd and its census made robust; 69 tests), vr4 (`216993f0d`: the inventory schema), the live trees: `.omx/state/canonical_frontier_pointer.json` = move 40 (`ddm_sj1_compose39_price`, archive 986d536b…), with `ddm_rp1_rate_directed_predistortion` (move 39) and `ddm_sj1_pass5_price` (move 38) SUPERSEDED but their seals/archives/runtime trees still the reproducers of exact rows — those stay; their parse-back trees, inflated raws, renders and twin-encode work dirs are rebuildable from the sealed archive + runtime. Axes: bytes `[measured]`; `score_claim=false`.

## MANDATE
Inventory the ten largest dirs on Vertigo (MAIN's `du` at spawn is in the memo dir), classify every ≥ 1 GiB file: (A) exact-row reproducer (seal, archive, runtime tree, receipts, the per-pair ledgers) → RETAIN; (B) rebuildable bulk of a SEALED candidate (parse-back trees, inflated raws, renders, twin-encode work/checkpoints, superseded generations like `superseded_binding_*`) → certify with the full certificate (path, bytes, sha, rebuild argv from the seal's own reproducer, archive/runtime shas) → DELETABLE; (C) live arm work (rp1 round 2 is LIVE on the move-40 field; tc3 is at seal) → BLOCKED with owner. Plan with `plan-retained` (a fresh rehash ledger you produce — a real sha per file), hand MAIN the exact `apply` command; do NOT apply (the gate runs outside the sandbox). Target ≥ 40 GiB certifiable on Vertigo; also report APDataStore's tonight's T4 artifact dirs.

## PRIOR-LAW PREDICTION (m38)
- ≥ 45 GiB certifiable across sj1's pass-5 and compose-39 work dirs (parse-back trees ~3.4 GB each × several, renders, encode checkpoints) and bnd2/bnd3/tc2 retained encodes; rp1 and tc3 dirs BLOCKED (live).
- **FALSIFIER:** if < 20 GiB is certifiable, the tier is genuinely full of reproducers; report the BLOCKED table and the drive the next arm must be routed to.

## SCOPE
Inventory + certificates + plan ledger + apply command. No deletion (MAIN applies), no score work.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; never touch any `SEAL_*.json`, `archive.zip`, `candidate_runtime/`, receipts, per-pair ledgers, or anything under a LIVE arm's dir (`ddm_rp1_*` round 2, `ddm_tc3_*`), nor the pointer tree. Read-only until the plan; `pgrep`/`lsof` at plan time recorded even though the apply re-runs them.
- Certify-or-block per CLAUDE.md; ledger `.omx/research/ddm_vr6_reclaim_plan_20260910.jsonl` committed; the rehash ledger on Vertigo with sha.
- Serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, `landing.patch` after ONE attempt. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_vr6`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).

## PRIOR NEGATIVE SIGNAL
- vr4: an empty lsof in a blind sandbox proves nothing; vr5 attempt 1: the arm's own custody artifacts are not consumers (excluded by glob now); cd3: full-argv owner matching false-positives; vg1: the cwd census must tolerate pids without a cwd row.
- Moving bytes between the two SSDs creates no capacity.

## OPTIMAL FORM
- Reference form: vr5's plan + MAIN's apply (`463337c85`, `484971b9c`); vr3's certificate schema (`ff1095d49`). SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no size-only admission; no "old = rebuildable" without the reproducer argv.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
The plan ledger, the rehash ledger, the BLOCKED table with owners, the apply command, memo `.omx/research/ddm_vr6_tonights_reclaim_20260910.md`. Commit via the serializer. End with the live frontier line.
