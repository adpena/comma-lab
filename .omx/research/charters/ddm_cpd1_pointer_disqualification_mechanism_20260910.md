# ddm_cpd1 — the canonical frontier pointer has NO disqualification mechanism: an exact row whose archive fails compliance (move 41, rule 118) still selects as `effective_frontier`; add a journaled `disqualified` field on anchor-mirror rows, make the refresh's selection rule skip them with the reason surfaced, give MAIN a CLI to disqualify/reinstate with a rationale, and a preflight that refuses a packet/pointer refresh citing a disqualified row (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, high) · Spawned by MAIN 2026-09-10. Sources: `.omx/state/canonical_frontier_pointer.json` (schema: `effective_frontier` = min over `our_local_frontier_contest_cuda/cpu` anchors and the upstream snapshot; `refresh_provenance`; anchor mirror schema `modal_auth_eval_anchor_mirror.v2` under `experiments/results/modal_auth_eval_mirror/`), `tools/refresh_canonical_frontier.py` + `tac.canonical_frontier_pointer` (selection rule; auto-refresh on dispatch completion per Catalog #343), `tools/pointer_move_packet.py` (writes moves; must refuse a disqualified lane as the prior), pr8's P0 (`6be09cf42`) and gs3 Addendum 22 (`6a17b63b9`): move 41 (lane `ddm_tc3_t4_lane_predictor_tail_20260910`, archive 299a8201…) is an exact row that does NOT qualify; the submittable pointer is move 40 (lane `ddm_sj1_t4_compose39_rp1_union_20260910`). CLAUDE.md "Frontier scores are pointer-only" (effective_frontier = min of QUALIFYING exact scores) and "Bugs must be permanently fixed AND self-protected against" (two landings). Axes: apparatus; `score_claim=false`.

## MANDATE
(1) Data: an append-only journal `.omx/state/frontier_disqualifications.jsonl` (lane_id, archive_sha256, reason class ∈ {rule118_content_in_code, decode_budget, determinism, custody}, evidence path, who, when, reinstated_at/why) and a `disqualified` projection onto the anchor-mirror row used by the refresh. (2) Selection: `effective_frontier` and both local anchors skip disqualified rows and carry `disqualified_rows: [...]` with reasons so the operator briefing shows them; `refresh_provenance` records it. (3) CLI: `tools/frontier_disqualify.py --lane … --archive-sha256 … --reason-class … --evidence … --rationale "…"` and `--reinstate` with a rationale; refuses placeholder rationales (Catalog #287 sister). (4) Packet: `tools/pointer_move_packet.py` refuses to use a disqualified row as the prior and refuses to packet a disqualified lane. (5) Preflight (claim a catalog number): refuses a pointer JSON whose `effective_frontier` cites a lane present in the disqualification journal (STRICT-flip in the same batch if live count is 0 after MAIN disqualifies move 41 — MAIN runs the CLI, not you). Tests ≥ 12 (select/skip, journal append-only, reinstate, packet refusal, preflight positive/negative/waiver).

## PRIOR-LAW PREDICTION (m38)
- ≤ 250 lines across the pointer module, the refresh tool, the packet tool and the CLI; after MAIN disqualifies move 41 the refresh selects move 40 (0.13763861019288715) and lists 41 with `rule118_content_in_code`.
- **FALSIFIER:** if the selection rule cannot skip a row without breaking Catalog #343's auto-refresh contract, say so and propose the narrowest change.

## SCOPE
Apparatus only; no score work; MAIN performs the actual disqualification with the CLI.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; never edit the anchor mirror rows themselves (append-only provenance; Catalog #110/#113) — the journal is the mutation. `.py` = 2 visible review passes + ruff; serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, `landing.patch` + bundle with the ref name recorded. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_cpd1`.
- Catalog quota: past #400 a new STRICT gate must retire/replace or carry the file-level waiver — check `check_catalog_quota_under_400`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).

## PRIOR NEGATIVE SIGNAL
- The pointer drift class Catalog #343 extincted (hardcoded literals) — this is its sibling: a qualifying rule with no disqualification input. # VERDICT_SCOPE_OK: apparatus charter; no negative verdict issued

## OPTIMAL FORM
- Reference form: Catalog #343's pointer landing and pm2's fix+gate+tests pattern (`8038f9e77`). SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no mutation of anchor rows; no silent skip without a surfaced reason.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
Journal + selection change + CLI + packet refusal + preflight + tests + memo `.omx/research/ddm_cpd1_pointer_disqualification_20260910.md`. Commit via the serializer. End with the live frontier line (the SUBMITTABLE one: move 40).
