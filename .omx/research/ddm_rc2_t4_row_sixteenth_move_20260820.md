# SIXTEENTH POINTER MOVE — composed port×rider row ADMITTED: S 0.14827847122030852 @ 180,456 B

`date_utc: 2026-08-20` · `owner: MAIN` · `axis: [contest-CUDA T4, n600]` · `call_id:
fc-01M0G7QCQPACVJV29D7AAQSXAA` · `score_claim: true` · `frontier_moved: TRUE (sixteenth move)`
· cost ≈ $0.16 (CUDA leg; CPU leg in flight)

## THE ANSWER, FIRST

The composed candidate — rider archive `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`
@ **180,456 B** through the {clean native port + RR5 rider} runtime — scored
**S 0.14827847122030852 [contest-CUDA T4, n600]**, recomputed from components (the receipt's
`final_score` field reads `0.15` = the rounded display, Catalog #877 — never cite it). The
projection (0.14827847122030854) realized to 2 ULP. `effective_frontier` updated:
**0.14839100138338618 → 0.14827847122030852 (−1.125302e-4)**.

| Quantity | jg5 (base) | THIS ROW (composed) | verdict |
|---|---|---|---|
| S (recomputed) | 0.14839100138338618 | **0.14827847122030852** | **−1.125302e-4** |
| d_seg | 0.00020139 | 0.00020139 | IDENTICAL |
| d_pose | 6.37e-06 | 6.37e-06 | IDENTICAL |
| archive bytes | 180,625 | 180,456 | −169 B (the entire delta) |
| whole Modal job | 1,491.6 s | **513.8 s** | **2.90× faster end-to-end** |

GPU `Tesla T4`, `gpu_t4_match: True`, n600, `evidence_grade: contest-CUDA`, receipt
`/Volumes/APDataStore/pact/ddm_rc2/t4_row_r2/MODAL_REMOTE_RESULT.json`.

## DECODE IDENTITY — confirmed at the component level; the rr2 desync did NOT occur

Both distortion components are byte-for-byte equal to jg5's across all 600 pairs on the
deterministic CUDA scorer. The rr2 (#1096) failure mode — CPU-prob encode vs CUDA-prob decode —
produced S 27.83, not identity; the rider's adaptive arithmetic coder round-tripped correctly on
CUDA.

**CORRECTED 2026-08-20 (rv17 round-1 HIGH F1, memo ddm_rv17_wave_end_review_round1_20260820.md
— MAIN re-verified all fields at source):** the original text here claimed the receipt "does not
emit a raw sha or per-stage timing split" and that the inflate wall was "NOT separately
measured." FALSE on every clause. The receipt carries `raw_sha256`, a full `stage_seconds`
split, `decoded_token_sha256` (`cc10a7b0…`, byte-matching the CPU leg — the exact rr2-desync
discriminator, so identity is proven at TOKEN level, stronger than the component-level form),
`inflate_elapsed_seconds = 458.752594349`, and the harness's own adjudicated wall verdict:
`Wall budget: PASS [contest-CUDA] … 498.476 s charged <= 822 s cold-cache ceiling`. The
464.559 s figure from the instrumented run on a different tree remains not-transferred — this
tree has its OWN measured inflate wall.

## THE DECODE WALL, CLOSED ON THE SHIPPING OBJECT

jg5's whole job measured 1,491.6 s; this composed job measured 513.8 s on the same schema.
Sharper (per the rv17 correction above): the receipt's own harness adjudicated the ship-wall
question directly — 498.476 s charged ≤ 822 s cold-cache ceiling, PASS — with the measured
inflate term at 458.75 s. Only the GitHub-Actions setup terms (checkout/deps/download) remain
wc2 PROJECTIONS (498–978 s); the decode term we control is measured, not projected, at ~1/3 of
what jg5 shipped with.

## LINEAGE (all receipts SHA-pinned)

rr6 (native port decode identity) → rr8 (instrumented T4 wall row, score-identity control) →
rr5 re-measure (−169 B lossless rider, C1/C2/C3) → rc1 (line-disjoint merge + parse identity) →
rc2 arm (clean-tree byte-neutrality + TWO composed full-receiver n600 runs, determinism repeat +
dual-axis seals, `d9af7d0f4c`) → r1 fire REFUSED (the tool self-defeated `require_active`,
defect #1167, receipt `t4_row_r1/FIRE_REFUSED.json`) → r2 fire ADMITTED (this row).

## STATE + WHAT FIRES NEXT

- CUDA claim `lane_ddm_rc2_composed_cuda_20260820` terminal-closed `completed_harvested`.
- CPU leg LIVE (`lane_ddm_rc2_composed_cpu_20260820`, seal `0651dd69…c934`) — the paired
  contest-CPU row on the SAME bytes; expect the known CPU pose degradation, informational for
  the packet's dual-axis discipline.
- Packet consequences → ddm_pq9 (live): the shipping candidate is now {df7fd266… × composed
  runtime}; the swap procedure applies — commit the archive + runtime, re-pin the hosted URL at
  freeze, re-derive the declared runtime figures from THIS row's schema.
- Object A (jg5 bytes × clean port) remains a proven fallback with its own authority row.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600] — MOVED this unit (−1.125302e-4,
sixteenth move), archive df7fd266…, call fc-01M0G7QCQPACVJV29D7AAQSXAA.**
