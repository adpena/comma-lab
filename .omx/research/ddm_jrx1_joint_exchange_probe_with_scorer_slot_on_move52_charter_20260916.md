# ddm_jrx1 — the joint (field + renderer) exchange probe, resumed WITH the scorer slot (charter, MAIN 2026-09-16; codex astra xhigh; fires only after pd5 releases the host scorer claim)

jrd1 (landed c9bf03e62; receipt `.omx/research/ddm_jrd1_20260916/RECEIPT.md`, handoff `HANDOFF.json`) ran every
scorer-free stage and stopped where the common contract queues scorer work without a slot assignment. This arm IS
jrd1's charter (`.omx/research/ddm_jrd1_joint_rate_aware_descent_step0_exchange_probe_on_move52_charter_20260916.md`,
commit 1c1aba7ce — read it in full; every row, gate, falsifier, boundary and OPTIMAL FORM binds unchanged) with the
scorer slot assigned (`--owns-scorer` at queue time) and the two definitions jrd1 asked MAIN to settle:

## Definition 1 — "zero net coded bytes" for the joint row
The joint step is admitted as zero-net when the REAL re-encoded total (tail stream + renderer weight section, twins) lies
within ±35 B of the move-52 base (179,332 B) — one container-lottery sd (memory
`container_break_delta_is_a_one_sample_lottery…`, sd 34.8 B). The residual ΔB inside that band is still CHARGED at the
pointer exchange 6.658589531221714e-7 S/B in the realized-exchange arithmetic; nothing is treated as free. jrd1's own
measurement binds: an int4 weight step re-encodes at −4 … +29 B, so the renderer part of every joint step carries its
real re-encoded delta, never 0.

## Definition 2 — the row-1 knee tolerance (instrument check)
Row 1 must reproduce pd4's per-pair single/2-token price on the same pool: the median bits per changed token over the
24 drawn pairs must land within ±10 % of pd4's 12.0 bits/token (per-bit ranking; memo
`.omx/research/ddm_pd4_pose_directed_pass4_price_lever_on_move52_20260913.md`), and the resolved-pose seg credit per
bit within ±15 % of pd4's row for the pairs both drew. Outside either band the INSTRUMENT is under test: STOP, report
the discrepancy with both numbers, and do not run rows 2–3 (the falsifier pd4/jrd1 pre-registered).

## Reuse, not rewrite
Start from `experiments/ddm_jrd1_byte_preflight.py` (jrd1's; the seeded pair draw, seed 20260916, and the SM1S int4
step coder are correct and retained) and pd4's pricer + admission scripts (`experiments/ddm_pd4_*.py`). jrd1's retained
control payloads under `/Volumes/APDataStore/pact/ddm_jrd1/` are read-only inputs; write under
`/Volumes/APDataStore/pact/ddm_jrx1/`. Report free space before every heavy step; retain ≤ 3 GiB with sha256. If pd5's
seal exists when you start, the pointer may have moved: re-derive the base from `reports/latest.md` and the hot
state, and re-pin every number that expires per move.

## OPTIMAL FORM
As jrd1's charter (reference forms pd4 + rw1/ren2 + real coder; SCOPE deltas K = 24 and one discrete step; no MECHANISM
reduction). Provenance pins: jrd1 landing c9bf03e62; jrd1 charter 1c1aba7ce; pd4 memo sha 48767941b8a9cf1e (first 16;
re-verify); move 52 archive ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e, pointer commit d1fc2a1c2.

## Prior negatives accounted (operator 2026-08-15)
jrd1: no scorer slot → UNMEASURED (cured by `--owns-scorer` + firing after pd5); int4 step ≠ 0 B (Definition 1);
rw1/ren2/cb1/pd4/m164/m132/m88 as in jrd1's charter.

Boundaries, serializer rules, the rc-17 continue rule, checkpoint (`ddm_jrx1`), lane
(`ddm_jrx1_joint_exchange_probe_with_scorer_slot_20260916`), and the final-message contract: exactly jrd1's, ending
with the current own-vehicle frontier line (re-derived at start).

<!-- # FORMALIZATION_PENDING: charter, not a finding; exchange rows land as EmpiricalAnchors at harvest -->
