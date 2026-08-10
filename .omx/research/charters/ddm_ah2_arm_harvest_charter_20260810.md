# ddm_ah2 — harvest the ~41 unread finished arms (operator 2026-08-10 "Harvest all signal")

## Mission

Task #1006 measured the cost of unread arms: the last partial harvest (12 of 53)
alone produced the lc2 exact row (S 0.16959899569230852 @ 187,226 B, now the
pointer's CUDA anchor) and cp2's 182,364 B composed candidate. ~41 finished arms
remain unread. Read ALL of them and route every finding. The bar CONTEXT changed
today: PR #135 at 0.162 is the new effective frontier; our row is +0.007599
above it — re-rank every harvested candidate against 0.162, not 0.172.

## Method (binding)

1. Enumerate finished arms: `.venv/bin/python tools/codex_arm_queue.py status`
   + each arm's final message (`-o *.last.txt` capture per the np1 surface) +
   its landed branch/commits (`git log --all --oneline --grep=<arm>`) + its
   memo under `.omx/research/`.
2. Per arm, ONE disposition row: {LANDED-CONSUMED / FINDING→ROUTED (name the
   consumer task/arm) / CANDIDATE→PRICED vs 0.162 (bytes, projected S, exact
   control owed) / EMPTY-honest}. Per the harvest law, every follow-on exits
   FIRED / FOLDED / QUEUED-WITH-FIRE-ORDER — no UNKNOWN rows.
3. Head items to adjudicate explicitly: cp2's 182,364 B archive (−4,862 B vs
   lc2 → projected S ≈ 0.166362 IF distortion holds — needs the exact-axis
   control); the −903 B lossless lever; any arm whose finding composes with
   the lc2 base or attacks the +0.007599 gap to PR135.
4. Durable memo `.omx/research/ddm_ah2_arm_harvest_20260810.md` + commit via
   tools/subagent_commit_serializer.py (post-edit --expected-content-sha256,
   tags [no-triality] [p0-ledger-ok]). Sister arms' artifacts are APPEND-ONLY.
   Checkpoint per the subagent protocol.

## OPTIMAL FORM

Reference form: the prior 12-arm harvest pass this session (produced the lc2
row; receipts in `experiments/results/ddm_lc2_exact_row_20260810/`, commit
151bccd6f4). This charter is SCOPE-extension to the FULL population (41 arms),
no mechanism reduction. Provenance pins: pointer anchor sha
f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45; PR130 bar
archive sha 0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd.
PRIOR-LAW PREDICTION (from the 12-arm sample's hit rate — 2 majors in 12): the
41 unread arms hold ≥3 route-worthy findings and ≥1 candidate that re-ranks
against the 0.162 bar; if the sweep comes back empty, that refutes the sample's
hit rate, not the harvest law.
