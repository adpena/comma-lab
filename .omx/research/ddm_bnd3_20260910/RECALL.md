# RECALL EVIDENCE

Owner ddm_bnd3. Queries and raw bounded results: `recall_queries.json` and
`equations_recall.json`. No result here is a new score or a transferred byte price.

- Research memos and arm handoffs: content query `address (cost|bytes|packet|term)|joint edge assignment|gap bridging|boundary.segment`, recursively over `.omx/research/**/*.md`.
- Canonical equations: executed `.venv/bin/python tools/list_canonical_equations.py --json`, then retained records matching `boundary_segment`, `address cost`, `joint assignment`, `token_tail_context_mixing`, `context_model_reorder`.
- Research index and graph: `boundary.segment|joint.assignment|gap.bridg|QPAIR|QEVENT` against `.omx/research/CANONICAL_RESEARCH_INDEX_20260629.md` and `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`. The first unsuffixed index path did not exist; the corrected dated path was searched. No matches for this exact query in that scope.
- Design/SPEC surfaces: the same recursive research search includes `SPEC*`; also searched `docs/` with the graph query. No additional matching construction found in the docs query scope.
- Task ledger: `boundary.segment|bnd2|bnd3` over `.omx/state/canonical_task_status.jsonl`; consumed bnd1 ITEM_2 and bnd2's completed/landing dispositions. No other active bnd3 owner found at entry.
- Durable Codex memory registry: `bnd3|address.term|joint.assignment|common_contract|ddm_bnd` returned no hits. No memory-derived scientific conclusion used.

Beyond the charter seeds, **AD2** (`ddm_ad2_addressing_cost_decomposition_20260822.md`, especially its QEVENT section) explicitly establishes that an interleaved shared compressed stream has no unique physical address/value split. This changed the decomposition: retain the original indivisible block, and price independently coded address/content streams as a new representation. AD2's NR1 reorder win also motivated the matched columnar serialization control; no AD2 byte number transfers to move37.

`boundary_segment_recode_price_v1` confirms the original 121,200 B price is a segment packet plus an overwhelmingly larger causal remainder, not a 121,200 B address packet. The TC1 context and reorder laws reinforce same-field and causal-state binding. OR1's row-start price is scoped to its different field/representation, and OF1's short arclength is only a prior here. GS3 Addendum 19 overstates the tested six greedy grammars if read as an all-representation closure; this charter's verdict must preserve that distinction.

Technical implementation reference: [SciPy milp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html). The installed SciPy passes the explicit one-thread and fixed-seed HiGHS options through with a visible warning. No external candidate payload, weights, labels, or video-derived representation was imported.
