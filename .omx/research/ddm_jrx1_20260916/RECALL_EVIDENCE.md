## RECALL EVIDENCE

The original recall was completed before proposal generation. `RECALL_SEARCHES.json` records literal argv, result hashes, line counts and return codes; retained output lives in `recall/`. This is a bounded content search over the full named corpus, not a claim that every memo was read.

| Surface | Query / tool | Retained output |
|---|---|---|
| Research memos and receipts | `joint.{0,50}(renderer|field)|renderer.{0,50}collateral|selection.on.{0,15}price|UNION.{0,8}SUM|price.first|zero.net.coded`, `.md/.json`, first 4 hits/file | `recall/memos.txt`, 488 matching lines |
| Canonical equations | `.venv/bin/python tools/list_canonical_equations.py --json` | `recall/equations.txt`, 490 registry rows; selected relevant laws in `selected_equations.json` |
| Index and DAG | `joint|renderer|collateral|real.*price` over `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*`, first 5 hits/file | `recall/index_dag.txt`, 18 lines |
| Design/SPEC | same query across `docs` and `.omx/research`, `*design*` and `*SPEC*`, first 3 hits/file | `recall/design.txt`, 622 lines |
| Task ledger | `jrx1|jrd1|pd5|pd4|joint.renderer|joint.field` | `recall/tasks.txt`, 5 rows |

Beyond the charter seeds:

- `ddm_jf1_joint_field_model_refit_20260823.md` and `ddm_rj2_joint_renderer_object_change_20260823.md` contain earlier joint mechanisms. JF1's epoch 2 field/model byte leg was +7,554 stream bytes at its named rung; RJ2 is a different n1 DX2 renderer/pose object. They do not measure this move 52 field/int4 joint direction. This prevents the overclaim that joint motion has never been attempted anywhere.
- `ddm_xr1_exchange_ratio_noise_floor_20260903.md` separates deterministic same-input encoding from across-candidate container variation. The ±35B band is an admission convention for different objects; it does not excuse mismatching identical encoder twins or make residual bytes free.
- The newly terminal pd5 continuation found a cheaper clustered token price, 8.976 bits/token, but its best composition reached only 0.883 of its admission bar. It did not produce a new pointer. Its source store remained read-only, its slot was released before this arm ran, and its field-only result does not close joint renderer/field motion.
- The registry's `compensated_semantic_edit_exchange_v1` requires the nonlinear square-root pose term for finite changes. `token_edit_composition_subadditive_on_pair_overlap_v1` forbids summing isolated benefits as composition. `renderer_edge_layer_foldback_reach_v1` keeps rw1's collateral negative scoped to its actual int4 action. We preserved those boundaries and never used single-axis sums as a measured joint row.
- The design/DAG/task search surfaced PC2's closed S1 diagonal, on its older object, and jrd1's pending scorer-phase fire order. This arm consumed the scorer assignment; it cannot inherit an older formulation's negative.

The pd4 source and retained price rows also changed the control interpretation. Its 12.0 bits/token is the pooled price of a selected 18-pair/24-token archive, not the median of random K24 pair prices. Its numerator includes resolved pose plus seg;11 of 12 sampled overlap winners have zero seg-only benefit. Frame-local adaptive ledger deltas rank proposals; full archive re-encodes are charges. Repeat-sheet observations measure context spill; selecting their cheapest observation is forbidden. No equation or frontier anchor is promoted from an incomplete control.
