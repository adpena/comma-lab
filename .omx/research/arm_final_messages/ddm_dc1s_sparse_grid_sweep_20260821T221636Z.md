The full n600 sweep closes sparse-grid Family A at the measured FX5 scope.

- Actual packet: **388,326 B**
- FX5 HPAC member: **113,777 B**
- Net loss: **274,549 B**
- Required +3 KiB gate shortfall: **277,621 B**
- Projected added search time: 8.69 s, within both wall budgets
- All 190 groups were individually negative
- No receiver, archive, scorer run, or score claim was made

The complete handoff is [ddm_dc1s_sparse_grid_sweep_20260821.md](/Users/adpena/Projects/pact/.omx/research/ddm_dc1s_sparse_grid_sweep_20260821.md). The retained packet and repeat are byte-identical with SHA-256 `9ca6e59e789abdd0c02c70c3d5d52d2b0da917518f03f792b3bbcc31c30fa839`.

Implementation and custody artifacts:

- [ddm_dc1s_sparse_grid_sweep.py](/Users/adpena/Projects/pact/experiments/ddm_dc1s_sparse_grid_sweep.py)
- [failed-run custody record](/Users/adpena/Projects/pact/.omx/research/ddm_dc1s_failed_frame1_state_v1_custody_20260821.json)
- [retained result](/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/full_frame_sparse_sweep/retained/result.json)

Ruff, compilation, self-tests, payload-retention census, resume replay, 600 ledger comparisons, and two review passes succeeded. The serializer commit did not land: the managed checkout refused Git index/object writes with `Operation not permitted`. The three artifacts remain verified, unstaged, and uncommitted.

Own-vehicle frontier remains `fx5_e1`, S=`0.14823186109359`, 180,386 B, `[contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **READY-TO-LAND** — owner: MAIN; consumer store: repository HEAD at `/Users/adpena/Projects/pact`; fire trigger: the checkout permits `.git` index/object writes; action: rerun the serializer for the three artifacts using current SHA-256 values `2e4ca257…`, `cb638dc0…`, and `323efb4d…`, retaining the documented false-positive secrets adjudication.

## LIVE-HYPOTHESES

- An implicit shared task-cell certificate could avoid the measured 227.4 KB position field; this is plausible because location metadata was the largest debt.
- A quotient/certificate representation could amortize the 101.8 KB per-block length table if widths are receiver-derived rather than stored.
- Family B/C may remain viable after the jo1u harvest because they alter the representation boundary instead of tuning this closed packet family.

## DEAD-ENDS

- Monolithic hash/free-run remains closed by the 26.2–27.9 reachable-rank-bit wall.
- Sparse-grid Family A with explicit positions, lengths, SHA-prefix questions, and block sizes 1/2/4/8 is closed at full n600 FX5 scope.
- Block-size retuning is closed: variable selection gained only 1,434 B over fixed b=8.
- Independent adjacent-group factorization is invalid because later conditional rows depend on earlier decoded observations and adaptive state.
- Receiver/archive construction for dc1s is stopped by the mandatory +3 KiB gate.