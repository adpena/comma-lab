# ddm_ax2 — ground-truth decoder axis reconciliation

UTC: 2026-08-09T12:14:56Z  
Scope: scorer-free source and receipt reconciliation; no eval, scorer call, dispatch, launch, archive build, promotion, pointer edit, upstream edit, or PR130-intake edit.  
Pointer delta: none.

## Answer first

The public leaderboard is not a single CPU or CUDA axis. The official workflow exposes a runner choice: `ubuntu-latest` sets `EVAL_DEVICE=cpu`, while `linux-nvidia-t4` sets `EVAL_DEVICE=cuda` (`upstream/.github/workflows/eval.yml:18-33,87-89`, SHA-256 `8a6cd6300b51a44f36b49774bc0c6100dbb37ef8290d42bf8e584f1dceddce56`). The README says GPU-required inflation runs on a T4 and otherwise runs on CPU, then ranks the public leaderboard (`upstream/README.md:108-121`, SHA-256 `68ea239d7333696e79716e47a9c4288d2918efbcd8912f78932b0befe0af872b`). Because `upstream/evaluate.py` maps CUDA to `DaliVideoDataset` and every other device to `AVVideoDataset` (`upstream/evaluate.py:21-42,58-60`, SHA-256 `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`), the public ranking mixes DALI-GT and AV-GT rows.

PR130 is a DALI-GT row. Its bot receipt says `device: cuda`, 600 samples, `d_pose=0.00002331`, `d_seg=0.00029660`, and 191,052 bytes (`/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/pr130_comments.txt:54-73`, SHA-256 `50566d66327da43a274802420bcbf5855ddc51a1624ccd6276e36cbd1595440e`). Recomputing from those published rounded components gives `S=0.17214129749189644`; the campaign shorthand `0.1721417` is slightly less precise. No published PR130 contest-CPU row was found in the read-only PR130 intake and the bounded PR129-132 intake (`.omx/research/public_pr129_132_intake_20260725.md:34-49`, SHA-256 `5342b08a72e7fe0a75ed34f64488eb2cff5f3770bccb12a5b6b5039c0469669e`).

The banked-row census selected 53 whole-`S` rows. Against PR130's DALI axis: 42 are AV, 4 are DALI, and 7 are `UNKNOWN-with-reason`. Thus 42/46 GT-resolved rows (91.30%) are axis-mismatched to PR130; the honest unreconciled remainder is 7/53 (13.21% of the selected corpus). The row-level evidence is `AX2_AXIS_ROWS.jsonl`.

For #984, the clean target is axis matching: use a DALI ground-truth build and require `S_DALI < 0.17214129749189644` from the published PR130 components. If #984 instead stays on AV ground truth and makes a cross-axis beat-PR130 claim, the fixed-archive decoder-swap upper bound must be cleared strictly: `0.17214129749189644 - 0.055021456790339165 = 0.11711984070155727`, so require `S_AV < 0.11711984070155727`. Using the campaign shorthand bar gives the equivalent rounded rail `S_AV < 0.11712024320966084`. The `0.055021456790339165` quantity is a triangle-inequality **upper bound on fixed-archive score movement under a GT-decoder swap**. It is not a measured score delta, a score, a loss, or recoverable score.

## 1. Which axis does the contest rank on?

It ranks the public leaderboard across whichever official runner each submission uses, so the answer is **mixed official axes**, not CPU-only or CUDA-only. The workflow defaults to CPU but permits the T4 runner, and the README assigns the T4 to GPU-required inflation. The effective frontier selector currently takes the minimum across local CPU, local CUDA, and the public leaderboard (`.omx/state/canonical_frontier_pointer.json:3-17`, SHA-256 `7898ea668e18e2c95ffc5dca7f08056207f340fba05312bef837d85d0f0d6eec`), which is also an explicit cross-axis comparison.

Consequences in plain language:

- The local PyAV/`frame_utils.yuv420_to_rgb` pipeline is correct by construction for a CPU-run official row. The old worry that PyAV necessarily confounds a CPU-axis result is closed.
- The same AV pipeline is mismatched for comparison to PR130, because PR130's official row used CUDA and therefore DALI ground truth.
- The registry claim “leaderboard ranks by CPU” is stale. It remains in `cpu_axis_optimal_archive_selector_v1` (`.omx/state/canonical_equations_registry.jsonl:11`, SHA-256 `c7e3ff80543b751fa47a1fed74a6e5feddab51ce8540b0e272c5c10f98096a91`) and in `src/tac/optimization/cuda_cpu_axis_calibration.py:4-19` (SHA-256 `42f496203feb943746fe306a6f684c5376b44b3fce8fdab6be134de25f4faf0f`). They are queued for supersession, not silently rewritten in this scorer-free audit.

## 2. What axis is PR130's 0.1721417?

`DALI`, `[contest-CUDA]`. The primary bot receipt is quoted above. The read-only intake independently records the same device and says no PR130 CPU row was published. The public leaderboard's displayed `0.172` is therefore a rounded CUDA/DALI result, not an axis-neutral number.

## 3. Which banked S rows are mismatched?

### Selection denominator

- `.omx/state/probe_outcomes.jsonl`: the append-stable first-675-line census prefix contained 675 physical rows, 383 latest rows after selecting the last append per `probe_id`, 35 latest rows whose metric name contains `score`, and 18 selected whole-`S` totals after excluding deltas, ratios, components, design-completeness values, z-scores, and drift units. Prefix SHA-256: `f8be1693495fa012e9f499d0db02b910f54aa7e3e0743676ea81c15380754565`. Later append-only rows do not change this bounded snapshot.
- Frontier pointer: 3 selected top-level rows. The embedded public leaderboard entry list is a display snapshot and was excluded to avoid double-counting external rows.
- tq1c: 27 rows: one measured baseline plus all 26 realized candidate scores. The 26-row SSD ledger is SHA-256 `e0427f27beae82340dcdc9e17d472d51d32984416e409ebda3c76e15078ca15c`; the receipt names the baseline and denominator at `.omx/research/ddm_tq1_20260805/tq1c/RECEIPT.md:7-14,24-33`, SHA-256 `907a30206f324f611e34ee6ba007590529fd92e608e0ac66de3bbc54dac90e30`.
- ms8 lineage: 3 composed-`S` rows at `.omx/research/ddm_ms8_menu_selector_solver_st_codebook_20260802.md:35-60`, SHA-256 `fd9127f0ef4ad4567e05ee7371d5f1796c2550a74b9a5924eea4e738ff91bdca`. The ms8 row is explicitly predicted and remains labeled that way.
- pu2 lineage: 2 measured end-to-end `S` rows at `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md:81-106`, SHA-256 `7138aabcd32cfb2e33dfb264a31b72e5c72347393422a1a4ae5df070a783b8fb`; its GT stream is explicitly `0.mkv` via `frame_utils.yuv420_to_rgb` at lines 709-713.

Total: `18 + 3 + 27 + 3 + 2 = 53` selected rows.

### Resolution result

| GT decoder | rows | share of 53 | relation to PR130 |
|---|---:|---:|---|
| AV | 42 | 79.25% | mismatched |
| DALI | 4 | 7.55% | matched |
| UNKNOWN-with-reason | 7 | 13.21% | unresolved; excluded from resolved mismatch denominator |

Resolved denominator: 46. Mismatched: `42/46 = 91.30%`. Unknown rows are two CPU+CUDA aggregate scores without a single GT decoder, two predicted CPU-composition rows without an eval invocation, and three MLX-prefilter totals whose inspected receipts do not state the target-cache decoder. This is the declared wall-clock reduction. No UNKNOWN row was graded from a filename or axis label.

## 4. What target margin must #984 clear?

The same-host T4 receipt measured AV-vs-DALI GT disagreement, not a candidate score:

- seg disagreement `1.7523023416288197e-04`, contributing `0.017523023416288197` to the metric bound;
- pose MSE `1.4061325055081397e-04`, contributing `0.03749843337405097`;
- sum `B=0.055021456790339165`.

The source receipt is `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/result_summary.json`, SHA-256 `15f43860a2d0a32bd7191ee12f3f1f1308cf3345090a4ad1cf2f3bb67bc5aa2c`. Since both terms induce metrics, triangle inequalities give `|S_DALI(A)-S_AV(A)| <= B` for a fixed archive `A` (the byte term cancels). Therefore:

- same-axis DALI claim: `S_DALI(#984) < S_DALI(PR130) = 0.17214129749189644`;
- cross-axis AV claim robust to the decoder ambiguity: `S_AV(#984) < S_DALI(PR130)-B = 0.11711984070155727`.

Equivalently, an AV margin below or equal to `0.055021456790339165` is inside the decoder-swap bound and cannot support “beats PR130.”

A concurrent research-signal row appended during this audit measured only the seg leg on one PR130 renderer checkpoint and reported a much smaller realized AV↔DALI seg difference (`0.00034077962239584866` S), while explicitly leaving the realized pose difference unknown (`.omx/state/probe_outcomes.jsonl:675`, final census SHA above). It is `[macOS-MPS realization-fidelity]`, not contest authority, and it cannot tighten the total bound without the pose leg. It does show why the preferred #984 action is a direct DALI comparison: axis matching removes the need to spend against a worst-case cross-axis guarantee.

## 5. Can live paths mix the axes within one comparison?

Three distinct answers matter:

1. **Official upstream evaluator: no literal within-run GT mix.** One device is resolved, one `DefaultDatasetClass` is selected, and that one class constructs `ds_gt`; `ds_comp` is always `TensorVideoDataset` (`upstream/evaluate.py:21-42,58-69`).
2. **The DALI-vs-AV probe: yes, deliberately and visibly.** `experiments/modal_dali_av_gt_cache_diff.py:195-267,295-322` constructs AV then DALI on one CUDA host to compare their caches. This is the positive-control experiment that produced the bound, not a score path.
3. **A live legacy wrapper has a silent axis-definition mix.** `experiments/auth_eval_renderer.py` maps an actual CUDA scorer device to `contest_cuda` at lines 91-141, but `score_with_upstream` hard-codes `AVVideoDataset` for ground truth at lines 544-596 and then moves both tensors onto the scorer device. Its result payload copies the device-derived axis label at lines 810-824. Thus a CUDA invocation is AV-GT plus CUDA network forward while being labeled `[contest-CUDA]`; it is not equivalent to upstream CUDA/DALI. The file SHA-256 is `41b61b8d6cd5f851fbcdee04d4a55bbf606157573460f21fc8ed224586716bae`. The bounded executable search found this path and intentional decoder-drift probes; it did not find a second production evaluator in `upstream/`, `experiments/`, `tools/`, or `src/tac/` that silently combines AV and DALI batches into one scalar. This is a scoped negative, not a global nonexistence claim.

Verdict scope: **FORMULATION** — the `auth_eval_renderer.py` wrapper's device-only axis labeling is invalid for CUDA comparisons. This does not impugn `upstream/evaluate.py`, which selects the matching decoder.

## Whole-measurement routing

Every field from the same-host receipt is preserved with a named downstream consumer in `RECEIPT.json#/measurement_routes`: coverage; environment; both elapsed times and their 9.214138× ratio; both cache byte counts and SHA-256 values; AV and DALI pose minima/maxima for all six dimensions; seg disagreement; pose MSE; maximum pose absolute delta; all six pose-dimension MSEs; dim-0 fraction; both metric-bound components; and the total upper bound. The existing probe-outcomes row `dali_vs_av_decoder_gt_ambiguity_same_t4_host` already holds the headline bound, components, hashes, coverage context, and per-dimension MSE in notes (`.omx/state/probe_outcomes.jsonl:667`, census snapshot SHA above). Canonical ingestion of the non-headline range and timing fields is queued until the already-dirty shared ledger is released; no unrelated ledger changes were absorbed into this commit.

## RECALL EVIDENCE

- Searched the full research corpus, canonical research index, sub-0.15 DAG, live state, and task/probe ledgers for `DaliVideoDataset`, `AVVideoDataset`, `PR130`, `contest_cpu`, `contest_cuda`, `axis`, `gt_cache`, `auth_eval_renderer`, `cpu_axis_optimal_archive_selector_v1`, and `cpu_cuda_score_gap_v1`.
- Ran `.venv/bin/python tools/list_canonical_equations.py --json` and found beyond the charter seeds that `cpu_axis_optimal_archive_selector_v1` still asserts CPU-only ranking, while `cpu_cuda_score_gap_v1` already treats CPU/CUDA as separate archive axes. This changed the plan by adding a canonical-equation supersession fire order.
- The source-body search found beyond the charter seeds the AV-GT/CUDA-label defect in `experiments/auth_eval_renderer.py`. This changed the reciprocal answer from “no official in-run mix” to a scoped live-wrapper confound and added a preflight repair fire order.
- The pointer's `min(CPU, CUDA, public)` selection rule is an explicit cross-axis join; it was included as a DALI row plus two local axis rows rather than treated as one axis-neutral score.
- A concurrent row beyond the charter seeds measured a small realized seg-only swap on one MPS renderer checkpoint but left pose unknown. It changed the wording by distinguishing the conservative total guarantee from vehicle-specific, non-authority realized effects; it did not change the DALI-first target.

## Boundaries and pointer honesty

No new score was measured. Neither same-host cache is challenge-host authority; the hosted evaluator rebuilds its own GT. No claim here promotes macOS, MLX, CPU labels, or CUDA-forward-device labels into a contest row without the matching decoder invocation. The official contest pointer and own-vehicle pointer are unchanged.

Own-vehicle frontier: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
