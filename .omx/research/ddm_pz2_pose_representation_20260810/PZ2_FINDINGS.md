# ddm_pz2 pose representation — DALI-axis findings

**Verdict:** the information in the six official DALI PoseNet targets is not close to the current
23,384-byte pose-carrier section. In the measured scalar-quantizer formulation, a parse-back-exact
direct packet is **1,817 B** at quantization MSE `2.32888e-5`, and the best joint additive-error
bracket is **2,860 B** at quantization MSE `6.91224e-7`. This is a representation opportunity, not
an archive win: no PR130 receiver consumes these target packets, no frames were changed, no scorer
ran, and the frontier did not move.

All target and packet measurements below are `[macOS-CPU scorer-free representation measurement;
official DALI GT targets, n600]`. All projected scores are `[TOY-BRACKET over
contest-CUDA,DALI,n600 base components; no receiver/scorer]`.

## What was measured

The source was the official DALI cache
`/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/caches/gt_cache_600_official_ada.pt`
(117,981,301 B, SHA-256 `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`),
key `pose`, shape `600x6`. The PR130 archive pin reproduced at 191,052 B and SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`; the read-only intake
HEAD reproduced at `e34f31bc4969042c0051ac81aa3c56884419a231`. The base raw pin is
receipt-verified only: 3,662,409,600 B and SHA-256 `a18eb42a...c0353`; its source bytes were
success-cleaned and were not consumed here.

The apparatus exhaustively allocated 0–16 uniform scalar bits per dimension and measured three
integer-stream charts, all with Brotli-q11: direct, first difference, and second difference. It
materialized 306 component streams and 919 actual Pareto packets. Every raw stream, compressed
stream, packet, and packet repeat was retained; 2,455/2,455 manifest entries passed current
existence, byte-count, and SHA-256 validation.

## Score sensitivity and bit allocation

At the base `d_pose=2.331e-5`,

`d sqrt(10*d_pose) / d d_pose = 327.4906216`.

Therefore `1e-6` aggregate `d_pose` costs locally `0.000327491 S`, or **491.832 archive bytes**.
The scorer averages all 3,600 errors, so every output dimension has exactly the same score
sensitivity: `54.5817703 S` per unit dimension-MSE, or **81.972 bytes per `1e-6` dimension-MSE**.
The square root changes the booked score, but because it is monotone it does not change the
allocation that minimizes aggregate MSE at a fixed byte cost.

The dimensions receive different bits because their variances differ, not because the scorer
weights them differently:

| Pose output | DALI variance | Minimum-byte allocation at `qMSE <= base d_pose` | Best additive-bracket allocation |
|---:|---:|---:|---:|
| 0 | 1.58159517 | 10 | 12 |
| 1 | 1.277224e-3 | 5 | 6 |
| 2 | 8.950838e-4 | 3 | 6 |
| 3 | 9.142494e-5 | 3 | 5 |
| 4 | 5.459999e-5 | **0** | 5 |
| 5 | 8.190096e-4 | 3 | 5 |

Dimension 4 can be represented by its mean in the coarse packet because its total variance costs
less aggregate MSE than another bit on a larger coordinate. This is the measured water-allocation
effect; it is not a free dimension in the scorer.

The seeded n120 scope check used PK2's 24-per-120-block stratified selection (seed `20260809`),
never a prefix. Aggregate n120/full-n600 qMSE ratios were `0.9273` for the 1,817-byte point and
`0.9689` for the 2,860-byte point. The packet conclusions use the full n600 population; n120 is
only a scope-bias screen.

## Entropy and achieved packets

| Operating precision | Allocation | Gaussian R(D) model | Empirical zero-order marginal entropy | Actual direct packet |
|---|---|---:|---:|---:|
| `qMSE <= 2.331e-5` | `[10,5,3,3,0,3]` | **1,328.38 B** | **1,438.75 B** | **1,817 B** |
| best additive bracket, `qMSE=6.91224e-7` | `[12,6,6,5,5,5]` | not used as acceptance | **2,378.58 B** | **2,860 B** |

The Gaussian number is an independent-coordinate model reference, exact only for that model. It
is neither an achieved payload nor a universal lower bound. The empirical entropy number is the
zero-order coding reference for the measured quantized symbols. The actual parse-back-exact packet
is the admissible upper point.

A DALI-specific KLT was remeasured because the older KLT result was not on a proved DALI axis.
Correlation is real (maximum absolute off-diagonal correlation `0.529696`), but KLT reduces the
Gaussian model from 1,328.38 B to 1,278.14 B, only **50.24 B**. Its video-derived float32 basis is
144 B (Brotli is 148 B), so KLT is **93.76 B worse** after counting the basis. Verdict scope:
`INSTANCE`, official DALI n600 target tensor at base d_pose under the Gaussian model.

## Temporal structure

Lag-1 correlations by output are `[0.606, 0.792, 0.869, 0.689, 0.552, 0.983]`, but the corresponding
first-difference standard deviations are `[0.886, 0.646, 0.511, 0.789, 0.946, 0.183]` times the
level standard deviations. Only dimension 5 is strongly smooth. At the same base-adequate
allocation `[10,5,3,3,0,3]`, actual packets are:

| Chart | Bytes | Delta from direct |
|---|---:|---:|
| direct | **1,817** | 0 |
| first difference | 2,163 | +346 |
| second difference | 2,340 | +523 |

At `[12,6,6,5,5,5]`, the same comparison is 2,860 / 3,564 / 3,845 B. Temporal differencing is
therefore closed only for this scalar-quantizer plus Brotli formulation on this DALI tensor.

## Joint score pricing

The following rows assume, only for arithmetic, that the entire 23,384-byte carrier section can be
replaced by the target packet without receiver or container overhead:

| Candidate | Packet | qMSE | Hypothetical archive | Perfect-realization S | Additive-error S | Worst-aligned S |
|---|---:|---:|---:|---:|---:|---:|
| `direct_p040_b10-5-3-3-0-3` | 1,817 B | 2.32888e-5 | 169,485 B | 0.1577738 | 0.1640999 | 0.1730414 |
| `direct_p092_b12-6-6-5-5-5` | 2,860 B | 6.91224e-7 | 170,528 B | **0.1458367** | **0.1586999** | **0.1611043** |

The second row is the best measured additive-error bracket: `-20,524 B`, `-0.0136661 S` rate, and
`-0.0134414 S` net versus the inherited base. The cheapest direct packet whose perfect-realization
bracket is below 0.15 is 2,408 B (`direct_p067_b10-5-5-5-3-4`, qMSE `4.97128e-6`, projected
`S=0.1499574`). None of these is a score. The receiver-realization error and receiver bytes are
unmeasured and may consume the apparent gain.

## Frame parity and scorer disposition

Frame parity is `NOT_RUN_NO_FRAME_REALIZATION`. A PZ2 packet decodes exactly to its quantized target
codes, but no PR130-compatible receiver consumes it to create even pose-carrier frames. Calling
target parse-back “frame parity” would be fake. Consequently no candidate is scorer-ready and no
scorer slot was claimed or used.

`direct_p092_b12-6-6-5-5-5` is `QUEUED-WITH-A-FIRE-ORDER`, blocked before scorer. Fire only after a
byte-closed receiver consumes the retained packet, deterministic decode passes twice, every odd
frame is byte-identical to the base raw, and all even-frame differences are declared. The machine
row is `PZ2_SCORER_QUEUE.jsonl`.

## RECALL EVIDENCE

Stores consulted: research memos/receipts, canonical equations, canonical research index, DAG FEED
surfaces, task/council/docs stores, and implementation code.

- Full-store query:
  `.venv/bin/python tools/corpus_query.py "PR130 pose target representation entropy temporal waterfill DALI six PoseNet outputs" --stores research,equations,memory,dag,council,tasks,docs --top 40 --json`.
  Retained result: `/Volumes/VertigoDataTier/pact/ddm_pz2_pose_representation_20260810_v3/recall_query.json`,
  SHA-256 `484f973a02988270feacef188c7f3a9fe96599809bc9f159d4bf12aec8a26af6`.
- Content searches included
  `rg -n -i '(pose.*(entropy|waterfill|fixed.point|delta|temporal)|reverse water|KLT)' .omx/research experiments`
  and the same pose/representation terms over `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, the hot
  state, design specs, and task ledgers. Canonical equations were enumerated with
  `.venv/bin/python tools/list_canonical_equations.py --json`.
- Beyond the charter seeds, recall found
  `.omx/research/ddm_pose_optimal_allocation_derivation_20260801.md` (commit `ad0c6df9d2f`),
  `.omx/research/order_exploit_rate_budget_20260627T053101Z.md` (commit `6da318bcc89`), and
  `.omx/research/ddm_pu1_pose_underpricing_and_tail_20260803.md`. They changed the plan: remeasure
  every target statistic on the official DALI cache; keep the monotone square-root outside the
  allocation ordering; rerun rather than transfer KLT; and test actual direct/delta payloads rather
  than assuming drive smoothness.
- Reused inputs were PK2's stratified n120 receipt (implementation commit `cfddfc503a7`, final
  receipt commit `c21d39b48d9`), RC2's coder closure (`52cb73adc0a`), and the paired-eval corrected
  ledger (`7e4f7a8c388`). No sister file was modified.

## Custody and reproducibility

- Runner: `experiments/ddm_pz2_pose_representation_20260810/run_pose_representation.py`; seed
  `20260809`; `--resume-from` implemented.
- Full receipt:
  `/Volumes/VertigoDataTier/pact/ddm_pz2_pose_representation_20260810_v3/PZ2_MEASUREMENT_RECEIPT.json`
  (SHA-256 `4c835983985cf58878765ae2e748d0b392d4f94390d1bff300dfc7120c8d832a`).
- Payload manifest:
  `/Volumes/VertigoDataTier/pact/ddm_pz2_pose_representation_20260810_v3/payload_manifest.json`
  (SHA-256 `9d2619518f07f70032052afef97da24d11309778db5f75ba09644e8d59099dc1`).
- Resume repeat left the full receipt byte-identical at SHA-256 `4c835983...d832a`.
- Four immutable stage checkpoints are retained. Nothing was deleted. Earlier v1/v2 apparatus
  outputs are retained separately and superseded by v3; they are not cited as final evidence.
- Static payload-retention scan of the runner: 0 findings. Current manifest validation:
  2,455 examined / 2,455 valid / 0 invalid.

## Conclusion

The target scalars themselves need roughly 1.8–2.9 KB in the measured achieved formulation, not
23,384 B. The current section is expensive because it is a learned frame realization carrier, not
because six target scalars are intrinsically expensive. PZ2 therefore closes cheap coder,
temporal-delta, and counted-KLT rewrites, but it does **not** close or replace the carrier. The only
live route exposed here is a jointly trained, byte-closed target-conditioned receiver that can turn
one retained direct packet into valid even frames without paying the 20 KB gain back in weights or
distortion.

Frontier unchanged: `S=0.172141297491896447 @ 191,052 B [contest-CUDA,DALI,n600]` inherited PR130
base; not remeasured, not moved.
