# DDM BG2 — EC2 introduced-error postmortem and fork execution (2026-08-14)

**Disposition:** `FOLDED` — BG1 v2's four-channel bilinear gate on the existing CP135 8-D frame state is not admitted.  
**Routed successor:** `QUEUED-WITH-A-FIRE-ORDER` — #978 coupled multi-token Seg representation.  
**Authority:** `[macOS-CPU analysis; no scorer forwards; no score claim]` over all 600 retained CP135/EC2 label fields.

## Result

EC2 fixed 12,075 base errors but introduced 52,854 errors at previously correct pixels, leaving 75,749 endpoint errors and a net increase of 40,779 errors. The introduced errors are strongly structured by class, edge, and image location, but the pre-registered causal question was narrower: after those exposures are controlled, does CP135's existing 8-D per-frame state predict which frames suffer more introduced errors?

It does not. A five-fold, seeded, shuffled, non-prefix held-out model using 217 exposure features reached OOF R² 0.427685. Adding only the existing 8-D frame state reduced OOF R² to 0.409237, an incremental R² of -0.018448. In the pre-registered 10,000-label-permutation control, 8,978 null deltas were at least as large as the observed delta, for one-sided p=0.897810. The gate required positive held-out incremental R² and p≤0.01. Both requirements failed.

The admissible conclusion is formulation-scoped: the current CP135 frame embedding is not a supported conditioning variable for BG1's proposed multiplicative `4×8` bilinear frame gate at the EC2 endpoint. This does **not** show that frames are globally unstructured, that nonlinear conditioning is dead, or that all new frame/token state is useless. It closes this gate formulation and routes the remaining Seg problem to a jointly learned receiver-native multi-token representation under #978.

## Input custody and reconstructed masks

Every named input was SHA-256 verified before it was loaded:

| Object | Bytes / shape | SHA-256 |
|---|---:|---|
| EC2 full endpoint argmax | 117,964,928 B; `uint8[600,384,512]` | `803a1d8755cafcf31b03d8ad1494d49f89f6e4fb2115341423308e0db20b3a1a` |
| EC2 batch receipts | 67,055 B | `ffa88ae4478727edb9df89a35f023407ddc1f6cdc8c029530448fdb601087b55` |
| EC2 trained adapter | 1,369 B | `9559c2ab5128f193c8b0c754c5d61851b7784070fa049e04cf48cfd157eead82` |
| CP135 base argmax | 117,964,928 B; `uint8[600,384,512]` | `7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727` |
| GT argmax | 117,964,928 B; `uint8[600,384,512]` | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` |
| CP135 existing frame embedding | 19,328 B; `float32[600,8]` | `1c46c81b2529d6608d879b1f1084c0cf0a2fe79b1beaa8e4a9e45a9fd6977135` |

For every pixel, with `B` the CP135 base label, `E` the EC2 label, and `G` the target:

- `fixed = (B != G) & (E == G)`
- `introduced = (B == G) & (E != G)`

All 38 contiguous batch ranges covering `[0,600)` were reconstructed independently. Their retained `argmax.npy`, `target.npy`, and `base_error.npy` payload hashes all match the sealed batch receipts. The full reconstructed masks contain 12,075 fixed and 52,854 introduced sites, exactly matching BG1's totals.

## Decomposition of the 52,854 introduced errors

### Frame distribution

Introduced counts average 88.09 per frame (standard deviation 26.16, minimum 39, maximum 443). Variance/mean is 7.77, Gini is 0.1403, and the top 10% of frames carry 15.41% of introduced errors. Frame 522 is the largest outlier at 443; the next largest counts are frame 518 at 207, frame 519 at 199, and frames 138 and 516 at 168 each. These are descriptive facts only; they do not satisfy the conditioning-mechanism gate.

### Class and directed-edge distribution

By ground-truth class, the introduced sites are Road 42,184 (79.81%), Lane 1,608, Undrivable 5,520, Movable 2,878, and MyCar 664. By EC2's resulting class, they are Road 7,654, Lane 31,677 (59.93%), Undrivable 8,604, Movable 3,749, and MyCar 1,170.

The dominant directed failure is Road→Lane: 31,542 sites, or 59.68% of all introduced errors, over 27,393,519 Road opportunities (rate 0.00115144). The next largest are Road→Undrivable 6,737, Undrivable→Road 4,584, Road→Movable 2,822, Movable→Undrivable 1,850, Lane→Road 1,502, and Road→MyCar 1,083. Thus the regression is not a class-agnostic spray; it is dominated by the Road/Lane decision surface with smaller collateral on other inter-class edges.

### Spatial distribution

The retained 12×16 grid of 32×32 cells is also nonuniform. The largest cell is `(row=5, column=8)` with 3,042 introduced sites (opportunity-normalized rate 0.004983), followed by `(5,9)` with 2,453, `(6,12)` with 2,421, `(5,7)` with 2,263, and `(6,3)` with 2,255. Spatial structure is therefore real, which is why spatial exposure was controlled before testing the 8-D state.

Full per-frame, class, directed-edge, unordered-edge, spatial-count, opportunity, rate, and coordinate-frequency payloads are retained rather than summarized away.

## Held-out mechanism test

The response is each frame's `introduced_count / base_correct_opportunities`. The exposure-only design has 217 columns: five GT-class exposures, twenty directed four-neighbor GT-boundary exposures, and 192 spatial-cell exposures. Five outer folds are seeded, shuffled, non-prefix, and stratified by quartile of `baseline_error_count / base_correct_count`. Exposure ridge strength is selected only inside each outer training fold. The 8-D state is then fitted by OLS only to inner-OOF exposure residuals from that training fold, with training-only standardization.

| Pre-registered quantity | Result | Gate |
|---|---:|---|
| Exposure-only OOF R² | 0.4276850053 | descriptive control |
| Exposure + existing 8-D state OOF R² | 0.4092369915 | must be >0 |
| Incremental OOF R² from 8-D state | -0.0184480138 | must be >0; **failed** |
| 10,000-permutation one-sided p | 0.8978102190 | must be ≤0.01; **failed** |

The null incremental-R² median is -0.010282, its 90th percentile is -0.002463, and its 99th percentile is 0.004528. The observed delta is worse than 89.78% of shuffled frame-state assignments. Two per-edge diagnostics had small positive deltas—Road→Undrivable +0.001845 and Road→MyCar +0.004803—but those diagnostics were not individually permutation-controlled and were pre-registered as unable to control the fork.

Plainly: class/edge/location exposures explain repeatable frame variation; the current 8-D CP135 frame state adds no significant held-out information after those controls. The data support neither the BG1 v2 gate nor the stronger phrase “frame-unstructured.”

## Fork execution

### BG1 v2

`FOLDED`, verdict scope `FORMULATION`: do not seal, train, price, or scorer-fire the four-channel bilinear gate conditioned on the existing CP135 8-D frame embedding for this base/EC2 endpoint. The pre-registered training ceiling of 543.597 seconds / $0.15 was never consumed because the admission gate failed before launch.

### #978 coupled multi-token Seg representation

`QUEUED-WITH-A-FIRE-ORDER`, owner `#978`, consumer store `#978 receiver-native representation store, then #984`:

1. Claim the #978 lane and register a new retained SSD store before any launch.
2. Build a seeded, stratified-random `n≥32` receiver-closed screen in which jointly learned multi-token semantic support/representative/probability state is compared with the HC1 direct representative at matched pose. No explicit changed-site list may be transmitted.
3. Persist every candidate payload plus hashes, parsed receiver state, commands/configs, split identities, and Seg/Pose/rate deltas. The screen fires onward only if sign-verified Seg survival improves after parse-back at matched pose.
4. On that positive trigger, run one resumable retained short joint train with periodic and stage-end checkpoints, then whole-container pricing. Only after that evidence is byte-closed and the scorer lane is free may one n600 authority row be queued.

This is a concrete queue, not an assertion that #978 already works. No scorer slot, paid dispatch, renderer call, or lane claim was consumed by BG2.

## Retained payloads and reproducibility

The authoritative root is:

`/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/bg2_postmortem_r3/`

Its `ARTIFACT_MANIFEST.json` has SHA-256 `90f04b6d260383b3d3ca5ddd3d2bdd1b7cb98a214633a83ce9b0928ff60e63e2`, records 182 artifacts totaling 910,531,810 bytes before the manifest itself, and is unchanged by an immediate `BG2_RESUMED_VERIFIED_COMPLETE` rerun. Including the manifest, the retained root is 910,568,058 logical bytes. Key receipts are:

| Retained receipt | SHA-256 |
|---|---|
| `decomposition/decomposition_tables.json` | `bd772ee147aaa876d84245db59c9b37fe443b1be4bded76c053b928c22b89b03` |
| `model/HELDOUT_MODEL_RESULT.json` | `117eda872c484baa2697b4b7bc6fa5eadf474a785dd74fa5674bb8d05c23ab52` |
| `permutation_control/PERMUTATION_RESULT.json` | `6b90ce3ad14b5554984b9258ca09cb0174589550de37d31057bccfde1a77ddf4` |
| `batch_recreations/BATCH_VALIDATION.json` | `bad1c5d31689a9b137dab9d26337b2d714e112cf4db90f1e1c49548480572361` |
| retained runner | `2909a7b1229e8a9f67be2103031aa5f2fd949032ca974801f5483bdfd0449615` |

Two earlier numerical runs are preserved but explicitly invalid for verdict use. R1 emitted non-finite matrix-operation warnings and is marked by `INVALID_FOR_VERDICT.json` SHA `996736fc761bfd282527e36e2e579002ef8758078a953305e55b93bdac64fb32`; R2 exposed one remaining unguarded held-out matrix product and is marked by SHA `6a4c071c4feba53249dcf9e9e661b66bd17e40f90ea84a1b1f82c8dd26e19d87`. R3 added fail-closed finite assertions and completed without warnings. No payload was deleted or substituted.

## Recall evidence and novelty accounting

The decision used the charter/common contract, `PROGRAM.md`, the operating manual, live hot state, BG1, EC1, JS1C, HP4, PK3/PK4, QS4, RFO1, the canonical-equation index, the DAG/index, and bounded task/ledger searches.

- PK3/PK4 barred an in-sample “the state correlates” admission, so this fork uses held-out incremental predictivity and a label permutation control.
- HP4 had already closed cheap post-hoc state compression at its tested scope but left interaction open; BG2 tests the named interaction mechanism without reopening compression.
- QS4 established neighbor collateral as a real risk but did not transfer a verdict to EC2; BG2 therefore measures full class/edge/spatial collateral directly.
- RFO1 records that the live receiver already mixes semantic and latent state. #978 is not claimed as a novel slogan; its unresolved work is a jointly learned receiver-native multi-token representation that survives parse-back and beats HC1 at matched pose.
- The equation index was searched, but no canonical algebraic identity can replace this empirical frame-label dependence test. No equation was promoted or revised.
- Bounded searches did not find an already-registered concrete #978 artifact path in the inspected index/DAG/task surfaces. The successor must claim and register its SSD store before fire; this is not an ownerless-work claim.

## Authority and frontier

This postmortem performs label-field analysis only. It ran zero scorer forwards and did not execute `upstream/evaluate.py`, a renderer, Metal, Modal, or a paid job. It produced no archive and cannot move any frontier. The live own-vehicle frontier remains LC2 at `S=0.16959899569230852`, 187,226 bytes, `[contest-CUDA T4, n600]`; the effective frontier remains the borrowed CP135 row at `S=0.16195513827824176`, 186,252 bytes, `[contest-CUDA T4, n600]`.

The required serializer was invoked once with the post-edit content SHA, `--base-content-sha256 ...=new`, `[no-triality] [p0-ledger-ok]`, and the Markdown review override. Git failed before staging with `unable to create temporary file: Operation not permitted` / `failed to insert into database`, the known managed-sandbox Git-write blocker. Per the common contract it was not retried or bypassed. This memo is therefore an uncommitted working-tree artifact; the pre-existing staged index remains untouched.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `#978`; consumer store: `#978 receiver-native representation store, then #984`; fire trigger: after claiming the lane and registering a fresh retained SSD store, execute the seeded stratified-random `n≥32` receiver-closed multi-token-vs-HC1 screen, and advance to a resumable short joint train only if the parsed candidate has sign-verified Seg improvement at matched pose.

## LIVE-HYPOTHESES

- A jointly learned multi-token semantic representation may separate Road/Lane support without BG1's scalar gate because EC2's dominant 31,542 Road→Lane regressions show a specific contested decision surface, and exposure-only features already explain substantial held-out variance.
- The small positive diagnostic deltas for Road→Undrivable and Road→MyCar may indicate narrow interactions not represented by the aggregate 8-D state, but they require separately pre-registered held-out controls before they can justify any actuator.
- Spatially concentrated cells may be useful strata for #978's screen because they carry much higher introduced-error rates, provided selection is defined before outcomes and the result is still judged after receiver parse-back at matched pose.

## DEAD-ENDS

- BG1 v2's four-channel bilinear gate on the existing CP135 8-D frame embedding is closed at formulation scope for the CP135/EC2 endpoint: incremental held-out R² is negative and permutation p=0.897810, far outside the admission gate.
- Descriptive clustering alone cannot admit the gate: frame, class, edge, and spatial concentration are real but are already partly captured by exposure controls and do not establish dependence on the current frame state.
- The two positive per-edge diagnostic deltas cannot rescue BG1: they were not independently permutation-controlled and were pre-registered as non-fork-controlling.
- The invalid R1 and R2 numerical outputs cannot support any conclusion: both are durably marked invalid, and only clean R3 is verdict authority.

OWN-VEHICLE FRONTIER: unchanged at LC2, `S=0.16959899569230852`, 187,226 bytes, `[contest-CUDA T4, n600]`.
