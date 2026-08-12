# ddm_js2 implicit edge conditioning — blocked before the solve

## Result first

The exact pointer did not move. F1 fired before any proposal was evaluated: the retained local CP135 control has **50,395 flips**, while the promoted shipping row has **34,964 flips**. The difference is **15,431 flips**, or **44.13%** of the promoted count, against the charter's **1%** maximum.

The outcome is **BLOCKED_INSTRUMENT**, not a negative on implicit conditioning. No edge-conditioned archive, no ON/OFF pass, no advisory winner, and no T4 candidate row exists.

The machine receipt is `.omx/research/ddm_js2_20260812/INSTRUMENT_VALIDATION.json`. Its durable copy and retained control payloads live under `/Volumes/VertigoDataTier/pact/ddm_js2_20260812/`.

## Instrument validation

| control fact | promoted shipping axis | available local axis | conclusion |
|---|---:|---:|---|
| CP135 archive | 186,252 B, SHA `6eb1a3b7…edb6` | identical archive | archive identity passes |
| population | 600 pairs / 117,964,800 pixels | same | denominator passes |
| Seg flips | 34,964 | 50,395 | fails by 15,431 flips |
| d_seg | 0.0002963935 from exact flip count; report 0.00029643 at 8 dp | 0.0004272037 | local Seg contribution is 0.013081 S higher |
| rendered `0.raw` | SHA `604459e2…e1a` | SHA `a641d1ef…d47` | receiver/render axis already differs before edge proposals |
| device and environment | Tesla T4 CUDA, batch 16, threads 2, torch 2.9.0+cu128 | macOS arm64 CPU, no CUDA or MPS, torch 2.12.1 | not a 1:1 instrument |

The et4 batch-shape law explains the one-pixel C1 cache disagreement, but it cannot repair this baseline. Changing only scorer batch size cannot turn one complete 3.66 GB raw stream into the different promoted raw stream. The unavailable shipping renderer/device axis is the load-bearing mismatch.

The promoted harvest preserves the raw hash and scalar components, but the searched custody scope did not contain the promoted `0.raw` or argmax bytes. Without those bytes, the Road-hub prior cannot be applied to the shipping-axis error cells. Running the solve anyway would optimize a different objective.

No new scorer job ran. The prior n600 control was reused because its archive, raw, logits, argmax, and receipt are complete and hash-bound. Repeating that settled control on the unchanged local surface would spend another full scorer pass without changing the missing authority axis.

## m94 claim-unit scope

Two capacities must not be conflated:

- The local diagnostic instrument covers all **117,964,800 macOS-CPU argmax pixels** for the exact CP135 archive. It supports an INSTANCE statement about that local surface only.
- The solve needs **117,964,800 contest-CUDA argmax pixels**. This run has **zero retained contest-CUDA argmax pixels**. Its shipping-axis instrument capacity is therefore 0 against an object capacity of 117,964,800, so a solve verdict is refused.

Verdict-scope ladder:

- **INSTANCE:** this local macOS-CPU instrument is closed as a js2 solve-admission surface.
- **FORMULATION:** no verdict; no edge-conditioned candidate was admissibly evaluated.
- **FAMILY:** no verdict.
- **PARADIGM:** no verdict.

This is an instrument blocker, not evidence that implicit conditioning fails.

## Deliverables and falsifiers

1. **Instrument-validation receipt:** complete; `BLOCKED_INSTRUMENT`, F1 fired.
2. **Edge-conditioned joint solve:** correctly not started. `state.json` records zero passes and a fail-closed resume gate.
3. **Candidate queue for MAIN:** complete and empty. F1 makes any candidate row fabricated; the control archive is listed separately and is not called a candidate.
4. **This memo:** complete with m94 fields, scope ladder, recall, custody, and honest boundaries.
5. **Skeleton annex line:** queued for MAIN in `.omx/research/ddm_js2_20260812/SKELETON_ANNEX_QUEUE.md`; the skeleton itself was not edited.

Pre-registered falsifiers:

- **F1 FIRED:** the instrument disagrees by 44.13% and cannot be repaired locally because no CUDA/MPS device or promoted raw/argmax custody is available.
- **F2 NOT REACHED:** no matched ON/OFF pass ran, so the formulation is not closed.
- **F3 NOT REACHED:** no advisory winner exists and no T4 gate row was requested.

## Candidate queue and payload custody

`CANDIDATE_QUEUE.json` contains zero entries and zero T4 requests. The empty denominator is explicit: **0 candidates materialized / 0 candidates eligible**.

The js2 SSD store retains:

- the exact CP135 control archive, 186,252 B, SHA `6eb1a3b7…edb6`;
- the complete local diagnostic argmax field, 117,964,928 B including the NPY header, SHA `b8f063eb…647a`;
- the GT argmax custody copy used by the diagnostic;
- `INSTRUMENT_VALIDATION.json`, `state.json`, the milestone checkpoint, the empty candidate queue, and the skeleton annex queue.

No candidate payload existed in memory, so none was discarded. The 3.66 GB local raw and 2.36 GB logits remain retained in the predecessor store and are bound by the js2 receipt rather than copied needlessly. The promoted raw payload was not available locally; only its SHA manifest was harvested.

## RECALL EVIDENCE

Sources searched before adjudication:

- the full charter seeds: fd135, every js1 skeleton annex, the shipping-axis law, cp135, and m91;
- the common contract, live board, active lane claims, evaluator, canonical equations registry, research index/DAG/task-ledger surfaces;
- corpus queries by content for `implicit edge`, `edge-conditioned`, `joint int12`, `basis FiLM`, `quantize compensate`, `Road Lane`, `batch shape`, `BLOCKED_AXIS_MISMATCH`, `#995`, and `m94`;
- the complete sr1 memo and na6 negative audit;
- the PR135 ExperimentBook inventory plus its joint-solve, margin-allocation, CBQ, FiLM-compensation, and F26 iterative-solve sources;
- the exact cp135, t1r1, and pass-4 CUDA harvest receipts and the local js1 payload receipt.

Findings beyond the charter seeds changed the plan:

1. **sr1 and na6 close only post-hoc rate conditioning.** Their −2 B causal-edge and +43 B scalar-pose results do not close the distortion-side joint proposal family. This prevents over-broad closure here.
2. **The promoted CP135 raw hash is retained, but its bytes are not.** The local raw has a different hash. This identifies the mismatch before the scorer and makes a batch-only repair insufficient.
3. **The F26 reference solver itself requires CUDA and batch 16.** Its eight-pass coordinate descent is a shipping-axis procedure, not a CPU-portable objective. This converts the missing CUDA surface from an inconvenience into the charter's F1 blocker.
4. **m94's canonical surface refuses richer object claims.** Zero shipping-axis argmax pixels cannot support a 117,964,800-pixel edge-allocation verdict.
5. **The exact lossless rate residue is already closed by cp135/lp135.** No rate-side substitute was reopened while the seg instrument was blocked.

No canonical equation displaced the axis law or the full-population control. The gap equation still says seg must supply about 0.004 S after the pose ceiling, but it cannot authorize a cross-axis solve.

## Borrowed-substrate accounting

The CP135 archive, PR135/F26 receiver, HPAC semantic model, int12 carrier, basis, FiLM state, scorer weights, GT cache, and prior js1 control are borrowed in-repo or granted PR135 substrate. js2's work is the independent instrument crosswalk, m94 refusal, custody packet, and fail-closed routing. It claims no new codec, solver result, or candidate.

## Boundaries

**Measured or directly verified:** archive identity; full-population local flip count; promoted reference flip count and report components; local and promoted raw hashes; environment/device mismatch; absence of CUDA and MPS locally; zero candidate denominator; retained file sizes and hashes.

**Not measured:** any edge-conditioned proposal, d_seg or d_pose trajectory, matched ON/OFF passes, dry-pass noise, new complete archive bytes, T4 transfer ratio, or new contest score.

The effective frontier remains **cp135 `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`**. The own-vehicle frontier remains **lc2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`**. This unit did not reach sub-0.15.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN shipping-axis instrument owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_js2_20260812/instrument_validation_cuda/`. Fire trigger: a 1:1 batch-16 T4 CUDA lane with artifact return, or retained promoted CP135 raw plus argmax custody, becomes available; retain the full n600 shipping argmax, rerun only the control, and start matched edge-conditioning ON/OFF passes only if flip disagreement is at most 1%.**

## LIVE-HYPOTHESES

- Distortion-side implicit conditioning remains plausible because F26's own accepted FiLM-plus-carrier compensation moved seg on the shipping axis, while sr1 tested only additive probability calibration.
- Road-hub proposal weighting may still pay on CUDA because m91's independent full-population decomposition puts 87.8% of flips on Road-incident interfaces, but the ordering must be rebuilt from shipping-axis argmax custody.
- Quantize-then-compensate may preserve a real seg proposal because PR135's accepted mechanism hard-quantizes first and re-solves nearby counted DOF; a CPU proposal is not evidence until the same CUDA instrument admits it.

## DEAD-ENDS

- The existing macOS-CPU control as a solve-admission instrument is closed at INSTANCE scope: 44.13% flip disagreement exceeds the 1% gate.
- Treating the one-pixel et4 batch-shape seam as the whole repair is closed: the complete local and promoted raw streams already differ by hash.
- Running edge-conditioned proposals before the control passes is closed by F1; it would be instrument overfit.
- Standalone additive edge/pose probability tables are closed at FORMULATION scope by sr1/na6 and were not retried.
- More CP135 lossless-coder hunting is closed by cp135/lp135 and cannot replace the missing seg axis.
