# ddm_gpp1 — generic pretrained prior at decode time

**Status:** COMPLETE. The tested formulation is closed. No candidate fired, no scorer ran, no
Modal job ran, and the exact frontier did not move.

**Verdict scope:** FORMULATION — TorchVision RAFT-small `C_T_V2`, reduced to the preregistered
17-level causal motion/boundary belief and realized by a cold online KT bank after the shipped
move-44 probability row. This is not a family-wide negative for all public priors or all ways of
injecting decoder-side information.

## Conclusion

The causal public-model belief has substantial **oracle** conditional information, but this
formulation cannot turn it into useful counted bytes and is far outside the strict receiver wall.
On all 600 pairs and 117,964,800 exact shipped symbols:

- receiver-Lane context alone saves **17,534.216 B** under LS1's Miller–Madow convention;
- receiver-Lane context jointly conditioned on the RAFT belief saves **25,705.658 B**;
- therefore RAFT's directly measured marginal upper bound is **8,171.442 B**, 31.55% of the
  25,899 B demand and 46.60% on top of the LS1 bound;
- that clears the charter's 8,000 B prototype trigger by **171.442 B**, so a physical prototype
  was mandatory;
- the best physical variant saved **3 B** in the exact RC64 stream, but its 6-byte envelope made
  the retained prototype archive **180,409 B**, or **3 B larger** than move 44;
- the model took **302.492 s** locally. The same-receiver factor projects **295.859 s** on T4,
  which is not a T4 measurement and exceeds the strict 27.581 s incremental allowance by
  **268.278 s**.

This is an upper-bound-to-wire collapse, not a rate win: the oracle uses in-sample hindsight and
free cell probabilities, while the real receiver begins cold and must learn only from already
decoded symbols. The existing receiver also already mixes 23 adaptive families. Do not promote
the 8,171 B bound as removable bytes and do not build a public receiver delta from this row.

## Rule 118 legality and publication boundary

The governing text is exact at `upstream/README.md:114,118-120`. Line 114 gives a 30-minute
official evaluation limit and T4/CPU hardware. Line 118 says external libraries and tools are free
**unless they use large artifacts such as neural networks**, in which case those artifacts should
be in the archive and count; it explicitly applies this rule to PoseNet and SegNet. Lines 119-120
allow arbitrary compression assets and require public-PR submission.

The charter's proposed interpretation — that a public, generic, non-video-derived model is an
external free tool — is not established by that wording. The cleaner literal reading is that this
4,006,189-byte neural-network weight artifact is counted even though it is public and generic.
This memo therefore records a publication risk, not a free-weight finding. Only a written operator
rule-118 ruling can change that classification. A model trained or selected on this contest video,
any model output embedded as free code, and all scorer weights or scorer derivatives remain counted
or forbidden regardless. The prototype is explicitly `candidate_archive=false` and makes no
rule-118 eligibility claim.

## Candidate model screen

| Model | Public size / parameters | Provenance | License finding | Disposition |
|---|---:|---|---|---|
| TorchVision RAFT-small `C_T_V2` | **4,006,189 B** retained weights; **990,162** parameters | TorchVision metadata says trained from scratch on FlyingChairs + FlyingThings3D; generic optical flow, not Lane detection | TorchVision BSD-3-Clause text retained, SHA-256 `6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d` | **SELECTED**: smallest cached real dense per-cell motion prior with clean code/weight custody |
| TorchVision LRASPP MobileNetV3-Large | published **12.49 MiB**, 3,221,538 parameters | COCO subset/VOC semantic labels; no Road/Lane label specific to this task | TorchVision BSD-3-Clause | NOT RUN: larger, uncached, and semantically weak for this cell map |
| PiDiNet | published approximately **2.74 MB** | generic edge detector | repository LICENSE restricts use to research and asks authors about commercial use | EXCLUDED: not a clean permissive license |
| NVIDIA SegFormer-B0 Cityscapes | published approximately **15 MB**, about 3.7M parameters | road-scene semantic segmentation | Hugging Face card says `License: other` | EXCLUDED: license is not cleanly permissive |

Primary public references: [TorchVision RAFT source](https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py),
[TorchVision LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE),
[TorchVision LRASPP source](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py),
[PiDiNet LICENSE](https://github.com/hellozhuo/pidinet/blob/master/LICENSE), and
[SegFormer-B0 Cityscapes card](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024).

The selected weight file is retained at
`/Volumes/VertigoDataTier/pact/ddm_gpp1/model/raft_small_C_T_V2-01064c6d.pth`,
4,006,189 B, SHA-256
`01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27`.
It is not scorer-derived, was not trained on the contest video, and produced no content copied into
free code.

## Bare environment and deterministic bootstrap proof

A clean Python 3.13 environment was created and populated offline from cached distributions at
`/Volumes/VertigoDataTier/pact/ddm_gpp1/bare_venv`. In that environment, `torchvision` imported,
the exact retained weights loaded without download, and `raft_small` constructed with 990,162
parameters. Exact primary pins are `numpy==1.26.4`, `torch==2.12.1`, and
`torchvision==0.27.1`; the resolver installed `filelock==3.32.3`, `fsspec==2026.7.0`,
`Jinja2==3.1.6`, `MarkupSafe==3.0.3`, `mpmath==1.3.0`, `networkx==3.6.1`,
`pillow==12.3.0`, `setuptools==81.0.0`, `sympy==1.14.0`, and
`typing_extensions==4.16.0`.

The installed `RECORD` hashes for the three primary packages are:

| Package | Version | installed `RECORD` SHA-256 |
|---|---:|---|
| numpy | 1.26.4 | `5ebd1c9946412c880fc79977b0a948cc5999825e29297a90e434c1bc6d9d1edf` |
| torch | 2.12.1 | `e20756a4699db198cec19bf81947023dac5494a619224a86f64daf9cbdd713ef` |
| torchvision | 0.27.1 | `50f5ed5704cdb7c9b52d72d1f14cebab65725be24645094b9d60b3a55b99592f` |

Inference used seed 20260911, fp32 CPU, deterministic Torch algorithms, four Torch threads, and
fixed integer postprocessing. Belief payloads repeated byte-identically for pairs 1, 100, 300, and
599. **Not measured:** a from-scratch network bootstrap during public evaluation, T4 wall time, or
local↔T4 belief identity. Those remain blockers even if rule 118 is resolved.

## Exact causal instrument

The source object is the move-44 public decode: archive SHA-256
`04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`,
decoded raw SHA-256
`2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc`, and token
field `subset6.u8` SHA-256
`a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8`.
The pointer commit named by the charter is `99625f32f`; the move-44 pointer memo SHA-256 is
`f7638e1e171e0d19309f0d0b2f2f9f0acf9b5c70c5fb87f14bc029dfbaeaba8b`.

For pair `t > 0`, the model sees only the two RGB frames of pair `t-1` — frames `2t-2` and
`2t-1` — which are fully rendered before pair `t` begins. Pair 0 uses sentinel cell 16. RAFT's two
quarter-pixel flow components are retained. Their L1 magnitude is quantized at q4 thresholds
4/12/32 and their integer flow-gradient magnitude at 2/8/24; crossing the four magnitude bins with
the four boundary bins produces belief cells 0-15. The joint key is exactly LS1's
`receiver_lane` cell crossed with this 17-level belief.

The oracle uses LS1's exact 190-group receiver rows and exact allocation convention. It measures
the joint conditional entropy directly; it never adds independent bounds. Both baseline and joint
cells use in-sample plug-in probabilities plus the Miller–Madow correction, making the result an
optimistic fixed-cell upper bound with free parameters.

## Measured n600 upper bound

**Axis:** `[macOS-CPU advisory / scorer-free n600 receiver probabilities]`.

| Quantity | Bytes | Relation to demand |
|---|---:|---:|
| Exact shipped receiver ideal codelength | 119,748.300 | — |
| LS1 receiver-Lane Miller–Madow gain | 17,534.216 | 67.70% of 25,899 B |
| Joint receiver-Lane × RAFT Miller–Madow gain | 25,705.658 | 99.25% of 25,899 B |
| **RAFT marginal upper bound** | **8,171.442** | **31.55% of 25,899 B** |
| Remaining joint shortfall | 193.342 | 0.75% of 25,899 B |

The marginal attribution is concentrated in Road and Lane, but it is not a Lane-only result:

| Class | Incremental upper-bound bytes | Share |
|---|---:|---:|
| Road | 3,295.423 | 40.33% |
| Lane | 2,743.372 | 33.57% |
| Undrivable | 869.539 | 10.64% |
| Movable | 887.033 | 10.86% |
| MyCar | 376.075 | 4.60% |

The joint atlas has 859,059 occupied cells and 405,708 singleton cells versus 491,314 and 206,209
for the baseline. That growth is another warning that a hindsight atlas is not a realizable causal
code. The complete gain map is retained as a 943,718,528-byte `.npy` payload rather than discarded.

Primary receipts:

- `/Volumes/VertigoDataTier/pact/ddm_gpp1/INFERENCE_RESULT.json`, SHA-256
  `f34fadb2558705f257aebac1b0928c602d7bb2db3c18dd932deba1fe9867724f`;
- `/Volumes/VertigoDataTier/pact/ddm_gpp1/oracle/RESULT.json`, SHA-256
  `6d0af84e028defe25ddbf95b6bae2dcda1c8eb2e55a7e9fd5124d08a66663530`;
- 600 per-pair `.npz` payloads plus receipts under
  `/Volumes/VertigoDataTier/pact/ddm_gpp1/beliefs/`;
- 25-pair atomic oracle checkpoints through `counts_0600.npz` and an immutable `COMPLETE.json`.

## Triggered physical prototype

**Axis:** `[macOS-CPU advisory / scorer-free n600 exact RC64 byte measurement]`.

The copied receiver prototype starts one KT observed/expected table cold for each incumbent
argmax × belief cell. It changes the shipped frequency row with fixed q5 strength 8, 16, or 32 and
updates only after a complete plane has decoded. Every candidate was twin-encoded. The control is
byte-identical to the shipped 119,749-byte RC64 stream.

| Variant | Ideal stream saving | Real RC64 saving | Retained archive | Archive change vs 180,406 B |
|---|---:|---:|---:|---:|
| q5=8 | +2.476 B | **+3 B** | **180,409 B** | **+3 B larger** |
| q5=16 | -0.625 B | 0 B | 180,412 B | +6 B larger |
| q5=32 | -23.062 B | -23 B | 180,435 B | +29 B larger |

The q5=8 decoder reproduced all 117,964,800 token bytes exactly, SHA-256
`a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8`, with zero byte
diff. Candidate twins were byte-identical. The measured local receiver mixer+RC64 time was
48.093 s and total prototype replay was 87.321 s. Model inference was measured separately at
302.492 s locally. No T4 timing was measured; 295.859 s is only the mxo1 same-receiver projection.

The physical boundary is exact token-stream replay over the copied GPP1 mixer and exact shipped
RLC1 rows. The prototype does **not** implement the actual RGB/render/token interleaving in public
`inflate.py`, and the prototype ZIPs are evidence envelopes, not contest candidates. Integration is
unjustified because counted bytes and timing both fail.

The complete physical receipt is
`/Volumes/VertigoDataTier/pact/ddm_gpp1/prototype/RESULT.json`, 6,414 B, SHA-256
`5da220d58fd4e10e683523fbf3d1d9bea0f5978cd9cf640d41c16eec15794b00`.
All streams, twins, riders, members, archives, the copied runtime, decoded field, and 25-frame
checkpoints remain retained. A terminal `--resume-from` reloaded the frame-600 model and seven
encoder snapshots, then revalidated the immutable result without replacing measured timing. The
measured source is retained byte-identically and the non-numerical resume change is recorded at
`prototype/source/PROVENANCE_MIGRATION.json`, SHA-256
`40c8713edc07ace966dbbd5063a1d2c962ff4479c885bb9a2bf52788dc2e8136`.

## Custody exception recovered

The first inference process continued after its supervising terminal stopped returning output. A
mistaken second resume overlapped it at pair 413, so both processes targeted one atomic temporary
name. The original compressed NPZ envelope expected at that instant was lost before detection. The
overlap file and receipt were quarantined; a single isolated regeneration reproduced all five
scientific arrays exactly. Thus the belief payload is preserved, but the first container byte
sequence is not. This is `RECOVERED_WITH_ENVELOPE_LOSS`, not perfect custody. The receipt is
`/Volumes/VertigoDataTier/pact/ddm_gpp1/collision_retained/RECOVERY.json`, SHA-256
`a5c015931ea0259065813e748a16a7f23ff86bec28a128f94a587a43215720a7`.

## Wall-clock adjudication

The charter's original approximately 540-second slack was wrong. The strict receiver ceiling is
1,260 s, and move 44 consumes 1,232.418725255 s, leaving **27.581274745 s**. The local model alone
is 11.0× that allowance, and its non-authority T4 projection is 10.7×. Even the prototype's local
mixer+RC64 time is above the allowance before model inference. This is a hard wall for the tested
formulation, independent of the rule-118 ruling.

## RECALL EVIDENCE

The recall was content-based across the full corpus, not limited to charter seeds. Queries included:

```text
rg -n -i "pretrained|public prior|world model|optical flow|rule 118|external tool|receiver.*render|decode.time.*prior|lane.*prior" .omx/research --glob '*.md' --glob '*.json'
rg -n -i "pretrained|public prior|world model|optical flow|rule 118|external tool|receiver.*render|decode.time.*prior|lane.*prior" .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_* docs .omx/state/canonical_task_status.jsonl
.venv/bin/python tools/list_canonical_equations.py --json
.venv/bin/python tools/canonical_task_status.py --json | jq '<gpp1/public-prior content filter>'
```

Findings beyond the charter's seed list, and what they changed:

- `openpilot_world_model_free_prior_v2_20260629T190505Z.md` already read rule 118 against an
  approximately 30 MB public generic neural model and concluded that shipping the weights is
  ambiguous/likely counted. This prevented the charter's proposed “generic means free” premise
  from being silently adopted.
- `pretrained_driving_prior_lane_scaffold_landed_20260513.md` is an offline/video-derived decoder
  codebook substrate. It is counted content, not precedent for a public decoder-side prior; it was
  excluded from the mechanism claim.
- `ddm_mxo1_free_decode_time_online_context_mixing_20260911.md`, SHA-256
  `60b5a34bb6146af82b346b591cfb836694f3736a9e7b1e2083d024fda223f565`, measured only
  27.581 s of strict T4 receiver slack, found the 23 incumbent adaptive families, and realized just
  368 B with its best new causal learner. This changed wall-clock from a late check to an immediate
  rejection gate and prevented online adaptation from being described as new.
- The canonical equation registry contains
  `wyner_ziv_decoder_side_information_conditional_entropy_savings_v1`,
  `token_rate_model_direction_dependence_v1`,
  `decoder_derivable_ideal_savings_ceiling_v1`, and
  `token_tail_context_mixing_bound_v1`. They support measuring the joint conditional ceiling but
  explicitly do not turn a decoder-derivable ideal saving into charged bytes. This kept the oracle
  and physical conclusions separate.
- No canonical task-status row matching `gpp1`, `generic pretrained`, or `decode-time prior` was
  found in the currently served 940-row task-ledger scope. The arm queue row is therefore the
  lifecycle authority for this charter; no unrelated canonical task row was fabricated.

## Disposition and next charter

`ddm_gpp1_free_generic_pretrained_prior_at_decode_time` is **LANDED / COMPLETE** with the tested
formulation closed. `ddm_gpp2_compact_public_prior_reopen` is **QUEUED-WITH-A-FIRE-ORDER**, owner
MAIN, consumer store `/Volumes/VertigoDataTier/pact/ddm_gpp1/`. It may fire only after all three
preconditions exist: (1) written operator clearance that the exact public weight artifact is free
under rule 118, (2) a concrete permissively licensed model demonstrates exact T4 n600 incremental
wall ≤27.581 s including bootstrap, and (3) a preregistered causal n600 screen predicts at least
8,000 B marginal value over the exact current receiver rows. Its first experiment must select or
gate the incumbent 23 family outputs with the belief; it must not retry the cold KT bank measured
here.

## LIVE-HYPOTHESES

- A much smaller public boundary/road model could preserve enough of RAFT's Road+Lane signal to fit
  the 27.581 s wall. This is plausible because 73.90% of the oracle marginal comes from those two
  classes and the final belief is only 17 integer cells, but no qualifying model has been timed.
- Conditioning the incumbent 23-family mixer on the public belief may realize more than the cold
  post-row KT table. This is plausible because the oracle signal is real and the shipped mixer has
  richer pre-mix directions; it remains untested and must clear the 8,000 B/27.581 s fire order.
- A coding order that renders a current non-Lane carrier before Lane tokens could expose more
  same-pair information than the previous-pair RAFT belief. This is plausible because it removes a
  full-pair temporal lag, but public receiver causality, exact rendering order, and physical gain
  have not been demonstrated.

## DEAD-ENDS

- RAFT-small `C_T_V2` as a full n600 decode-time prior is closed for this formulation: its local
  time and projected T4 time exceed strict slack by an order of magnitude.
- A cold observed/expected KT bank keyed by incumbent argmax × 17-level RAFT belief at q5 strengths
  8, 16, and 32 is closed: the best stream saved 3 B and its archive grew by 3 B.
- Treating the 8,171 B Miller–Madow marginal as removable bytes is closed: a real causal coder
  realized 0.037% of that amount before envelope cost.
- The approximately 540-second decoder-slack premise is closed: the binding 1,260-second policy
  leaves 27.581 s on move 44.

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)

## Independent verification (arm resumed on Opus, 2026-09-11)

The codex predecessor's run was audited before this memo was committed; nothing above was taken on
trust. Re-derived from primary artifacts, not from the memo's own text:

- **Instrument identity.** The baseline recomputed here, **17,534.216 B**, equals LS1's independently
  published `receiver_lane` MM row to three decimals (`ddm_ls1_...20260911.md` line 164, memo SHA-256
  `b729faa146b62ea394ec502be37fb0d66d11f577202431cfcd9d5618f79f40bc`, matching the charter pin). The
  producer binds `LS1_ROOT=/Volumes/VertigoDataTier/pact/ddm_ls1`, whose `rows/` holds 1,200 files
  (600 `.npz` + 600 `.json`) — n600, not a prefix.
- **Real stream, real coder.** `read_frame` re-verifies each row's receipt fact, refuses on drift,
  asserts frequency mass `== TOTAL`, and asserts `bits == -log2(freq[symbol]/TOTAL)` — the shipped
  coder's own frequency rows, not a proxy model. `symbols = 117,964,800 = 600 x 512 x 384`.
- **UNION != SUM honored.** The incremental value is `base.mm_bits - joint.mm_bits` computed
  directly (producer line 565); no independent bounds are added. Arithmetic re-checked:
  `817,712.676 - 752,341.138 = 65,371.538 bits = 8,171.442 B`.
- **Positive control.** The prototype fail-closes with `INSTRUMENT_FALSIFIED` unless the control
  RC64 stream reproduces the shipped stream byte-identically; `control.rc64` on disk is
  119,749 B, SHA-256 `c499972a33dac497412c18839b632a8e1bbf75d518528039db0e2aca80c8eb13`,
  matching the receipt. Candidate twins are required to agree.
- **Producer-sha exception resolved, not waved past.** `prototype/RESULT.json` pins producer
  `37fb7bdd...` while the working file is `df8ca6c9...`; the measured source is retained
  byte-identically at `prototype/source/ddm_gpp1_physical_prototype_v1.py` (`37fb7bdd...`) and
  `PROVENANCE_MIGRATION.json` records `numerical_or_receiver_change: false`
  ("terminal-resume validation added after the measured run"). `ddm_gpp1_runtime_prior.py`
  (`479ca5be...`, 4,263 B) matches its `prototype/INPUTS.json` pin.
- **Weights are the genuine public artifact.** `raft_small_C_T_V2-01064c6d.pth`, 4,006,189 B,
  SHA-256 `01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27` — the filename carries
  TorchVision's own published download hash prefix `01064c6d`.
- **Slack re-derived at source:** mxo1 lines 33/38 give `1260 - 1232.418725255 = 27.581274745 s`.

Two quantities the memo did not state, and they sharpen the closure:

1. **If rule 118 counts the weights, the cost is decisive on its own.**
   `25 x 4,006,189 / 37,545,489 = 2.667557 S` — **19.44x the entire current score** (0.13724). No
   byte saving of the measured magnitude survives that; a favorable ruling is a precondition for the
   family, not a detail.
2. **The wall-clock closure is independent of that ruling.** 302.492 s local is **10.97x** the
   27.581 s allowance; the non-authority T4 projection 295.859 s is **10.73x**. The formulation is
   closed on timing whichever way rule 118 is read.

Realization ratio, stated plainly: the physical coder realized **3 B of the 8,171.442 B bound =
0.037%**, and the 6 B envelope made the archive **3 B larger**. A 3 B stream delta also sits inside
the measured container-break noise (sd 34.8 B), so it is not evidence of a real gain.

`ruff check --select F` passes on all three producers. Axis unchanged: `[macOS-CPU advisory]`,
`score_claim=false`. No Modal job, no scorer run, no contest eval, no candidate archive.
