# DDM EC2 — sparse-event-conditioned HPAC, scorer-free build and MAIN fire order

**Status:** `READY_TO_FIRE` · build/preflight complete · MAIN Metal result pending · pointer unmoved.

EC2 is implemented as a thin subclass runner in
`tools/run_ddm_ec2_sparse_event_hpac_conditioning.py` (post-edit SHA-256
`1dd28a1a0bc489f4f69a94a96a204f58fc0d199e01973f93044ecd77cf35b553`). It preserves the pinned
CL1 integer-HPAC topology, unwarped previous-decoded-partition prior, 60-epoch lambda-1 schedule,
EMA deployment rule, and exact Range receiver. It adds one real local learned channel:
`conv_event = IntegerConv2d(1, 64, 3x3)`. The new convolution starts at zero, is trained jointly,
is self-compressed with the rest of the model, and consumes a binary mask decoded from the counted
EC1 coordinate payload. This is not a replacement context and not a metadata-only label.

No scorer was imported or run. No exact contest row, d_seg, d_pose, or S was measured. The result
of this arm is a byte-real, resumable build and an explicit MAIN fire order, not goal progress.

## MEASURED BUILD RECEIPT

| Quantity | Result | Authority / boundary |
|---|---:|---|
| Banked CL1 Range payload | 116,716 B, SHA `ac2c549c…` | MEASURED real n600 Range bytes; banked control read-only, not retrained |
| Banked CL1 model XZ | 15,088 B, SHA `b74be4d…` | MEASURED retained model payload; supplementary custody fact, not a reinterpretation of the charter's 116,716 B bar |
| JS5 proposal rows verified | 200/200 | MEASURED scorer-free custody: index row, in-store event bytes, proposal receipt, EC1 parse-back |
| EC1 sites | 234 with multiplicity / 234 unique | MEASURED over the retained 200 proposals, drawn from EC1's stratified n32 frame sample; not an n600 event-population census |
| Counted coordinate payloads | raw 447 B · Brotli-q11 413 B · XZ 473 B | MEASURED real coder outputs, `[macOS-CPU scorer-free]`; all primaries and repeats retained; exact decode equality |
| Selected coordinates | 413 B, SHA `485e78cf…` | MEASURED Brotli-q11 winner; receiver-consumed and charged in every complete package |
| Retained training mask | 117,964,928 B `.npy`, SHA `856785fc…` | Derived/debug evidence on SSD; never shipped or charged; encoder re-derives and equality-checks it from the 413 B payload |
| XI2 full-scale peak | 4,879.953 MiB, SHA-bound safe-run receipt `ee06cff3…` | MEASURED matched n600 Metal substrate anchor |
| EC2 governed cap | 6,144 MiB RSS / 6 GiB projected | 1,264.047 MiB (25.90%) margin above the measured XI2 peak; live EC2 peak pending |
| Focused tests | 6/6 pass | CPU unit/integration surface: coordinate codecs, package framing, strict law, learned local consumption, self-compression, fire command, cleanup retention |

Durable CPU-side evidence:

- `/Volumes/VertigoDataTier/pact/ddm_ec2/BUILD_RECEIPT.json`, SHA `e5e80f71…`
- `/Volumes/VertigoDataTier/pact/ddm_ec2/READY_TO_FIRE.json`, SHA `f2aab767…`
- `/Volumes/VertigoDataTier/pact/ddm_ec2/retained/contexts/coordinate_custody_manifest.json`
- `/Volumes/VertigoDataTier/pact/ddm_ec2/retained/CLEANUP_RETENTION_MANIFEST.json`, SHA `9a6f2ad2…`
- `/Volumes/VertigoDataTier/pact/ddm_ec2/queue/main_metal_fire_order.json`

## RECEIVER AND COMPLETE-CONTAINER LAW

The EC1 coordinates are video-derived and are not derivable from the banked CP135/CL1 decoded
partition. EC2 therefore counts them. Only occupancy coordinates are carried: EC2 does not consume
source class, target class, or event-type metadata, so charging those unused fields would be fake
accounting in the other direction.

The terminal receiver consumes one binary `EC2PKG1` object:

`20-byte header + model.xz + tokens.range + tagged coded coordinates`

The package stage materializes and retains a complete container for every coordinate coder, parses
all sections back exactly, chooses by actual complete-container bytes, and emits an independent
winner repeat. The decoder starts from that selected package, restores the added convolution from
the model payload, decodes the coordinate mask, consumes the Range stream, requires
`RangeDecoder.maybe_exhausted()`, and requires the n600 raw-token SHA to equal the canonical CL1 raw
partition `c5c7671d…`.

The preregistered falsifier is implemented literally:

`complete EC2PKG1 bytes < 116,716`

The largest passing integer is 116,715 B. Equality fails. Failure is
`FORMULATION_CLOSED_FULL_SCALE` with verdict scope `FORMULATION`. The 116,716 B threshold is not
silently enlarged by the banked 15,088 B model; it is the operator-specified complete-container bar.
Consequently the learned local context must save enough token bytes to pay both its full learned
model and its 413 B coordinates plus framing.

This strict accounting directly incorporates XI1's counterexample: at matched d_pose, CAP1 was
22,242 B, learned CPR1 was 22,327 B (CAP1 won by 85 B), and the counted geometric-xi package was
30,072 B (+7,745 B versus CPR1). A learned prior is not admitted because its conditional entropy
looks better; sparse context must amortize in the receiver-consumed package.

## P0 RESUME, DETERMINISM, RETENTION, AND DISK HYGIENE

- Seed `20260716`; `PYTHONHASHSEED=0`; deterministic Torch algorithms; MPS fallback forbidden.
- `--resume-from auto` resolves the newest immutable epoch checkpoint under the EC2 custody root.
- Checkpoints preserve live weights, terminal-candidate EMA shadow, optimizer/scheduler, shuffle
  generator, Torch/MPS/Python/NumPy RNG, history, input/source identity, and causal-state digest.
- Initial stage-start, every epoch, continuous-stage end at epoch 30, and discrete-QAT-stage end at
  epoch 60 are distinct payloads. `latest.pt` is only the replaceable resume pointer.
- Every materialized coordinate/model/Range/package candidate and deterministic repeat is retained.
  Exact decoded raw output is retained; later replays get distinct retained filenames.
- UTF-8 paths must already be NFC; AppleDouble `._*` and `__MACOSX` paths fail closed. The binary
  package does not encode filesystem names or metadata.
- `cleanup_stage` writes a SHA-256 retention manifest. It deletes nothing: checkpoints, counted
  context, candidate/repeat payloads, and exact-decode evidence are preservation-mandated; an
  uncertified partial also remains `KEEP` under certify-or-block.

## RECALL EVIDENCE

### Surfaces and queries searched

- Full `.omx/research/` and arm-final-message corpus by content for `HPAC`, `learned prior`,
  `event coordinate`, `sparse event`, `context`, `conditional`, `complete package`, `CAP1`, and
  `rule 118`.
- Canonical equation registry via `tools/list_canonical_equations.py --json`, including
  `partition_temporal_transport_amortization_jitter_bound_v1`,
  `worldsheet_transport_residual_event_rate_v1`,
  `ddm_lp1_deepest_home_context_waterfill_v1`,
  `decode_determinism_integer_arithmetic_v1`, and the L30 constriction Range-coding precedent.
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC surfaces, task-ledger files,
  and `.omx/state/main_hot_state.md` for the same query family. I did not find a separate EC2 task
  row in those bounded index/DAG/ledger scopes; EC1's memo carries the explicit fire-order-3
  learned-context successor and the charter is the governing EC2 row.

### Material findings beyond the charter seeds

- `ddm_sr1_implicit_edge_conditioning_20260811.md`: post-hoc explicit edge calibration saved only
  2 charged bytes; a 74,408 B explicit mask was already known to be dead. This kept EC2 inside the
  probability model rather than adding a post-hoc edge section.
- `ddm_tf1_theoretical_floor_and_beyond_20260812.md`: generic raster temporal conditioning was
  negative (453,449 B global versus 356,636 B intra), but boundary/local learned priors remained an
  explicit reactivation surface. This selected sparse local conditioning, not a global raster prior.
- `ddm_ec1_event_coordinate_producer_20260812.md` plus the EU3 fresh-eyes receipt: EC1's full SP1
  event object is 44,410 B and a standalone sidecar is dead. EC2 therefore uses only the 234 sites
  actually present in the 200-proposal consumer store and charges their 413 B coordinate code
  inside the probability-object package. These denominators are different and are not conflated.
- `ddm_hp3_hpac_section_and_zip_frame` receipts: model and token deltas interact nonlinearly at the
  complete package boundary. This caused EC2 to race actual fully framed containers, not sum an
  estimated token win after the fact.
- `ddm_hm1_20260810/FINAL_REPORT.md`: post-hoc dimension deletion lost joint/archive closure. EC2
  learns its context jointly and serializes the new weights.
- XI2 terminal custody: its initial 4,096 MiB cap OOMed at 4,562.547 MiB; the successful matched run
  peaked at 4,879.953 MiB. This replaced inherited-memory projection with the measured 6,144 MiB
  cap and 25.90% margin.
- The canonical context-waterfill law requires decoder-known same-object context and charges all
  video-derived parameters. That changed the plan from a possibly free training feature into a
  counted receiver channel. The integer-determinism law kept the extra path entirely inside the
  pinned integer/fixed-point HPAC inference surface.

## FIRE ORDER

**Disposition:** `QUEUED-WITH-A-FIRE-ORDER`  
**Owner:** MAIN Metal executor  
**Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_ec2/FULL_SCALE_RESULT.json`  
**Fire trigger:** local Metal lane remains free; `torch.backends.mps.is_available()` is true; source,
control, JS5, storage, and governed 6 GiB admission pins pass.

Exact command:

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/safe_run.py --rss-mb 6144 --projected-gib 6 --timeout 7200 --label ddm_ec2_sparse_event_hpac_n600 --status-receipt /Volumes/VertigoDataTier/pact/ddm_ec2/run/main.safe_run.json -- .venv/bin/python tools/run_ddm_ec2_sparse_event_hpac_conditioning.py --leg all --resume-from auto
```

## BOUNDARIES

- No Metal train/pack/Range/decode stage was fired by this arm; MAIN owns that stage.
- The 413 B coordinate price and 200-row custody are measured. Candidate model bytes, candidate
  token bytes, complete-container bytes, package admission, and live EC2 peak are pending MAIN.
- This arm is scorer-free. It cannot claim acceptance, score movement, or promotion.
- The JS5 store is 200 receiver-effective proposals from a stratified n32 frame sample and is
  explicitly not scorer-acceptance-tested. A negative EC2 result closes this precise sparse-mask
  HPAC formulation; it does not globally kill all learned local priors.

## POST-COMMIT VERIFICATION

The code/test/memo landing is serializer commit `d05b803022`; post-commit HEAD content matched all
three declared SHA-256 values. Ruff, Python compilation, the six focused tests, and the targeted
P0 measure-and-discard scanner (0 findings over the EC2 runner) passed after landing.

The full fast developer preflight was also run. It reported 17/25 green and 8 red. The complete
red-set diagnosis did not name either EC2 Python file or this memo: the findings were the existing
strict-load writer in `probe_outcomes_ledger.py`, authoritative-tag site in `submission_chain.py`,
25 legacy ad-hoc launch surfaces, missing terminal-claim prose in `AGENTS.md`, 124 external-memory
landing memos, eight pre-existing `lane_program_delta` registry references in EC1/JS6 surfaces, 56
legacy substrate scorer-contract violations, and 21 legacy trainer pose-default violations. These
unrelated baseline failures were not widened, waived, or modified by EC2.

## LIVE-HYPOTHESES

- The 234 event sites coincide with disproportionately surprising local HPAC tokens, so a 3x3
  learned local prior may save far more token bits than its 413 B coordinate charge and small model
  increment. This is plausible because EC1 selected receiver-effective topology events rather than
  uniform pixels, but it remains untested until the real n600 Range stream exists.
- Zero-start additive event conditioning can preserve CL1 initially and specialize only where the
  mask fires. This makes harm less likely than replacing the proven previous-partition prior, but
  training/QAT may still spend model bits without enough token return.
- Event type/source/target channels could increase bits saved per marked site if occupancy alone is
  under-informative. They are plausible because EC1 already classifies those fields, but they are a
  different counted formulation and must not be smuggled into this result.

## DEAD-ENDS

- Treating EC1 coordinates as free: closed by rule 118 and the same-object context-waterfill law;
  the coordinates are video-derived and not decode-derivable from the current package.
- Shipping EC1's full 44,410 B SP1 event object or a dense/global raster prior: closed for this use
  by the standalone-sidecar price and the measured global-temporal negative.
- Assuming a learned prior wins from lower conditional loss: closed by XI1's exact CAP1-beats-CPR1
  counterexample; only the complete receiver-consumed package can admit EC2.
- Inheriting XI2's original 4,096 MiB cap: closed by its measured 4,562.547 MiB OOM and successful
  4,879.953 MiB peak.
- Replacing or retraining the banked CL1 control: forbidden by charter and unnecessary; EC2 keeps
  the exact attested control read-only and tests one isolated added mechanism.

Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**;
EC2 did not move it.
