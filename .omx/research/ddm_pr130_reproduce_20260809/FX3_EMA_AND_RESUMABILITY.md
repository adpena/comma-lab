# FX3 — EMA and crash-resumable PR130 semantic QAT

**Date:** 2026-08-09  
**Verdict scope:** PR130 lifted PyTorch semantic-QAT mechanism; n4 updated-EMA parity/resume plus
n600 evaluation with seeded n32 deployed parity on CPU
**Authority:** `[CPU mechanism smoke; no score]`; `score_claim=false`  
**Machine-readable receipt:** `FX3_EMA_AND_RESUMABILITY_RECEIPT.json`

## 1. Measured EMA deployed-path argmax parity

**PASS in the declared n4 mechanism scope.** The selected `tac.training.EMA` shadow was run through
the lifted QAT exact forward and independently through the actual PR130 semantic pack/unpack receiver
path. The two paths emitted exactly equal frames and SegNet argmax labels:

| quantity | measured result |
|---|---:|
| pair selection | seeded random without replacement from 600 |
| seed / pair IDs | `20260809` / `[458, 460, 529, 585]` |
| pairs | 4/600 |
| SegNet argmax pixels | 786,432 |
| different argmax pixels | **0/786,432** |
| argmax delta rate | **0** |
| generated-frame equality | **exact** |
| maximum absolute frame delta | **0** |
| QAT/deployed non-finite values | **0 / 0** |
| deployed semantic blob | **40,252 B** |
| deployed semantic blob SHA-256 | `e20f729fa505149f22e0fa36ae7cbf13d9ffe71ab4a68cebe442ea3922ca4441` |

The diagnostic d_seg on these four real cached pairs was `0.0003751118977864583`; it is not an
n600 result and is not score authority. The parity gate itself compares the object that QAT evaluates
with the object the packer parses, and fails closed on any changed argmax pixel, unequal frame, or
non-finite value. The gate therefore admits the measured EMA shadow for this mechanism smoke. It does
not authorize a production score claim; an n600 production checkpoint must pass the same gate before
shipping.

A separately governed two-step run evaluated the selected shadow on **600/600 pairs** and then ran the
same deployed gate on a seeded **32/600 pairs**. It also passed: **0/6,291,456 argmax pixels differed**,
frames were exactly equal, maximum absolute frame delta was 0, and both paths had zero non-finite
values. Its selected full-population diagnostic was `d_seg=0.00029702080620659725`, and its 40,252-byte
semantic blob had SHA-256 `81058169865ffc7d1a400feba7dbe174d3610b5d55af78d13aa595062ecc1ea9`.
That run finished terminally with exit 0 in 879.456 s. Its best selection remained the step-0 EMA
shadow because the step-2 n600 diagnostic was worse; the n4 result above is therefore the load-bearing
check on an EMA shadow that actually incorporated optimizer updates, while the n32 result expands the
pair coverage of the pack/parse gate. Neither is a contest score.

The smoke consumed the real 600-pair cache (117,981,133 B, SHA-256
`8248a60da56119eb4b3ad76bfa32f5498dee849eaf4b83b304275064141fd828`) and the real stage-07
checkpoint (283,432 B, SHA-256
`1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf`). Only the evaluated and
trained pair population was reduced, as the charter permits.

### EMA implementation and authority

- The implementation uses the canonical `tac.training.EMA`, including its float-buffer guard,
  late-bound-module guard, and warmup behavior.
- `ema.update(model)` occurs exactly once after every `optimizer.step()` and fixed-zero projection,
  before `scheduler.step()`.
- Evaluation snapshots the live state, applies the shadow, evaluates, and restores the live state and
  training mode. The shadow never remains installed inside the training loop.
- The inference checkpoint's top-level `state_dict` is the selected EMA shadow. The live state remains
  separately present inside `training_state` for resumption.
- Decay resolves through LawRef `ema_decay_run_geometry_v1`, mode
  `decay_from_seed_fraction`, with target terminal seed fraction `0.01`. The two-update smoke resolved
  `d=0.1`; stage-08's 6,000-update geometry resolves `d=0.9992327661102197`. The `0.997` fallback is
  reachable only when update geometry is genuinely absent and was not used here.

## 2. Real kill/resume trajectory receipt

**PASS in the same seeded n4, two-step CPU mechanism scope.** Three governed runs used the same causal
configuration and exact input bytes:

| role | terminal receipt | child result | elapsed |
|---|---|---:|---:|
| uninterrupted | `uninterrupted_cpu_n4_v2/safe_run_status.json` | exit 0 | 26.308 s |
| interrupted after atomic step-1 checkpoint | `interrupted_cpu_n4_v2/safe_run_status.json` | **exit -9** | 9.877 s |
| resumed from preserved step-1 checkpoint | `resumed_cpu_n4_v2/safe_run_status.json` | exit 0 | 16.968 s |

The interruption was real: after
`semantic.periodic.step000001.full_state.pt` existed, the exact child PID from the governed pidfile
(`57443`) received `SIGKILL`; the wrapper returned 247 and wrote a terminal receipt with child exit
`-9`. The preserved checkpoint is 1,697,219 B, SHA-256
`23626ae4c4e832422924ec8b37fa95f029734e86825da41a3ca505d469996215`. The resumed invocation
loaded it with `--resume-from` and began at step 2.

The uninterrupted and resumed step-2 checkpoints then matched recursively:

- **15/15 causal training-state fields equal:** live model, EMA decay/warmup/update count/shadow,
  optimizer, scheduler, generator, order, cursor, Torch/NumPy RNG, history, best key, best d_seg,
  best RGB diagnostic, selected EMA state, step, and phase;
- **6/6 causal metadata fields equal:** schema, architecture, producing-stage config, parent identity,
  resume schema, and input-artifact manifest;
- final selected EMA `state_dict` equal;
- history, deployed-parity result, and d_seg diagnostic equal;
- independently packed semantic blobs byte-identical: 40,252 B, SHA-256 `e20f729f…`.

The outer `.pt` files have different SHA-256 values because their embedded result receipts retain
different output/resume paths and Torch serialization can encode container names. No raw-checkpoint
identity claim is made. The causal trajectory and shipped packed bytes—the charter's load-bearing
objects—are bit-identical.

Every periodic and stage checkpoint is written as a temporary sibling, fsynced, atomically renamed,
and retained under a distinct filename. The complete state includes all live/deployment weights,
optimizer/scheduler, EMA, RNGs, sample order/cursor, best selection, history, causal config, and input
byte hashes. Actual phase-boundary derivation is tested; the final phase is always preserved even for a
short single-phase run.

### Failure found and cured by the real resume

The first resume attempt refused because the resolved LawRef manifest retained a wall-clock
`resolved_at` field. Identical causal inputs therefore compared unequal. That was a real deterministic
resume defect missed by the first unit test. The serialized policy now strips both `resolved_at` and
`resolved_at_utc`; the test asserts both are absent, and the fresh kill/resume trajectory above passes.

## 3. Stale-config sweep — denominator 19/19

The v2 checkpoint schema removes the ambiguous top-level `config`. It stores three typed surfaces:
`architecture_config`, `producing_stage_config`, and immutable `parent_checkpoint`. A legacy checkpoint
can cross the compatibility boundary only as an architecture allow-list; schedule, precision, seed, and
paths are discarded. Any request to treat legacy metadata as producing-stage provenance refuses.
An unrecognized schema or a v2 object that also carries top-level `config` refuses.

The RR2 population remains the denominator: **19 direct reads**.

| consumer class | population | typed adapter | v2 fail-closed | stale propagators |
|---|---:|---:|---:|---:|
| immutable intake, read-only by charter | 7 | 0 | 7 | 3 |
| lifted source/pose paths | 7 | 1 | 6 | 2 |
| active mx1/wc3 paths | 5 | 5 | 0 | 0 |
| **total** | **19/19** | **6/19** | **13/19** | **5/19** |

All **5/5 stale propagators refuse a v2 checkpoint** because the stale alias is absent; they cannot
silently copy ancestor schedule facts from the new format. The launch-admissible semantic-QAT wrapper
is typed and emits v2. The pose resumable wrapper and all five active mx1/wc3 sites now use the typed
architecture adapter. The 13 unchanged direct readers are either immutable intake or reconstructive
lift custody; they intentionally fail closed on v2 rather than receiving undeclared body edits while
FX2's custody contract is in force. This closes the stale-provenance hazard for new checkpoints, but
also means a legacy packer entry point will deliberately refuse a v2 EMA checkpoint until it is migrated.

Schema tests cover: legacy architecture-only sanitization, refusal of legacy provenance, separated v2
metadata, refusal of an ambiguous alias, and refusal of an unknown schema. In the swept population,
**0/19 sites use the stale LR/steps/AMP/curriculum/path fields to control current execution**, matching
RR2; the cure prevents those fields from masquerading as new-checkpoint provenance.

## 4. Implementation surface and borrowed-substrate boundary

`src/tac/pr130_lift/train_semantic_quantized_resumable.py` is the launch-admissible wrapper. It imports
the lifted PR130 renderer, QAT forward, curriculum, exact-R evaluator, selection rule, and packing
primitives without rewriting their scientific mechanism. Its declared OURS scope is EMA, LawRef
resolution, deployed parity, atomic complete checkpoints, stage preservation, resumption, and the
seeded real-pair smoke reduction.

`src/tac/pr130_lift/checkpoint_schema.py` is the sole legacy compatibility boundary. The active mx1,
wc3, MLX configuration, and pose-resume consumers call it. The reconstructive lifted bodies remain
under FX2's exact-source-plus-declared-adaptation test rather than being silently edited.

Unit verification: **9/9 FX3 tests passed**. They cover LawRef/fallback behavior, EMA snapshot/restore,
phase-boundary preservation, recursive full-state continuation, deployed int4 frame/argmax identity,
and typed schema/refusal behavior.

## 5. Ranked residuals and falsifiers

1. **HIGH before a production ship — full 600-pair parity and contest-device adoption remain
   unmeasured.** The updated-EMA parity result is n4 CPU, with a supplemental selected-shadow n32 gate;
   these prove the mechanism and caught real defects but cannot establish rare-pair behavior or score
   movement. Falsifier: the final production EMA checkpoint passes the same deployed gate on 600/600
   pairs and its exact archive passes
   `upstream/evaluate.py` on contest CPU/CUDA.
2. **MEDIUM operational barrier — 13/19 historical direct readers fail closed rather than parse v2.**
   This is safe for provenance but blocks an old pack/eval CLI from consuming the new checkpoint.
   Falsifier: declared custody adaptations migrate those readers to the typed adapter, the FX2
   reconstructive test accounts for every adaptation, and a 19/19 census shows typed reads or explicit
   legacy-only refusal with pack/render identity.
3. **MEDIUM evidence gap — EMA quality versus PR130's historical sparse best-state selection is
   unmeasured.** Zero deploy-path delta does not show EMA improves d_seg. Falsifier: matched full-stage
   live-best versus EMA-shadow A/B on identical bytes/config, followed by exact parsed-public-wire
   evaluation.
4. **LOW telemetry gap — safe-run RSS samples reported 0 MiB.** The receipts are terminal and the
   state/byte checks are unaffected, but the sandbox did not expose process-group RSS to the sampler.
   Falsifier: repeat on a host where the governed receipt records a nonzero process-group peak.

## 6. Could not check / why

- **CUDA/contest authority:** `torch.cuda.is_available()` is false. No contest score or CUDA parity was
  measured.
- **MPS:** Torch 2.12.1 is MPS-built but reports `mps_available=false`; the governed MPS attempt exited
  1 after 5.262 s with the backend-unavailable error. It produced no result and is not evidence about
  EMA quality.
- **Production 6,000-step trajectory:** not launched. This arm's required mechanism proof is legal at
  reduced n/steps, while a production launch would be a separate score-directed event with its own
  checkpoint and exact-evaluation custody.
- **Exact score/frontier:** no archive was built or evaluated. The PR130 CPR1 base remains
  `S=0.172141297491896447` at 191,052 B `[contest-CUDA, DALI GT, n600]`; this arm did not move it.

## RECALL EVIDENCE

Before implementation, I searched the project research corpus, canonical research index/DAG, current
hot state, PR130 lifted/intake source, canonical equation registry, and tests for `PR130`, `RR2`,
`semantic QAT`, `EMA`, `ema_decay_run_geometry_v1`, `resume`, `stage checkpoint`, `config`, the two
checkpoint hashes, and the 19-reader/5-propagator census.

Material recalled facts changed the work:

- `RR2_SEMANTIC_LEG_AUDIT.md` supplied the settled 19-site denominator, the five propagators, exact
  stage behavior, and the packed-byte resume falsifier, preventing a new census denominator or a
  rewrite of the already-clean QAT core.
- `tac.training.EMA` and the prior EMA-lag findings required warmup plus eval-only snapshot/restore;
  they ruled out a flat `0.997` default and persistent shadow application.
- canonical equation sources established `ema_decay_run_geometry_v1` as executable authority and the
  0.01 target-seed-fraction convention.
- FX2's concurrent reconstructive-custody contract made raw lifted-body edits inadmissible unless
  declared; that caused the 13 legacy reads to fail closed on v2 while active consumers use the typed
  boundary.
- the actual PR130 packer/receiver and stage-07 checkpoint fixed the deployed object, allowing parity
  to compare real serialized int4 output rather than a proxy quantizer.

All bulky checkpoints and receipts remain under
`/Volumes/VertigoDataTier/pact/ddm_fx3_20260809`; none were deleted. Each governed receipt preserves
the argv, and each full-state checkpoint preserves causal config plus input paths, sizes, and SHA-256.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / PRODUCTION-VALIDATION** — owner: next PR130 semantic-QAT launch owner;
  consumer store: `.omx/state/main_hot_state.md` PR130 reproduction row; fire trigger: before any EMA
  checkpoint is packed or presented as a score candidate. Run the final checkpoint through the n600
  deployed argmax-parity gate, then exact parsed-public-wire contest evaluation.
- **QUEUED-WITH-A-FIRE-ORDER / TYPED-PACKER-MIGRATION** — owner: PR130 checkpoint-schema and packaging
  owner; consumer store: `.omx/research/ddm_pr130_reproduce_20260809/OFF_THE_SHELF_VS_PORTED.md`;
  fire trigger: before a v2 EMA checkpoint is passed to a legacy lifted pack/eval CLI. Add declared
  FX2 custody adaptations for typed architecture reads and prove old-stage load/pack/render identity.
- **QUEUED-WITH-A-FIRE-ORDER / EMA-EFFICACY-A-B** — owner: PR130 semantic optimization owner; consumer
  store: `.omx/research/ddm_pr130_reproduce_20260809/FX3_EMA_AND_RESUMABILITY_RECEIPT.json`; fire
  trigger: when a full stage-08-equivalent run is funded. Compare selected live-best and EMA-shadow
  packed bytes under identical training and exact evaluation; do not infer benefit from parity alone.

## LIVE-HYPOTHESES

- EMA can improve tail stability because PR130 historically observes only every 250th live state while
  the shadow integrates every optimizer update; this is plausible but needs the matched packed-byte
  A/B above.
- Full-n600 QAT-versus-deployed parity is likely to remain exact because the measured frames were equal,
  not merely the argmax labels, and both paths use the same fp16 scales and int4 codes. Rare pair
  embeddings or numerical edge cases can still falsify it, so n4 is not promoted.
- Migrating the legacy pack/eval readers to typed architecture metadata should be score-neutral because
  their current use is architecture-only and strict tensor loading fixes the shape. Exact old/new pack
  bytes can decide this without retraining.

## DEAD-ENDS

- A flat `0.997` EMA default is closed when run geometry exists: it violates the named LawRef authority
  and can recreate the settled short-run shadow-lag artifact.
- Leaving the EMA shadow installed during training is closed: it overwrites live learning state. Only
  snapshot/apply/evaluate/restore is admissible.
- Loop-end-only or one-path-overwritten saves are closed: neither survives the real step-1 `SIGKILL` nor
  preserves stage outputs.
- Treating a successful checkpoint load as resume proof is closed: the first real attempt exposed
  wall-clock LawRef metadata drift that load-only testing missed.
- The first resume-checkpoint format with `resolved_at` retained is closed: identical invocations
  refused as unequal; the field is now stripped and the fresh real kill/resume passes.
- Raw outer `.pt` SHA equality is closed as the trajectory criterion because noncausal receipt paths and
  container naming differ. Recursive causal state plus exact deployed-blob identity is the correct
  byte-closed test.
- This host's MPS route is closed for this arm because the installed Torch build reports MPS unavailable;
  the terminal failure says nothing about the EMA formulation.
