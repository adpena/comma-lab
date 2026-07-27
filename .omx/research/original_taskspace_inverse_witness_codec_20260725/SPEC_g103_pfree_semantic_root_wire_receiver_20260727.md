# G103 — P-free `SemanticRootY1V1` counted wire and receiver

Date: 2026-07-27  
Lane: `lane_g103_pfree_semantic_root_wire_receiver_20260727`  
Authority: research-only substrate engineering  
Pointer delta: none  
Candidate/archive/score claim: none

## Outcome

G103 lands the smallest real counted semantic-root packet and receiver section
that is broad enough to carry a fresh learned semantic prior without
reintroducing a raster payload home. It is subordinate to the canonical
G17/G21 selected-solution compiler:

- G17 remains the **single owner** of placement, logical ownership,
  pair-population identity, obligation coverage, lifecycle, proof dependencies,
  and evaluator authority.
- `SemanticRootY1V1` owns only a strict counted wire, canonical parser/serializer,
  deterministic receiver, and exact adapters into G17 logical values.
- There is no historical V15/P/PVSA/G85/G57/C1 payload or archive reuse.
- There is no source compiler, public `inflate.sh` dispatch, archive, full-n600
  scorer replay, score, or pointer mutation in this unit.

This corrects the initial too-small procedural grammar. Four analytic shapes
and sparse texture atoms are not presented as a complete codec. The top level
is a closed typed section union with counted learned model and temporal state.
New architecture, tensor, or entropy-codec IDs fail closed until their exact
receiver is implemented and reviewed.

## Settled negative that shapes the wire

FEED-ah is a hard prior, not a rerun target:

- The exact symbolic partition cost 255–332 KB and had direct `d_seg=0`, but a
  palette realization through R and the frozen scorer realized
  `d_seg≈0.005–0.008` (`≈0.0064`) with about 24% boundary flips.
- The palette frame carried no motion and realized `d_pose≈188`.
- The best measured task-space split composed to about `S=1.118`; Seg, Pose,
  and rate were each independently dominated.
- The cause is shared RGB physics: SegNet and PoseNet read the same realized
  RGB, so texture, chroma, parallax, and post-R boundary survival must be
  represented jointly.

Canonical source:
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md:805-829`.
The G103 packet therefore refuses a palette/direct-label-only interpretation.
Topology may factor the field, but cannot be the claimed solution.

## Closed typed section union

Every packet has exact section order, exact length, body CRC32, and exact EOF.
Unknown, missing, duplicated, reordered, trailing, or malformed sections fail
closed. Cryptographic section/tensor/custody hashes are external evidence, not
charged decoder operands.

| Tag | Typed content | Ownership intent |
|---|---|---|
| `PROF` | five role RGB prototypes and deterministic renderer constants | G17 realization gauge |
| `TOPO` | optional bounded analytic templates | G17 semantic topology |
| `EVNT` | optional chronological topology events | G17 semantic topology |
| `MODL` | mandatory counted quantized shared generator tensors | G17 learned residual |
| `LATN` | mandatory counted full-n600 temporally coded latent stream | G17 population sharing |
| `RGBF` | exclusive gauge mode plus optional procedural RGB basis | G17 realization gauge |
| `IRRQ` | sparse typed scorer-native RGB quotient atoms | G17 analytic/learned residual |

There is no generic `bytes`, raster, frame, plane, semantic-P, PVSA, V15, or
archive field. Quantized tensor bytes are admitted only under an
architecture-constrained role, dtype, rank, shape, scale, zero-point, and
length. Exact packet/model/tensor/latent hashes are frozen externally. The only
implemented learned ABI is:

`ORIGINAL_COORDINR_FILM_MLP_V1`

- `INPUT_WEIGHT/Bias`
- one to eight exact `HIDDEN_WEIGHT/Bias` pairs
- `FILM_WEIGHT/Bias`
- `OUTPUT_WEIGHT/Bias`
- int8 weights, big-endian int16 biases/state, int32 accumulation, Q12 state
- deterministic fixed-point coordinate features and tiled integer forward

The tensor role/shape sequence is exact. Raw scorer-plane shapes are rejected,
and bytes cannot trail the declared tensor sequence.

## Learned receiver is live, not metadata

The receiver consumes every counted learned component:

1. Decode the full `(600, modulation_dim)` temporal stream.
2. Select the requested Y1 row.
3. Build deterministic Q12 coordinate/topology features.
4. Execute the int8/int16/int32 input, hidden, FiLM, activation, and RGB-output
   graph in bounded tiles.
5. Compose the learned RGB residual with the baseline realization, optional
   explicitly owned procedural gauge, and typed irreducible RGB quotient.
6. Clip once to scorer-grid uint8 RGB.

The behavior test perturbs every one of the eight tensor roles and the selected
temporal latent row. Every perturbation must change output RGB. Dead counted
model or latent operands therefore do not pass the landed receiver test.

This original ABI is **not V9-compatible**. Settled V9 uses a
FiLM-conditioned `tanh(beta * sin(w*x))` phase-advection trunk with positional
Fourier state, 5-class `out_sdf`, `out_tex`, palette, and optional
`film_pl/concat_pl`. Exact V9 typed tensor roles and a deterministic reference
forward remain blocked by:

`V9_FILM_TANH_BETA_SIN_PHASE_ADVECTION_TYPED_ABI_AND_REFERENCE_FORWARD_OWED`.

No LVLS1/V9 source-compatibility claim is made.

## Temporal ownership and G94-V2 chronology

The default and preferred mode is
`DERIVED_BY_SHARED_GENERATOR`: the entropy-coded temporal latent is the sole
chronological actuator and the raw 600-row RGB gauge table is absent. This
avoids paying twice for phase/chroma/parallax.

The schema can represent an explicit post-generator RGB gauge table, but G17
binding refuses that mode until measured non-overlap and value-per-byte
arbitration exist:

`EXPLICIT_RGB_GAUGE_OVER_TEMPORAL_LATENT_VALUE_PER_BYTE_ARBITRATION_OWED`.

The temporal codec is deterministic signed int16 delta-Rice coding with:

- exact pair count 600;
- typed latent width and Rice parameter;
- decoded min/max;
- external exact decoded big-endian int16 SHA-256;
- exact EOF and zero-padding validation.

For a future physical V9 producer with `code[1200,*]`, the landed adapter counts
only `code[2*p+1]` (odd/Y1) as the 600-row semantic stream. Even/Y0 rows are
discarded and remain encoder-only. The test proves changing even rows does not
change the counted stream while changing an odd row does. G94-V2 is the sole
conditional Y0 owner after final Y1 freeze.

## G17/G21 sole-authority adapter

`semantic_root_g17_logical_values()` returns exact canonical values, not a
parallel lifecycle:

1. `G17SemanticTopologyV1`
2. `G17RealizationGaugeV1`
3. `G17LearnedResidualOwnershipV1` for the shared model
4. `G17PopulationSharingV1` for the n600 temporal stream
5. analytic or learned G17 residual ownership for the quotient

`bind_semantic_root_to_g17()` requires an exact canonical
`G17PairPopulationV1` identity map over `0..599`, exact canonical owner types,
and byte equality to each wire section. It creates no placement, obligation,
proof, or lifecycle object.

## Receiver closure

- Scorer Y1: deterministic `uint8[384,512,3]`.
- Population iterator: exact upstream order `0..599`, batch size bounded to
  1–16, streaming population SHA domain.
- V10: public generic factor-2 realization from scorer Y1 to
  `uint8[874,1164,3]`, with `certified_exact` verification.
- Later G94-V2: domain-separated final-Y1 binding over root-packet SHA, G17
  population SHA, and decoded scorer-Y1 population SHA.

## NO-FAKE and custody boundaries

- Model tensors and temporal latents are counted packet bytes.
- Generic integer decode/realization algorithms may live in public receiver
  code.
- External `SemanticRootSourceLineageV1` binds the exact root packet,
  source-video, target-custody, compiler-source, compile-config, originality
  declaration, exact model section, and decoded latent identities.
- The manifest is absent from candidate bytes and retained only as canonical
  `G17EncoderOnlyTeacherOracleEvidenceV1`.
- These hashes are a runner seam, not a claim that the source was reopened.
  Canonical G17 evidence must reopen and retain the real source/custody
  artifacts.
- The packet rejects exact known historical payload hashes and foreign payload
  magics at untyped homes. Typed model/latent bytes must satisfy their full
  declared semantics.
- Tests are behavior proofs only. No test result is a score or empirical
  full-n600 quality/rate result.

## Validation

Focused command:

```text
uv run pytest -q src/tac/witness_dsl/tests/test_taskspace_pfree_semantic_root_v1.py
```

Result at freeze: `12 passed`.

Static checks:

```text
uv run ruff check src/tac/witness_dsl/taskspace_pfree_semantic_root_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_pfree_semantic_root_v1.py
uv run python -m py_compile \
  src/tac/witness_dsl/taskspace_pfree_semantic_root_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_pfree_semantic_root_v1.py
```

Both passed.

## Remaining real compiler/product blockers

1. A fresh source-backed G17 compiler must fit the semantic topology (if used),
   shared model tensors, Y1-only temporal state, and irreducible quotient
   through the real R/scorer objective. No historical EP725/V9/V15 payload may
   be copied.
2. The physical V9 producer needs an exact dual-head phase-advection ABI and
   reference receiver before it can target this wire; current original MLP is
   not cross-cast as V9.
3. `inflate.sh` needs a reviewed public dispatcher for the exact packet and
   extracted-directory parse-back/double-decode closure.
4. The final decoded n600 scorer-Y1 population must be frozen and hashed.
5. G94-V2 must consume that exact frozen final-Y1 binding and freshly fit only
   conditional Y0.
6. One same receiver-closed archive must pass full-n600 batch-16
   `upstream/evaluate.py` on an authority lane. Until then there is no
   candidate, score, or pointer delta.

The frontier pointer is unchanged. This unit is means, not the end.

## Rate status

This is a correctness-first research ABI, not yet a rate-optimal candidate ABI.
G103 removes the obvious zero-value proof bytes from the counted packet:
lineage hashes, per-section SHA-256s, per-tensor SHA-256s, and the decoded
latent SHA-256 are external evidence. The packet retains decoder operands,
typed lengths/contracts, and one body CRC.

Before any rate claim, a distinct **candidate-minimized wire pass** must
whole-archive arbitrate:

- fixed-width versus entropy-coded section headers;
- tensor metadata amortization;
- topology/event and sparse-quotient coding;
- whether profile/procedural fields earn score value;
- outer ZIP interaction and duplicate integrity fields;
- exact decoder/runtime bytes and parse-back.

That pass must preserve the same typed semantics and external custody hashes
while proving exact decode equality. A byte count from this research ABI is not
a competitive rate claim.
