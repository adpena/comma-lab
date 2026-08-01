# G10 production PASS-G / optional-G8 / conditional-A composition (2026-07-26)

Status: implemented bounded receiver contract, research-only. This landing invokes no scorer or evaluator, produces no exact row, closes no through-R target debt, and moves no frontier pointer.

## Decisive causal evidence and ordering

The current ordering is anchored by the advisory n2 stage ablation:

- receipt: `ep725_n2_taskspace_stage_ablation_macos_cpu_advisory.json`;
- SHA-256: `2af9b50d70f342224aa438e95b4d53a05be3f253709c1fa1835da089f37e0f61`;
- P+exact semantic G: d_seg `0.0202229805290699`, d_pose `121.4552917480`;
- P only: d_seg `0.0033213298302`, d_pose `158.0920410`;
- P0+exact target Y1: d_seg `0`, d_pose `10.57589149`;
- exact target Y0+G: d_seg `0.02022298`, d_pose `15.52652359`;
- exact target Y0+predictor Y1: d_seg `0.00332133`, d_pose `57.28706360`;
- exact target both: approximately zero.

These are `[macOS-CPU advisory]` n2 values, never an exact-score claim. They establish only the causal composition constraint: Y1 realization is the first large lever, and A must condition on the Y1 that actually exists after the semantic stage and optional realization stage.

## Production path versus diagnostic control

The primary production object is:

`P -> nonempty PASS_SEMANTIC_G -> optional real G8 -> conditional A -> [Y0,Y1]`.

Exact semantic G is not mandatory production work. The already-built `TACG8S1 + TACA8P1` path remains a distinct diagnostic control for exact semantic G -> overlay -> G8 -> post-G8 A. It is never aliased to the production path.

The legacy `TACG1C + TACA3P1` receiver also remains unchanged. Crossed packet domains fail closed.

## Counted production G envelope

`taskspace_pass_semantic_g.py` defines two nested, nonempty domains:

- `TACGPS1` version 1: a source-bound semantic identity coordinate. It embeds the exact pair window plus predictor semantic, program, labels, and full predictor-surface binding hashes. It emits predictor labels and P1 byte-identically.
- `TACPG81` version 1: the counted G envelope. Its closed mode is either `PASS_NO_G8_V1` or `PASS_THEN_G8_V1`.

`PASS_NO_G8_V1` contains no repair bytes or repair hash and requires output Y1 to equal P1. `PASS_THEN_G8_V1` contains one exact `TACG8R1`, requires strict G8 receipt custody, and rejects a no-op repair whose resulting Y1 hash equals the pre-repair hash. Thus no empty legacy G and no fake no-op G8 can stand for the production semantic stage.

The optional G8 compiler uses the public frozen `SameClassRealizationRepairSurfaceV1` seam with exact counted P and pass-packet section identities. It does not modify the frozen G8 implementation.

Public API:

```python
compile_pass_semantic_g_envelope(
    *,
    predictor_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    repair_program: SameClassRealizationRepairProgramV1 | None = None,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1 | None = None,
    predictor_codec_id: str = "LVLS1.v1",
) -> CompiledPassSemanticGEnvelopeV1

decode_pass_semantic_g_envelope(
    envelope: bytes,
    *,
    predictor_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    predictor_codec_id: str = "LVLS1.v1",
) -> DecodedPassSemanticGEnvelopeV1
```

## Counted production conditional A

`taskspace_pass_conditional_a.py` defines `TACAPG1` version 1. It carries a canonical `PredictorPreservingA3ProgramV1` body with the existing three modes:

- `PASS_P0_V1`;
- `SPARSE_CONSTANT_RGB_V1`;
- `COPY_CORRECTED_Y1_SUPPORT_V1`.

The no-G8 control is therefore a real nonempty source-bound PASS-A packet, not a fabricated repair. The source binding includes:

- pair IDs and explicit PASS-G mode;
- exact predictor state, semantic, program, renderer, surface, upstream-receipt, labels, P0, and P1 hashes;
- exact PASS-G envelope and receipt hashes;
- inner pass packet and receipt hashes;
- semantic-label and pre-repair-Y1 hashes;
- optional repair packet and receipt hashes;
- exact resulting conditional-Y1 hash.

Its `binding_sha256` is SHA-256 over canonical finite ASCII JSON of that complete closed record. `PASS_NO_G8_V1` requires repair hashes `None` and pre-Y1 == resulting Y1. `PASS_THEN_G8_V1` requires both repair hashes and pre-Y1 != resulting Y1.

Decode reuses `apply_disjoint_camera_cell_reconstruction_v1` exactly. Copy mode reads the resulting conditional Y1; every mode emits chronological `[Y0,Y1]` and proves A leaves Y1 byte-identical.

Public API:

```python
build_pass_conditional_a_surface(
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    pass_g: DecodedPassSemanticGEnvelopeV1,
) -> PassConditionalASurfaceV1

derive_pass_conditional_a_source_binding(
    surface: PassConditionalASurfaceV1,
) -> PassConditionalASourceBindingV1

compile_pass_conditional_a(
    program: PredictorPreservingA3ProgramV1,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    pass_g: DecodedPassSemanticGEnvelopeV1,
) -> CompiledPassConditionalAV1

decode_pass_conditional_a_packet(
    packet: bytes,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    pass_g: DecodedPassSemanticGEnvelopeV1,
) -> DecodedPassConditionalAV1
```

## Monolithic receiver dispatch

`receive_ep725_taskspace_monolithic_pga_archive(...)` admits exactly:

- `TACG1C + TACA3P1` -> legacy `DecodedTaskspaceMonolithicPGAV1` and byte-identical legacy receipt contract;
- `TACG8S1 + TACA8P1` -> diagnostic-control `DecodedTaskspaceMonolithicPGAG8V1`;
- `TACPG81 + TACAPG1` -> primary `DecodedTaskspaceMonolithicPGAPassV1`.

The production branch strict-parses the outer archive and directory, reopens exact P, decodes the nonempty PASS-G envelope and optional G8, decodes A against the resulting Y1, checks directory-owned hashes and every foreign key, compares receipts and arrays across deterministic double replay, and strict-parses/re-emits `TaskspaceMonolithicPGAPassReceiverReceiptV1`.

The production receipt nests exact causal-P, PASS-G, optional-G8 (inside PASS-G), conditional-A, section-directory, archive, and chronological-frame custody. Its optional-G8 truth is derived from the envelope mode. It claims no scorer/evaluator/n600/through-R/score/candidate/originality/promotion closure.

## Cached causal-P seam

Allocator callbacks may call:

```python
receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
    archive: bytes,
    *,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
    expected_encoding: OuterArchiveEncoding | str | None = None,
    expected_archive_sha256: str | None = None,
    expected_member_sha256: str | None = None,
    max_member_bytes: int = ...,
) -> (
    DecodedTaskspaceMonolithicPGAV1
    | DecodedTaskspaceMonolithicPGAG8V1
    | DecodedTaskspaceMonolithicPGAPassV1
)
```

This is not a raw-array or P-bypass API. Every candidate is strict-parsed; its directory-owned P bytes/hash/length/state/receipt must equal the cached causal surface. Only then is P reused. G and A still double-decode per candidate.

## Required falsifiers now covered

- PASS packet/envelope/A truncation, trailing bytes, CRC mutation, mode smuggling, and strict re-encode mismatch;
- foreign same-pair P/pass surfaces and substituted archive-owned P;
- empty or fabricated legacy G presented as PASS semantic G;
- fake repair bytes/hash in no-G8 mode and no-op G8 in G8 mode;
- crossed legacy, exact-semantic diagnostic, and production G/A domains;
- both production `PASS_NO_G8_V1` and `PASS_THEN_G8_V1` through the monolithic cached-P receiver;
- all three A modes against both production G modes;
- exact Y1 invariance through A and chronological `[Y0,Y1]` order;
- truth, numeric-type, duplicate-key, nested-receipt, and noncanonical JSON smuggling;
- unchanged legacy and diagnostic receiver regressions.

## Remaining blocker

This seam makes production P/PASS-G/optional-G8/A a coherent counted receiver object. It does not select optimal G8/A controls, provide n600 evidence, close through-R realization, package a standalone contest runtime, or establish an exact score. The next lawful operation is allocator-side proposal composition through this exact cached-P API, followed by governed byte-closed authoritative evaluation of selected archives.
