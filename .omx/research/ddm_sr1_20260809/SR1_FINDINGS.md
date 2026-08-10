# DDM SR1 — counted semantic allocation receiver closure

SR1 is complete on its scorer-free scope. The counted `SD1M` parser was not
reimplemented: recall found that CX2 had already landed it in
`cf53216e3e856c15f849bcfe96a5dd4717da2d04`. SR1 closed the remaining receiver
gaps by making `SD1M v1` dispatch independent of payload length, pinning the
sixteen allocation names, proving both legacy forms against the shipped parser,
and retaining an actual 3,662,409,600-byte decoded-RAW comparison.

No scorer, evaluator, training, Modal, CUDA, or MPS job ran. `score_claim=false`.

## Result

| Obligation | Result | Evidence |
|---|---|---|
| Counted per-tensor schema | PASS | Reused the 14-byte `SD1M v1` parser from CX2; SR1 pins its sixteen-name order and routes every `SD1M v1` length to the fixed PR130 width-96 template. |
| No-record legacy q4 identity | PASS | 38/38 decoded tensors and carrier bytes exactly match the old FX1 parser. |
| All-q4-record identity | PASS | 38/38 decoded tensors and carrier bytes exactly match FX1; the full retained RAW is byte-identical too. |
| Selected mixed parity | PASS | 38/38 tensors match the independent SD1 research parser after exact public outer-receiver dispatch. |
| Real archive pricing | PASS | The selected archive is 190,204 B versus 191,052 B, so the honest already-counted net is −848 B. |
| Pose | NOT MEASURED | Preserved as the paired q4-versus-selected n600 fire-order held by the scorer owner. |

The implementation surface is
`src/tac/pr130_runtime/dv1_cpu_runtime/inflate.py`. DV1 is the receiver tree
that CX2 made evaluator-runnable and receiver-closed; the older shared FX1 tree
remains the untouched reference parser. This is a receiver capability, not a
new vehicle or score.

## Schema

`SD1M v1` is counted inside the semantic section:

- four bytes: `SD1M`;
- one version byte: `1`;
- one tensor-count byte: `16`;
- eight allocation bytes: two low-nibble-first bit depths per byte.

Every depth must be in `[2, 8]`. Absence of the magic selects the exact legacy
q4 path. The version binds this fixed allocation order:

1. `token_embed.weight`
2. `frame_embed.weight`
3. `coord_mix.weight`
4. `blocks.0.dw.weight`
5. `blocks.0.pw.weight`
6. `blocks.0.film.weight`
7. `blocks.1.dw.weight`
8. `blocks.1.pw.weight`
9. `blocks.1.film.weight`
10. `blocks.2.dw.weight`
11. `blocks.2.pw.weight`
12. `blocks.2.film.weight`
13. `blocks.3.dw.weight`
14. `blocks.3.pw.weight`
15. `blocks.3.film.weight`
16. `head.weight`

The prior full loader selected width only from three previously observed byte
lengths. That admitted the selected 39,090-byte mixed payload but rejected a
valid 40,266-byte all-q4 record and every other valid unseen allocation length.
SR1 instead recognizes `SD1M` first and selects the version's fixed width-96
template; the byte-length table remains unchanged for no-magic legacy payloads.

## Byte identity

The exact 191,052-byte base archive and the rebuilt all-q4-record archive were
routed through `split_payload` and `decode_models`; the exact `models_raw`
returned by that outer receiver was then fed to `unpack_semantic_pose`.
The old FX1 no-record parser, the extended no-record parser, and the extended
all-q4 parser produced identical canonical state bytes for all 38 tensors.
Carrier basis and coefficient bytes also matched.

For the strict whole-output proof, the finish-proven retained n600 token tensor
was rendered on CPU through two independent paths:

| Path | RAW bytes | SHA-256 |
|---|---:|---|
| FX1 legacy no-record | 3,662,409,600 | `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353` |
| DV1 all-q4 `SD1M` record | 3,662,409,600 | `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353` |

A streaming comparison covered all 3,662,409,600 bytes and found zero
differences. Both RAW files, their stage receipts, the exact token checkpoint,
the extracted archive sections, and every decoded state byte stream remain
under `/Volumes/VertigoDataTier/pact/ddm_sr1_20260809/retained/`.

The render reused retained token bytes rather than re-running the long token
decoder. That reuse is closed by a completed n600 Range receipt whose exact
output SHA is `c5c7671d…32ece`, plus byte identity of both decode inputs across
the base and all-q4 archives: HPAC SHA `b07fff73…0b58` and Range token payload
SHA `94837987…15eb`. The public CLI `main` function itself was not invoked;
this is an exact compositional receiver proof using its actual outer parser,
inner loader, completed Range decode output, and renderer functions.

The selected archive also passed public outer dispatch and exact comparison of
all 38 tensors against the independent SD1 parser. Its selected q3 set remains
`frame_embed.weight` and `blocks.{1,2,3}.film.weight`; the other twelve allocated
tensors are q4.

## Real archive pricing

| Object | Archive bytes | Delta versus base |
|---|---:|---:|
| PR130 no-record q4 | 191,052 | 0 |
| All-q4 `SD1M` record | 191,044 | −8 |
| Selected mixed without its header, invalid counterfactual | 190,204 | −848 |
| Selected mixed with counted 14-byte header | 190,204 | −848 |

The 14 raw header bytes have a measured zero-byte marginal in the selected
complete ZIP. On the all-q4 object they make the complete ZIP eight bytes
smaller. Compression is nonlinear, so neither result can be replaced by raw
parameter arithmetic. The selected archive already contains the header;
subtracting fourteen again would double-count. Its byte-only rate change is
`25 * -848 / 37,545,489 = -0.0005646483922476013 S`. SR1 did not rerun either
distortion component and makes no new score claim.

## Custody and reproducibility

The full machine receipt is
`/Volumes/VertigoDataTier/pact/ddm_sr1_20260809/SR1_RECEIVER_PROOF.json`,
91,619 B, SHA-256
`7b956662c772a32b80637b11b0cd0162b66850d0cd35b09e3870ab0d3a3e6f58`.
It records 54 retained payloads and source hashes. The retained tree is about
7.0 GB. The intended extended runtime was isolated from concurrent unrelated
TM1 working-tree edits before the final proof; its `inflate.py` SHA-256 is
`9a42628e6306ddaa4682c915db31196ffdace8fa502c6322a4586e3c4a7562a2`
and its receiver SHA-256 is
`6239649cc81e9c5a86273502be0beff19805720854b980f167bb71a0a80c3a42`.
That exact runtime delta is landed at
`58f62cd22ff07562c0534c999d705fb9edfe5279`; unrelated TM1 working-tree
changes remain outside SR1's commit and claims.

The proof command is restartable by stage:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/ddm_sr1_semantic_alloc_schema.py \
  --extended-runtime /Volumes/VertigoDataTier/pact/ddm_sr1_20260809/runtime_under_test/src/tac/pr130_runtime/dv1_cpu_runtime \
  --out-dir /Volumes/VertigoDataTier/pact/ddm_sr1_20260809 \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_sr1_20260809/resume \
  --render-byte-proof
```

Its completed-stage replay returned `complete=true` without rerendering. The
focused suite passed `8/8`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider \
  src/tac/tests/test_ddm_sr1_semantic_alloc_schema.py \
  src/tac/tests/test_ddm_cx2_compose_end_to_end.py::test_legacy_q4_semantic_loader_is_tensor_identical \
  src/tac/tests/test_ddm_cx2_compose_end_to_end.py::test_real_sd1m_state_matches_the_research_parser_and_full_loader \
  src/tac/tests/test_ddm_cx2_compose_end_to_end.py::test_mixed_semantic_headers_fail_closed -q
```

## Original-work accounting

- Borrowed PR130 vehicle and legacy parser: reproduced archive SHA
  `0491d5df…c7cd`; SR1 claims no originality for them.
- Borrowed SD1 measured allocation and real archive: commit `600af8ef7d` and
  archive SHA `010a8a52…fa67`.
- Reused counted parser: CX2 commit
  `cf53216e3e856c15f849bcfe96a5dd4717da2d04`.
- SR1's original delta is the general magic-based full-loader dispatch, the
  explicit v1 order guard, the retained all-q4 positive control, exact public
  outer-dispatch receipts, strict RAW identity, and real schema repricing. The
  runtime delta is commit `58f62cd22ff07562c0534c999d705fb9edfe5279`.

## RECALL EVIDENCE

The recall searched beyond the charter seeds before any design decision:

```text
.venv/bin/python tools/list_canonical_equations.py --json

.venv/bin/python tools/list_canonical_equations.py --json |
  jq -r '.[] | [.equation_id,.name,.one_line_summary,(.provenance.source_path // "")] | @tsv' |
  rg -i 'semantic|quantiz|per.tensor|allocation|receiver|parse.back|pr130|pose|schema'

rg -n -i --max-columns 500 \
  'PR130|semantic quant|quant_bits|per[-_ ]tensor|bit[-_ ]alloc|mixed[-_ ]precision|receiver|parse[-_ ]back|backward.{0,20}compat|legacy q4|SD1M' \
  .omx/research/CANONICAL_RESEARCH_INDEX*.md .omx/research/sub015_DAG_*

rg -n -i \
  '600af8ef7d|010a8a52|0491d5df|SD1M|selected_mixed_n600|semantic.*alloc|mixed.*tensor|pose.*fire' \
  .omx/research .omx/state

git log --all --date=iso --format='%h %H %ad %s' \
  --grep='PR130\|semantic\|allocation\|mixed.q\|receiver' -i
```

Beyond the seeds, recall found that the SR1 charter landed at `e82e3a6e5b`
at 19:05 UTC and CX2 landed the requested parser six minutes later at
`cf53216e3e` at 19:11 UTC. It also found the later AI1 resumability edit
`46c7b85219`, the missing all-q4 positive control, the selected test's manual
outer extraction, and drift in `runtime-dependencies.json`. Those findings
changed the plan from “build another parser” to “fold the existing parser,
generalize only the full-loader dispatch, pin the order, and close the two real
proof gaps.”

The equation registry had no exact `SD1M`, `010a8a52`, or 190,204-byte match.
Historical `heterogeneous_per_tensor_bit_allocation_compounding_v1` was
predicted old-HNeRV evidence only, while
`pr95_family_l40_fixed_28_tensor_schema_list_v1` supported fixed indexed
schemas but not this wire. FEED #336 independently confirmed that separable
allocation marginals can reverse under joint replay. Existing variable-level
and mixed-precision codecs supplied fail-closed dispatch precedent but not a
PR130-compatible format.

Queue row 110 and probe row 680 still described receiver integration as open;
CX2 queue row 115 already superseded their useful scorer follow-on with the
paired q4/SD1M control. SR1 therefore folds the receiver action as complete and
preserves only that paired scorer fire-order.

## Boundaries

- Measured here: exact archive bytes, outer receiver parse-back, 38/38 tensor
  bytes, carrier bytes, full decoded RAW identity, and deterministic replay.
- The CLI `main` function was not invoked; exact closure is compositional and
  binds its actual outer parser output, inner loader input, unchanged HPAC and
  Range payload, a completed Range decode receipt, and the renderer output.
- Not measured here: `d_seg`, `d_pose`, full `S`, contest-CPU, contest-CUDA,
  Linux closure, or interaction with other rate levers.
- No source under `upstream/` or the intake clone changed.
- The selected archive is receiver-readable but not promoted. Pose remains the
  unresolved component gate.
- PR130 remains unchanged at `S = 0.172141297491896447 @ 191,052 B`
  `[contest-CUDA, DALI GT, n600]`; SR1 moved no exact pointer.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner: MAIN scorer owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/evaluation/q4_control/`; fire trigger: the n600 scorer lane is free and both exact archives can run through the identical pinned real evaluator path.** Measure paired uniform-q4 versus selected-SD1M n600 `d_pose`; only if the full component delta stays negative may a successor consider promotion or bit-specific QAT.

## LIVE-HYPOTHESES

- The selected four-q3 allocation may preserve its byte win after paired pose replay because only the semantic model changed and its measured segmentation delta was small, but PoseNet consumes those changed frames directly, so this remains genuinely untested.
- Bit-specific QAT may recover or improve the small selected segmentation debt while retaining most of the 848-byte win, but it is worth firing only after pose confirms that this allocation is favorable on the complete component objective.

## DEAD-ENDS

- Reimplementing `SD1M` is closed: CX2 already landed the counted parser and selected 38/38 proof.
- Dispatching only the exact 40,266-byte all-q4 length is closed: a per-tensor schema has many valid lengths, so magic-plus-version must select the template.
- Subtracting fourteen bytes from the measured −848-byte result is closed: the selected archive already contains the header, whose measured final-ZIP marginal is zero on that object.
- Treating decoded tensor equality alone as the legacy proof is closed: SR1 rendered and compared all 3,662,409,600 output bytes.
- Summing per-tensor distortion marginals is closed by measured interaction; joint replay remains mandatory.
- Pricing from raw parameter counts is closed; only rebuilt complete archives are admissible.
- Assuming pose invariance is closed; no promotion occurs before the paired n600 pose measurement.
