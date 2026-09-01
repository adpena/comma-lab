# SFP1 — scorer-free SCMDL field-proposal preparation

**Verdict: `GENERATOR-READY` (`verdict_scope: FORMULATION`).** SFP1 produced three fresh,
region-parametrized dense-field proposals, a byte-identical null control, a retained positive
instrument control, a typed schema, and a consumption-ordered gate-2 handoff. This arm ran no
scorer, no RC64 encode/refit, no archive build, no Modal job, and no exact evaluation. Every candidate
rank is therefore labelled **`PROJECTION`**, not measured score evidence.

The own-vehicle frontier is unchanged: **AFR1 `S=0.14797617125559104 @ 180,002 B`, archive SHA-256
`cbb8d9283f435204800e31250bad7880490658012e2b6d8aa196ac4666bc84f5`, `[contest-CUDA T4 n600]`.**

## Result first

| object | rank | selector | changed sites / 117,964,800 | field SHA-256 | status |
|---|---:|---|---:|---|---|
| null empty proposal | — | none | **0** | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | byte-identical control |
| `sfp1_p01_atlas24_boundary1` | 1 | G3 top-24 pair cells ∩ token boundary distance ≤1 ∩ `X != A` | **1,084** | `75fe37daf8c3f615cd943a76697e9c6e8eabc56cb1c23d55a6b4251fc4553690` | `PROJECTION` |
| `sfp1_p02_atlas64_boundary1` | 2 | G3 top-64 pair cells ∩ token boundary distance ≤1 ∩ `X != A` | **2,831** | `656bd0c5c102109c3327eccd0c6e3a606aac44cbce7d9144396f8c171e24b76e` | `PROJECTION` |
| `sfp1_p03_mi1_patch12_boundary1` | 3 | top-12 MI1 `patch192` cells ∩ boundary distance ≤1 ∩ `X != A` | **9,723** | `fe6a9dd8ce770e308c7c3d1903ea1e40bee44938cc836188e486eefd408f527a` | `PROJECTION` |

Here `X` is the current byte-custodied 600×384×512 five-class field and `A` is the retained realized
CUDA-terminal Seg argmax of that same field lineage. Each proposal changes `X` itself. The proposal
contains no coordinate list, address stream, exception stream, or other separately coded side object.
Its optional `G` edit is an explicit cross-group causal-schedule refit over the observed class-transition
order; it is marked `refit_required=true` and `stored_side_stream=false`.

The materialized verification passed: the null SHA equals the base `X` SHA, the three candidate SHAs
are unique, p01's 1,084-site delta is exactly a prefix of p02's 2,831-site delta, and the assignments
agree at every prefix site. The candidate documents contain none of `token_gt`, `gt_argmax`,
`ground_truth`, `exception_stream`, or `address_stream`.

## Denominator and evidence boundary

- Law rows considered: **8/8**.
- Law rows generating an X or G family: **4/8**.
- Law rows folded or control-only: **4/8**.
- Ranked materialized proposals: **3/3**.
- Controls: **2/2**.
- Atlas coverage: **600/600 pairs**.
- Dense-field denominator: **117,964,800 sites** per object.

Measured by SFP1: source bytes and hashes, field geometry, materialized candidate bytes and hashes,
changed-site counts, class-transition counts, null identity, candidate uniqueness, and the p01⊂p02
prefix relation. Reused measured evidence is labelled by its source receipt. Not measured by SFP1:
physical RC64 bytes for any new proposal, refit behavior, `d_seg`, `d_pose`, archive size, exact score,
or contest promotion. The pointer did not move.

## Recall evidence

Before constructing the families, I searched the full research corpus and live state for `SCMDL`,
`joint field`, `field model`, `RC64`, `causal schedule`, `G3`, `MST1`, `MSR1`, `BHW2`, `WJ1`, `MI1`,
`RR9`, `FCD1`, and `WWC1`. I checked `.omx/state/canonical_equations_registry.jsonl`,
`.omx/research/canonical_research_index_rate_20260629.md`,
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, the live
`.omx/state/main_hot_state.md`, the task ledger entries surfaced there, and the actual source/receipt
files named below. I did not find an existing scorer-free gate-2 generator or a current-vehicle,
whole-field, scorer-realized SCMDL proposal in that searched scope.

The governing objective came from
`.omx/research/ddm_jc1_afr_rc64_joint_redesign_20260901.md` SHA-256
`fb035f4db92c78fba3357285b707995f5d0265b2ed4a38187c951a0b5fcbe05a`:

```text
L(X,G,M) = bytes_RC64(X | G,M) + bytes_model(G,M)
```

The SFP1 addition is the missing proposal half: a typed, executable change to `X`, optionally paired
with a first-class `G` refit, while leaving physical bytes and scorer acceptance to the gated consumers.

## Source custody

The complete machine-readable table is
`/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/SOURCE_CUSTODY.json`, SHA-256
`08f9a0856f02fb648e6a79cc141fea0dee7ba728705b8bcf77f94586356e2fc1`. Load-bearing pins are:

| source | path | bytes | SHA-256 | use |
|---|---|---:|---|---|
| current dense `X` | `/Volumes/VertigoDataTier/pact/ddm_xs1_cross_section_conditioning/measurement_v1/retained/input/dx2_tokens_decoded.u8` | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | coded base object |
| realized terminal `A` | `.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local/retained/inputs/cuda_terminal_argmax_n600.npy` | 117,964,928 | `e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34` | assignment target; copied byte-identically into durable SFP1 custody |
| G3 atlas | `/Volumes/VertigoDataTier/pact/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_n600.jsonl` | 9,121,001 | `faaff7299d86aa49c97e25e9cce2eeb0201f64e919f110015d31708788bcec09` | pair-rank selectors; old-vehicle advisory source |
| MSR1 boundary map | `/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/characterize_r1/token_boundary_distance.n600.npy` | 117,964,928 | `df0c931653964354b9bf7a4d13edd302517950f1dde6a619c9d7ba4c392a78cf` | receiver-field region geometry |
| MI1 cell table | `/Volumes/APDataStore/pact/ddm_mi1_indicator_model_axis/measurement_v1/RETAIN_cell_tables.json` | 95,366 | `070fe1024c3e920d22e24bf7475ef46bd0699b01ca6221530c089a5bf70c89aa` | position-cell ranking |
| RR9 verdict | `/Volumes/APDataStore/pact/ddm_rr9_reorder_refit/VERDICT.json` | 2,565 | `243c083973b037e39ab75b090a667d2d28a05eb20a207d269d1f959c9dcf2c82` | fold within-group reorder; keep cross-group refit distinct |
| BHW2/JF2 row | `/Volumes/APDataStore/pact/ddm_bhw2_jf2_oe1_argmax_screen/jf2/JF2_RESULT.json` | 13,909 | `11b69b1bec08b7bd3042a3a9da12d1fac31aada098d692c9975cbfc77e8368b5` | positive-control receipt only |
| WJ1 join | `.omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1/JOIN_RESULT.json` | 118,718 | `253511041b2b03209ee2dd138c4c1b753c4b95604784e730c01c11e321693ef0` | law evidence only; position masks folded |

The durable copy of `A` is
`/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/custody/cuda_terminal_argmax_n600.npy`, with the
same `e89e1ac0…` SHA. The generator verifies all fourteen source/control pins before touching an
output stage.

## Law-by-law fold

| prior law | disposition | why |
|---|---|---|
| G3 pair score-mass atlas | generate X family | legal realized scorer-cell rank; output rank remains projection |
| MST1/MSR1 stage/boundary split | generate X family | terminal argmax plus token-field boundary geometry; GT arrays are never loaded |
| MI1 position residual | generate X family | receiver-derived cell is joined to current realized scorer disagreement |
| RR9 reorder/refit | generate G edit | fixed within-group reorder was byte-neutral; cross-group refit is a distinct unmeasured cell |
| WJ1 cost×error positions | fold | retained positional masks are GT-conditioned |
| BHW2/JF2 benefit field | control only | known real byte outcome validates the instrument; GT-conditioned B and the JF field cannot seed proposals |
| FCD1 same-field diagonal | fold | reusing its field would reproduce a closed prior candidate cell |
| WWC1 token-GT cone | fold | token-GT proposal material is forbidden and its measured cone is closed |

This is the original-work boundary. SFP1 does not rename a retained JF/FCD field and does not hide a
position list in metadata. It generates a new full field from a realized scorer map plus generic region
predicates, and requires the consumer to code that changed field as the object.

## Controls and handoff

The null control is a fully persisted 117,964,800-byte field and is byte-identical to `X`.

The positive instrument control is deliberately not a candidate. It points to the retained BHW2/JF2
field SHA `da09731f140a0ddbd79520004a41cc4a77eab5efd85f3b3d012bf1e48756553a` with **8,301** edits and
its known physical RC64 stream of **108,108 B**, SHA
`ab7e327b46ed60da37b2de9f812ac8e68a230aaf5131771121c54d38365736bb`. Its `d_seg` and `d_pose`
remain unmeasured. Gate 2 must reproduce that byte receipt before trusting the instrument, then must
stop using the control field.

Consumption order is fail-closed:

1. Verify every source pin.
2. Decode/compare the null and require the base SHA.
3. Reproduce the positive control's 108,108-byte physical stream.
4. Only after RXC1 says `GATE-1-PASSED`, refit and byte-close p01, then p02, then p03.
5. Only a physically admitted proposal proceeds to frozen Seg/Pose measurement.
6. MAIN alone composes an admitted row and owns any exact contest fire.

The one queued action is: **when RXC1 publishes `GATE-1-PASSED` against the pinned SCMDL schema,
the MAIN-selected gate-2 scorer arm consumes the controls in order and then refits/byte-closes p01
before any scorer measurement.** Consumer store:
`/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/HANDOFF.json`.

## Artifacts and verification

| artifact | bytes | SHA-256 |
|---|---:|---|
| `HANDOFF.json` | 5,886 | `83b92462e8deec3686f6ed1b23f93302493f5ed9e8403da66641083f8f682e97` |
| `CANDIDATE_SET.json` | 6,430 | `00885e9a77d2779df9e68c3daf12869f95b91f28b704d7fd07a4cbd25e9a4787` |
| `CONTROLS.json` | 1,745 | `4cfa3eb73ae5570a6dc7a6ba55f5fe2de1ec0d8a7c5f7f49a755588b6c364eb5` |
| `VERIFY.json` | 355 | `802d26fc28c3fd412473c861df3149cf88a498f53793ad9238fdb20eb773c979` |
| `LAW_FOLD_TABLE.json` | 1,822 | `a5519a58f1b2ea4f0ca5ae1af20f26073c2bffe1eab0c42e0b09b9ad5428b47f` |
| generator | 32,975 | `8a50d46038b8321bcb4cacafa7b4d76cec50fbee599b0f110b8cf3e279411621` |
| tests | 4,264 | `b617bffbd02c844f344ef14967b61525175059f89b82bdf86b49d8274cce7e1b` |

The materializer is deterministic and restartable from
`/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/RUN_CHECKPOINT.json`. It writes each field
atomically, checkpoints after the null and after every candidate, retains every complete payload,
and blocks on a mismatched pre-existing output instead of overwriting it. A replay resumed from all
four verified field hashes and reproduced `GENERATOR-READY` without rewriting any field.

Validation: `.venv/bin/python -m pytest -q tests/test_ddm_sfp1_scmdl_field_proposal_prep.py` →
**7 passed**. The AST review tracker ingested and reviewed all 25 generator entities. The tracker does
not ingest `tests/`, so the test file was reviewed manually and exercised by the seven behavior tests.
The optional Ruff wrapper was unavailable because its installed binary lacks execute permission; this
did not weaken compilation, behavioral, custody, or materialized-payload verification.

## LIVE-HYPOTHESES

- p01 may be the best first physical row because the G3 top-24 selector concentrates the realized
  score-mass signal while changing only 1,084 boundary sites, limiting both refit disturbance and
  distortion spill. This is plausible but unpriced and unscored.
- The explicit cross-group transition schedule may realize the RR9 cell that within-group permutation
  could not test: changing X alters group membership, so a refit can change causal probabilities rather
  than merely reorder byte-neutral symbols.
- p03 may expose a rate/scorer composition because MI1's position cells were missing from the model and
  its assignments are restricted to realized terminal disagreements. Its 9,723-site size also makes it
  the highest-spill proposal, so it remains third.

## DEAD-ENDS

- Reusing any retained JF or FCD field is closed for SFP1: it reproduces a prior candidate cell rather
  than generating an original SCMDL field.
- WJ1 target masks and BHW2 B/H/W masks are closed as proposal sources because their membership rules
  consume GT. BHW2 remains useful only as a positive instrument control.
- WWC1 token-GT assignments are closed by the charter and by the realized broken-cone evidence.
- Fixed within-group RR9 permutation is closed as a byte lever on the measured object: it was exactly
  byte-neutral. Only the distinct cross-group refit cell remains live.

## ERRATA (MAIN harvest annotation, 2026-09-01 — appended before first commit)

The "own-vehicle frontier" line above transcribes the AFR1 archive SHA-256 as
`cbb8d9283f435204800e31250bad7880490658012e2b6d8aa196ac4666bc84f5` — that value is a
NON-AUTHORIZING TRANSCRIPTION ERROR (m153 genus, second instance after nx1's). The canonical
AFR1 archive SHA-256 is `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`
(180,002 B, pointer receipt). Every OTHER pin in this memo (base field `cc10a7b0…`, candidate
field SHAs, control SHAs) was authored from measured digests and is unaffected. Class note: two
arms have now mistyped this one sha in frontier lines — routed to the charter-lint/scaffold
surface (#1169) as a candidate check: any string matching the canonical sha's 8-hex prefix with
a divergent tail should be verified against the pointer at lint time.
