# DDM SD1 — PR130 semantic renderer rate–distortion findings

## Outcome

PR130's shipped q4 point is **optimal among the measured uniform q3/q4/q5
post-hoc quantizations**, but it is **not the semantic-leg optimum inside the
measured mixed-bit research formulation**. Four tensors at q3 and the other 12
at q4 produced a real 190,204-byte archive, 848 bytes below CPR1. On the full
n600 fixed-cache replay it incurred 166 extra Seg errors and still won the
charter's semantic-leg objective by `−0.00042392844867121244` S units.

This is `[macOS-CPU advisory]`, `score_claim=false`, and not promotable. The
non-int4 bytes were physically packed into the complete archive, parsed back,
and evaluated from the decoded tensors, but they use an isolated counted
research format that the public receiver does not yet parse. Pose was not
measured even though PoseNet consumes the changed semantic frame. Therefore no
full score, exact pointer movement, or shippable candidate is claimed.

## n600 measured curve

All rows use the retained official-Ada cache as both deployed renderer
conditioning and target. The denominator is `600 × 384 × 512 = 117,964,800`.
`ΔS_sem` is exactly `100·Δd_seg + 25·Δarchive_bytes/37,545,489` relative to the
matched q4 CPU row.

| decoded allocation | semantic blob (B) | archive (B) | Seg errors | d_seg | Δbytes | 100·Δd_seg | rate ΔS | ΔS_sem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform q4, legacy | 40,252 | 191,052 | 33,757 | 0.000286161634657118 | 0 | 0 | 0 | 0 |
| uniform q3 | 32,274 | 184,828 | 1,952,562 | 0.0165520731608073 | −6,224 | +1.62659115261502 | −0.00414430612423 | **+1.62244684649079** |
| uniform q5 | 48,258 | 202,324 | 155,228 | 0.00131588406032986 | +11,272 | +0.102972242567274 | +0.00750556211959 | **+0.110477804686867** |
| four q3 tensors, other 12 q4 | 39,090 | **190,204** | 33,923 | 0.000287568834092882 | **−848** | +0.000140719943576 | −0.000564648392248 | **−0.000423928448671** |

The selected allocation sets `frame_embed.weight`,
`blocks.1.film.weight`, `blocks.2.film.weight`, and
`blocks.3.film.weight` to q3. Every other rank≥2 tensor remains q4. Its exact
archive is
`/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/cpu_screen/archives/selected_mixed_n600.zip`,
SHA-256
`010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67`.

The curated machine-readable summary is `SD1_RESULTS.json`. The complete raw
41-record result is
`/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/cpu_screen/results.json`,
201,485 bytes, SHA-256
`3267ab65ee3659349d43552a64531790bf5ed515f4a4da6a1f6b9ae2dc8b417c`.

## Per-tensor allocation evidence

All 16 bit-packed tensors were measured individually at q3 and q5 on a seeded
stratified-random n120 screen: 12 random pairs without replacement from each of
10 contiguous 60-pair strata, seed `20260809`; pair-list SHA-256
`a72f6f54d7d6cd17f1b7c5a2f53d7a258ee9847b63d2fe078f310250f64e5987`.
No prefix population was used.

Only four of the 32 single-tensor moves had negative screen `ΔS_sem`, all q3:

| tensor moved q4→q3 | actual Δarchive bytes | Δd_seg, n120 | ΔS_sem, n120 |
|---|---:|---:|---:|
| `blocks.3.film.weight` | −184 | +2.96698676215e−7 | −0.0000928481797530 |
| `blocks.2.film.weight` | −160 | +2.11927625868e−7 | −0.0000853446699127 |
| `frame_embed.weight` | −332 | +1.73780653212e−6 | −0.0000472845192248 |
| `blocks.1.film.weight` | −168 | +7.62939453125e−7 | −0.0000355703588120 |

No q5 single won. The q3 screen losses ranged up to `+1.03602262933997`
for `coord_mix.weight`; the q5 losses ranged from `+0.000162354860649` to
`+0.0602221665825`. These are screening results, not tensor-family negatives.

The four favorable moves were replayed cumulatively; the full four-way screen
measured `ΔS_sem=−0.000306096688689`. A pairwise interaction canary for the
two strongest moves measured
`ΔS(A∪B)−ΔS(A)−ΔS(B)=−0.0000731437119028`, so the result was not obtained by
summing marginals. The exact selected four-way archive was then replayed on
n600, where it measured the `−0.000423928448671` result in the main table.

## Byte/decode closure

The q4 positive control exactly reproduced both the 40,252-byte semantic blob
(`9b98360b…b99`) and the 191,052-byte CPR1 archive (`0491d5df…c7cd`). For every
candidate the experiment rebuilt the complete one-member archive with the
original carrier, HPAC, and token bytes unchanged; extracted its semantic
section; consumed the entire section; decoded all 38 tensors; and required
exact tensor equality with the packer's dequantized state before scoring.
An independent post-harvest rebuild of the selected allocation reproduced the
190,204-byte archive and SHA-256 `010a8a52…fa67` byte-for-byte, then repeated
complete parse-back and tensor equality.

Non-int4 candidates carry a counted 14-byte `SD1M v1` header: magic, version,
the 16-entry count, and eight allocation nibbles. This makes the bit map
video/model-derived counted payload rather than free receiver code. Uniform q3
and q5 raw semantic sizes are consequently 32,274 and 48,258 bytes, not the
32,260/48,244 header-free arithmetic projections. Only the rebuilt archive
sizes in the table were used for `ΔS_sem`.

The governed run used batch 8 and six CPU threads, with one scorer process and
sequential n600 passes. It exited 0 after 3,186.604 seconds. Receipt:
`/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/cpu_screen/safe_run_status.json`,
SHA-256
`48400c3b27c4bc09bdbbb951cf0ea958373c9ab36427a51e96c15db201b4682e`.
The lane has a terminal completed claim.

## Verdict and boundaries

- **Uniform operating point — INSTANCE/FORMULATION negative:** uniform q3 and
  q5 lose decisively when physically packed and decoded from this fixed q4-QAT
  master. Do not re-run these post-hoc uniform cells.
- **Mixed operating point — INSTANCE positive:** the shipped q4 allocation is
  not the semantic-leg optimum inside this measured mixed-bit research format;
  the four-q3 allocation wins by `0.000423928448671` S units at n600.
- **Shippable operating point — BLOCKED:** q4 remains the only point accepted
  by the public receiver. The mixed result does not move a frontier.
- **Bit-depth family — OPEN:** all non-int4 cells re-quantize a q4-QAT master.
  They do not settle matched q3/q4/q5 QAT optima.
- **Capacity/width — UNMEASURED:** width 96 is encoded in shapes. No compatible
  width-80 checkpoint, measured capacity ladder, or resumable architecture
  morphism was found. Metadata mutation cannot produce a valid rung.
- **Full score — UNMEASURED:** PoseNet reads the changed semantic frame. Pose
  must be remeasured on exact decoded frames before combining this result with
  PR130's contest score.
- **Hardware — ADVISORY:** the q4 CPU control had 33,757 errors, 54 above the
  retained MPS receipt's 33,703. All deltas here use the matched CPU baseline.
- **Population — FIXED:** conditioning and target both use
  `/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt`,
  SHA-256
  `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`.
  AV/AV and fresh-T4-DALI/fresh-T4-DALI would change the deployed conditioning
  object and were not measured.

The raw run's shared axis display string accidentally included `n600` on n120
records; the explicit `population_kind` and `population_n` fields are correct.
The committed source removes that display token. The exact executed source is
recoverable by reverse-applying the verified unified diff in
`MEASUREMENT_SOURCE_DELTA.patch`; executed-source
SHA-256 is
`c114c30ab0a7e44668b38dded2277eac164d340b5e5441914dbe4c9e33e6357a`.
The post-harvest source also fail-closes future reuse on the pinned cache,
checkpoint, renderer, scorer code/weights, environment, screen design, and
archived candidate artifacts. Those review hardenings did not alter the raw
measurement or selection result; post-harvest source SHA-256 is
`85faba437a7f57ac49442b31701720cb785d20bc709e4c1dd5f92774f6941520`.

No contest evaluator, CUDA scorer, public-receiver candidate, pose replay,
bit-matched QAT, AV target, fresh T4-DALI target, capacity rung, or width rung
was run. The exact pointer did not move. Own-vehicle frontier remains
`S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`.

## RECALL EVIDENCE

Sources searched before design and adjudication:

- `.omx/research/`, arm receipts, `CANONICAL_RESEARCH_INDEX*`, and every
  `sub015_DAG_*` with content queries for `PR130`, `semantic quantization`,
  `quant_bits`, `mixed precision`, `per-tensor sensitivity`, `bit allocation`,
  `KKT`, `waterfill`, `width80`, `30748`, and checkpoint expansion;
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  quantization, byte-price, KKT, waterfill, and per-tensor allocation;
- design/SPEC documents, the task ledger, harness bridge, lane registry,
  active dispatch claims, codex arm queue, the pinned PR130 intake, and retained
  artifact paths;
- source queries for the packer, public receiver, evaluator, resumable trainer,
  exact renderer path, carrier training, and PoseNet input path.

Literal query forms included:

```text
rg -l -i 'PR130|CPR1|train_semantic_quantized|quant_bits|66,339|40,252|36,580' .omx/research
rg -n -i 'per[-_ ]tensor.*(sensitiv|bit|quant)|bit[-_ ]alloc|mixed[-_ ]precision|reverse[-_ ]waterfill|KKT' .omx/research docs
rg -n -i 'width80|30748|expand_semantic_checkpoint|semantic.*capacity' .omx/research src /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
rg -n 'ddm_rc1_receiver|ddm_cl1_capacity|ddm_sd1_semantic' .omx/state .omx/research
.venv/bin/python tools/list_canonical_equations.py --json
```

Findings beyond the charter's seeds changed the plan in five load-bearing ways:

1. The packer, public receiver, and governed resumable QAT wrapper are int4-only.
   Therefore non-int4 work used an isolated counted research format and made no
   public-receiver claim.
2. The #336 measured allocator failure predicted `ΔS=−2.790` from marginals but
   measured `+8.721` jointly. Therefore SD1 measured all singles, a pairwise
   cross-term, cumulative joint prefixes, and one selected n600 joint replay.
3. The deployed official-Ada token tensor differs from fresh T4-DALI at
   1,644/117,964,800 sites and from AV at 20,749 sites. Therefore the retained
   official cache stayed fixed as renderer conditioning and target.
4. PoseNet consumes both RGB frames and the carrier was retargeted to the q4
   semantic master. Therefore the reported objective is semantic-leg only and
   the full score is explicitly open.
5. No compatible width-80 checkpoint or PR130 capacity receipt was found, and
   RC1 owned the active public receiver surface. Therefore SD1 neither faked a
   width mutation nor raced an independent receiver edit.

Load-bearing anchors were the int4 refusals in
`src/tac/pr130_lift/train_semantic_quantized_resumable.py:667-709`, intake
`code/pack_semantic_pose.py:97-154`, and intake `code/inflate.py:171-215`;
the non-additive replay at
`.omx/research/witness_sensitivity_bitalloc_336_20260713.md:62-107`; retained
conditioning provenance at
`.omx/research/ddm_op1r_20260809/OP1R_PATH.md:117-139` for the 20,749-site AV
comparison and `CONDITIONING_COMPARISON.json` for the independently reproduced
1,644-site fresh-T4-DALI comparison; PoseNet's two-frame
input at `upstream/modules.py:61-84`; and historical depth expansion at intake
`code/expand_semantic_checkpoint.py:14-51`.

Scoped absence: no measured PR130-specific variable-bit public-receiver row,
matched q3/q5 QAT row, or semantic capacity ladder was found in those scopes.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN receiver/runtime successor;
  consumer store: `src/tac/pr130_runtime/fx1_runtime_tree/` and
  `.omx/state/probe_outcomes.jsonl`; fire trigger: MAIN elects to promote exact
  archive SHA `010a8a52…fa67` and a clean receiver lane is claimed; add a counted
  semantic allocation schema, prove legacy-q4 byte identity, and parse the
  selected mixed archive to identical tensors.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: full-candidate component evaluator;
  consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/pose_replay/`; fire
  trigger: the selected allocation becomes public-receiver-readable or MAIN
  explicitly accepts research-decoded advisory frames; measure matched n600
  q4-versus-selected `d_pose` before any full-ΔS claim.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: PR130 resumable-QAT successor; consumer
  store: `/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/bit_qat/`;
  fire trigger: public receiver plus pose replay leave a negative full ΔS;
  enable q3/q4 matched resumable continuations with preserved stage checkpoints
  before judging the bit-depth family.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: later semantic-capacity arm; consumer store:
  `.omx/research/ddm_sd1_semantic_20260809/CAPACITY_LADDER.md`; fire trigger: a
  compatible retained checkpoint or a receiver-aware, resumable identity
  depth/width morphism exists; only then measure a capacity curve.

## LIVE-HYPOTHESES

- The four Film/frame tensors may retain or enlarge their small win after
  bit-matched QAT because the q4 master already tolerates their q3 grids; this
  is plausible from the n600 joint replay but untested by matched training.
- Pose drift may be small enough for the 848-byte saving to remain favorable;
  the semantic perturbation changes only 166/117,964,800 Seg decisions, but
  PoseNet reads continuous pixels, so Seg sparsity is not proof.
- A lower-overhead public allocation schema may preserve most of the 848-byte
  archive saving because the current counted research header is only 14 bytes;
  decoder integration and whole-archive recompression still decide the result.
- Identity depth expansion may offer a valid capacity rung before width
  mutation because PR130 historically grew blocks by function-preserving
  residual initialization; no compatible resumable rung has been built.

## DEAD-ENDS

- Uniform post-hoc q3 on the shipped q4-QAT master is closed: actual n600
  `ΔS_sem=+1.62244684649079`.
- Uniform post-hoc q5 on the shipped q4-QAT master is closed: actual n600
  `ΔS_sem=+0.110477804686867`.
- Summing single-tensor deltas is closed: the measured two-tensor cross-term is
  nonzero, and the selected allocation was therefore replayed jointly.
- Treating analytical raw parameter bytes as archive bytes is closed: the real
  q3/q5 archive deltas were −6,224/+11,272 bytes, not the header-free raw
  ±7,992-byte arithmetic.
- Calling the isolated `SD1M` archive public-receiver-closed is closed: current
  public decode is int4-only.
- Treating AV/AV or fresh-T4-DALI/fresh-T4-DALI as the shipped candidate is
  closed: each substitutes a different conditioning tensor.
- Treating width as a checkpoint metadata knob is closed: the stage-08 tensor
  shapes cannot strict-load a different width.
- Assuming pose invariance from the semantic result is closed: PoseNet directly
  consumes the changed frame and no invariance receipt exists.
