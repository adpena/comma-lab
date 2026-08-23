# DDM RJ1 renderer-joint move — three receiver-closed precompensation forms, no mechanism verdict

**Status:** `MECHANISM-INCOMPLETE-WITHHELD`

**verdict_scope:** `NO-VERDICT:DX2_PRECOMPENSATION_REPRESENTATION_RUNGS`

**axis:** `[macOS-CPU scorer-free exact byte/container + receiver parse-back]`

RJ1 did not produce a renderer-joint candidate. It produced three distinct, counted renderer
representations on the exact DX2 container and retained deterministic primary/repeat archives, but it
did not jointly optimize the moved renderers, re-solve compensation against those final objects, or
re-solve the carrier. The charter forbids this arm from touching the scorer lane or launching heavy
work, so realized `d_seg`, `d_pose`, B/H/W collateral, and joint delta-S are all **UNMEASURED**. The
three rows are precompensation build rungs, not candidates, and they do not close the renderer family.

## Exact retained rungs

The source is DX2 `archive.zip`, 180,368 B, SHA-256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`. Each row keeps four
renderer blocks and the inherited WD2S int4/fp16 deployment law. No row lowers quantizer depth.

The rate price is the charter-pinned `6.658590e-07 S/B` from
`.omx/research/ddm_tx1_toolbox_crosswalk_20260819.md` section 0; it is cited, not re-derived. “Rate-only
credit” is the maximum distortion debit that the byte cut could pay at that going price. It is not a
realized score improvement.

| rung | different representation | exact archive and repeat SHA-256 | bytes bought | rate-only credit at the going price | fixed-distortion 42,382 B demand | zero-distortion 150 B demand |
|---|---|---|---:|---:|---:|---:|
| `nested_group_dense_w72` | Salience-selected, nested 8-channel GroupNorm groups change width/topology from W96 to W72; all four blocks remain. | 169,489 B; `a731065431f1b134a5a2ceb51c969666e68def57bc0ca2c4a51dc7e2fb45d2f6` | 10,879 B | 0.007243880061 S | 25.668916% covered; 31,503 B remain | 10,729 B nominal surplus if distortion is exactly zero |
| `pointwise_svd_w96_r32` | Rank-32 SVD down/up factors replace every full 96x96 pointwise operator; original pointwise biases and per-block FiLM remain. | 175,177 B; `ea9303074b3083f45d12bd22dac56ef66ad17de6d316887d3224911977d756d1` | 5,191 B | 0.003456474069 S | 12.248124% covered; 37,191 B remain | 5,041 B nominal surplus if distortion is exactly zero |
| `film_amortized_flat_w96` | One trunk FiLM replaces four block-local FiLM maps while retaining W96 and all four full pointwise blocks. | 179,290 B; `34855e3c43e564d48adc492d919afa81662ebff847386d36bbf1a07304b26d21` | 1,078 B | 0.000717796002 S | 2.543533% covered; 41,304 B remain | 928 B nominal surplus if distortion is exactly zero |

For every row, the exact WD2S packet survived CK2+Brotli decode, the patched DX2 residual parser
returned the packet byte-identically, the student model strict-loaded, packet repack was byte-identical,
and primary/repeat ZIP hashes matched. This is receiver **parse-back**, not full frame inflation and not
evaluator evidence. The realized exchange ratio versus the going rate is therefore **UNMEASURED** for
all three rows; only the exact rate-side break-even budget above is known.

| rung | realized d_seg | realized d_pose | B | H | W | joint delta-S |
|---|---|---|---|---|---|---|
| `nested_group_dense_w72` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| `pointwise_svd_w96_r32` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| `film_amortized_flat_w96` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |

No interpolation is made between these rungs, and no parameter/token agreement is used as a SegNet
proxy.

## Compensation and carrier gate

There is **no compensation re-solve proof** for these precompile rows, so the charter's complete
renderer-joint mechanism is not present. The implementation asserts the missing proof rather than
faking it:

- `compensation.status = NOT_SOLVED`, with the required binding named as the final moved-renderer packet
  SHA, its exact rendered odd-frame field, and the DX2 archive SHA;
- `carrier_resolve.status = NOT_SOLVED`, ordered after final renderer optimization and the exact-object
  compensation solve;
- `candidate_admissible = false`; and
- the fire gate refuses admission until compensation is `SOLVED_EXACT_OBJECT`, the carrier is
  `PARSEBACK_EXACT`, primary/repeat archives match, and MAIN owns the scorer lane.

This is the explicit `TOY-BRACKET`/mechanism-incomplete outcome required when the compensator or carrier
re-solve is absent. QS5 proves that an in-compile exact-object compensation can cure stale-object pose
damage; it does not transfer a solved constant to any of these new renderer objects. Likewise, the
JG-line proves the need for carrier re-solving but supplies no carrier solution for these packets.

## Payload custody and boundary

Canonical retained root:
`/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1/`.

- `RESULT.json`: 20,334 B, SHA-256
  `405f89fb32a23fded3ded5b715989c2bf6efe7df6cedc79b98bbf89323fa26f0`.
- `CUSTODY_INVENTORY.json`: 73,543 B, SHA-256
  `dd3b89b7f9d68f11f3d828457316b748b796aa55fea36304d566dd5cd2f8467c`; 192 files,
  5,375,503 payload bytes, tree SHA-256
  `576c16b2159cd3262dfa18e2df7bd53b7f8ac80c9c8dc546ccdc7dd5cd17d88a`.
- `source/SOURCE_CUSTODY.json`: 838 B, SHA-256
  `22af08f39372ca3a78835c985b68b199a9762696c70544c0953af50929f82e6a`.
- Reproduction command: `.venv/bin/python experiments/ddm_rj1_renderer_joint_move.py`; minimum free
  space 1 GiB. Vertigo, the first-priority SSD tier, had 8,986,861,568 free bytes at preflight.

The first development run violated the payload-order rule: it materialized the dense WD2S packet, then
aborted on a comparison against the non-serialized `candidate_id` before persisting that packet. The
packet was deterministically reconstructed from the pinned DX2 semantic state, but reconstruction does
not erase the incident. The source was fixed to persist the float initialization and packet before
closure validation. A second interrupted run retained its generated payloads and partial runtime. The
superseded local receipt remains at
`.omx/tmp/arm_receipts_local/ddm_rj1_renderer_joint_move/precompile_r1/RESULT.json`, 21,191 B, SHA-256
`a4e55707bbf1e2cac86c2b2f665dca2682e449768ae2784a8d7b42be45e7c292`; nothing was deleted. The
canonical build was then rematerialized on Vertigo and its completed receipt was successfully verified
twice through the resumable path.

RJ1 launched no trainer, no full inflation, no scorer, no n600 job, no Modal job, and no paid action. It
did not modify `upstream/`. The live scorer lane remains MAIN-owned and is intentionally untouched while
JF1's seven reference fits are live.

## RECALL EVIDENCE

The corpus search was not limited to the charter seeds. Searches covered `.omx/research/` memos and arm
receipts for `renderer`, `semantic renderer`, `width distill`, `factorized`, `flattened`, `FiLM`,
`compensation`, `carrier re-solve`, `joint`, `DX2`, `SM3R`, and `WD2S`; the canonical-equation listing;
`CANONICAL_RESEARCH_INDEX*`; `sub015_DAG_*` FEED blocks; design/SPEC files; the task ledger; and the live
hot state. The exact DX2 receiver/runtime and retained semantic state were inspected rather than inferred
from a memo.

Charter-seed pins were verified: gestalt `7184ec1a...`, AR1B `388185a6...`, MST1 `a22b7825...`, QS5
`f67595d4...`, TX1 `4bf730e5...`, and AP1 `3f739cf3...`. The JG1/JG2/JG3/JG5 and SA2/SA3 joint-move
receipts were also read.

Beyond those seeds, WD2 supplied three already-implemented, legal renderer forms and a counted WD2S
receiver: dense, low-rank pointwise factorized, and single-FiLM flattened. WD3 showed only
instance-scoped fresh-student boundaries. WD4 then changed the plan materially: its warm-lineage W64
four-block dense slice was already scored at `d_seg=0.03182023`, `d_pose=13.43292999`, and recomputed
`S≈14.8829`; it is an instance negative, not a family result. RJ1 therefore did not rerun W64 or infer a
width law. It used the existing forms only as scorer-free representation rungs, raised the nested rung to
W72, pinned the later DX2 object rather than WD4's FX5 base, and withheld every distortion claim pending
the complete joint mechanism. HR1 reinforced the retained-runtime and receiver-closure shape; it did not
supply a current-DX2 renderer solution.

## VERDICT

`MECHANISM-INCOMPLETE-WITHHELD`, with
`verdict_scope = NO-VERDICT:DX2_PRECOMPENSATION_REPRESENTATION_RUNGS`.

No rung has realized joint delta-S below zero, because no rung has a realized joint delta-S at all. There
is therefore no dual-axis fire-order and no renderer-axis closure. RJ1 advanced a real, retained
representation ladder and receiver apparatus, but it did not achieve the charter's end state or move the
score.

Own-vehicle frontier: **DX2 — S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, archive
SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; RJ1 did not move it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — mechanism completion.** Owner: a MAIN-designated RJ1 successor. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/joint_r1/`. Fire trigger: JF1's live reference fits are terminal and harvested, the successor has a governed resumable/checkpointed renderer trainer, and storage preflight passes. Action: jointly optimize each retained representation on the exact DX2 object, solve compensation in-compile against each final packet/render pair, re-solve and re-encode the carrier, retain every primary/repeat archive, and fold any row that fails a required mechanism gate before scoring.
- **QUEUED-WITH-A-FIRE-ORDER — n600 joint measurement.** Owner: MAIN as exclusive scorer-lane and Modal-fire owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/scorer_n600/`. Fire trigger: at least one exact archive reports `SOLVED_EXACT_OBJECT` compensation, `PARSEBACK_EXACT` carrier closure, matching primary/repeat hashes, full receiver closure, and MAIN has claimed the unique idle n600 lane after JF1. Action: score only those exact archives in chunks no larger than 120, retain scorer outputs, report realized d_seg, d_pose, B/H/W, and recomputed joint delta-S separately, then seal a dual-axis fire-order only for a negative joint delta-S row.

## LIVE-HYPOTHESES

- The W72 nested dense rung may retain a useful fraction of its 10,879 B cut after joint optimization because it preserves all four receptive blocks and 75% of DX2's channel width; this remains plausible but is not supported by the catastrophic unoptimized W64 score.
- Rank-32 pointwise SVD may be the safest medium byte move because it preserves W96 embeddings, per-block FiLM, depthwise filters, and original pointwise biases while initializing from the exact DX2 operators; only receiver-realized joint training can show whether the discarded singular subspace carries hard Seg/Pose structure.
- Single-FiLM W96 may have the lowest distortion risk because it preserves full width and full pointwise spatial mixing while asking joint training to amortize temporal conditioning once; its 1,078 B cut is small but already exceeds the zero-distortion 150 B requirement.
- Exact-object compensation plus carrier re-solving may turn one otherwise losing renderer form into a joint win: QS5 demonstrated that re-solving can reverse stale-object pose harm, and every measured campaign pointer move used a coupled rate/distortion action. Neither precedent predicts which renderer form wins.

## DEAD-ENDS

- Renderer depth/precision coarsening alone is closed by AP1: all 12 tested rungs lost and the measured waterfill bought 0 B. RJ1 did not repackage that move.
- The exact WD4 warm-lineage salience-pruned dense W64 instance is closed: `d_seg=0.03182023`, `d_pose=13.43292999`, `S≈14.8829`; do not rerun it as a new renderer result.
- Carrying a compensation solved for another renderer object is closed by QS4's `+2.396e-4` pose damage; every future compensation must bind to the final moved object.
- Parameter, token, or field agreement cannot substitute for evaluator d_seg; the charter records a 349x understatement, so no successor should score these precompile rungs by agreement.
- The three current archives are not score candidates. Scoring them before joint optimization, exact-object compensation, and carrier re-solving would measure an explicitly incomplete mechanism and cannot close any representation family.
- Local disk is not the canonical custody route while the first-priority SSD has sufficient free bytes. The local receipt is retained only as superseded incident evidence; the canonical consumer must use the Vertigo tree.
