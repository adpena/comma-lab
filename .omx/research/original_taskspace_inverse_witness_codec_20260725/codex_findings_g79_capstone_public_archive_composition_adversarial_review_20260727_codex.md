# G79 adversarial review — capstone public archive composition

Date: 2026-07-27  
Lane: `lane_g79_capstone_public_archive_composition_adversarial_review_20260727`  
Mode: read-only L0 review; source was not edited  
HEAD observed during review: `11bd4ea44b1269d21aeca403afaffef8fb499ec1`

## Executive verdict

The smallest coherent product is **not** a relaxation of the G17 monolithic
archive and is **not** a complete P ZIP nested in its P section. It is an
additive flat public-product ABI:

1. store each of the five exact CarrierCompose P member payloads once at its
   canonical name;
2. store one counted, role-aware `TSPPV2` actuator member after them;
3. admit later typed actuator members in a closed, ordered state-transition
   sequence, beginning with a conditional `Y0|Y1` member after the TSPPV2/Y1
   state exists;
4. let public `inflate.py` reconstruct the exact canonical STORE P ZIP in
   memory from extracted member bytes plus generic fixed ZIP constants;
5. bind the reconstructed P byte count/SHA before opening G74; and
6. stream all 600 pairs through typed actuator transitions, double decode, raw
   output, two clean public inflations, and finally `upstream/evaluate.py`.

The in-flight G77 and `taskspace_selected_preimage_program_v2.py` already close
most of the local product/type seam: they add TSPPV2/G74RA1 without
reinterpreting G49's role-stripped TSPPV1 and locally flatten P members plus the
counted operand. They correctly state that public inflate remains open
(`taskspace_g77_flat_v15_selected_preimage_product_v1.py:16-18,228-267`;
`taskspace_selected_preimage_program_v2.py:2-19,52-73`).

Two material gaps remain under that local success:

- G77 currently reconstructs P by cloning `ZipInfo` from the still-packed
  product archive and accepts STORE members only
  (`taskspace_g77_flat_v15_selected_preimage_product_v1.py:81-119,122-165,
  228-249`). The public evaluator first runs `unzip` and passes only the
  extracted directory to `inflate.sh` (`upstream/evaluate.sh:41-47`), so that
  is not yet the public decoder path.
- G7 cannot price or admit this product. Its evaluator hard-calls the G17
  monolithic P/G/A/E builder, then the one-member `0.bin` outer codec
  (`taskspace_whole_archive_allocator.py:409-429,542-563`). That codec races
  only whole-member STORE against whole-member DEFLATE
  (`taskspace_outer_archive_codec.py:477-533`).

The exact pointer did not move. The competitive pointer observed during review
was the official-display `0.172` row. No archive was promoted and no score was
claimed.

## Stores and exact objects consulted

- byte-identical `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, the current canonical
  pointer, lane registry, and dispatch claims;
- G7, G17, G49, G65, G68, G69, G72, G74, G75, and G76 sources/specs/receipts;
- in-flight G77/TSPPV2 source;
- the retained fresh-current-lineage five-member n600 P:
  `.omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z/ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes`;
- public `upstream/evaluate.sh`, `upstream/evaluate.py`, README, G29/G55 public
  closure machinery, layered public runtime, monolithic receiver, and exact
  outer archive codec;
- prior project memory only as a recall index for P-once, pair streaming, and
  public-entrypoint closure; every load-bearing current claim above was
  reverified in the repository.

## Premise attacks and measured falsifiers

### 1. Public unzip does not prevent exact logical P reconstruction

The original CarrierCompose writer uses an ordered mapping and the same generic
metadata for each member: date `1980-01-01 00:00:00`, STORE,
`external_attr = 0100644 << 16`, and Unix `create_system = 3`
(`direct_description_entropy_priced_member.py:1945-1954`). The retained P has:

1. `manifest.json`
2. `predictor.zip`
3. `predict/movable_polygon_worldsheet.g1s`
4. `render/receiver_realization.ddrp`
5. `render/scorer_solved_templates.ddst`

A review-time falsifier used the actual public `/usr/bin/unzip` behavior into a
context-managed temporary directory, then read only those five extracted file
payloads in the fixed generic order. It created fresh `ZipInfo` objects from
decoder-owned constants; it did **not** reuse the original archive's
`ZipInfo`. Result:

```text
source_bytes       133941
source_sha256      759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df
rebuilt_bytes      133941
rebuilt_sha256     759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df
byte_identical     true
scratch_auto_cleaned true
```

Thus top-level flattening can preserve exact **logical P archive bytes** after
public extraction. It does not physically store the complete 133,941-byte P ZIP
string. The honest invariant is:

```text
each semantic member payload occurs physically once
and canonical_P(extracted semantic members) is byte-identical to exact P
```

Any field called `semantic_p_occurrences=1` must mean one reconstructed logical
object, not one physical occurrence of the full ZIP byte string.

### 2. Outer compression may differ while reconstructed P remains exact

The public outer methods affect only counted archive bytes. After unzip, the
member payloads are identical, so inflate reconstructs the original all-STORE P
regardless of each outer member's method.

An exhaustive `2^5` race over the actual five n600 semantic members, with fixed
metadata and level-9 DEFLATE, measured:

| top-level method assignment | exact bytes | delta from P |
|---|---:|---:|
| all STORE, exact P | 133,941 | 0 |
| all DEFLATE | 129,812 | -4,129 |
| exact minimum `11001` | **129,800** | **-4,141** |

`11001` means DEFLATE `manifest.json`, `predictor.zip`, and
`render/scorer_solved_templates.ddst`; STORE the G1 worldsheet and realization
profile. The 4,141-byte delta is `0.0027573219248789114` score units in the
contest rate term. G77's STORE-only v1 policy is therefore a material score
cost, not cosmetic cleanup. The exact race must include TSPPV2 and every later
actuator member as well.

### 3. Recursive flattening is rate-worse

Do not cargo-cult "flatten" recursively. A second exact review falsifier found:

| physical layout | minimum exact bytes |
|---|---:|
| stop at P's five top-level members | **129,800** |
| also flatten `predictor.zip` | 130,150 |
| also flatten nested `chart.zip` | 130,825 |

Flattening predictor adds 350 bytes; flattening predictor and chart adds 1,025
bytes. `predictor.zip` has twelve STORE entries including `chart.zip`, whose six
payloads are already near entropy-coded. Outer compression of the nested ZIP
beats repeated outer names and headers. The correct boundary is therefore:

```text
flatten the complete semantic P envelope once;
retain predictor.zip and chart.zip as opaque exact P member payloads.
```

This also corrects imprecise language about "no nested ZIP": the final product
does retain `predictor.zip` as one required P member. What it avoids is nesting
the **complete P archive** as a G17 P-section payload.

### 4. G17's refusal is a legacy invariant, not a ZIP limitation

G17 promises P once in one monolithic P/G/A/E member. The monolithic parser
deliberately rejects any section payload recognized as a ZIP. G68 therefore
correctly stopped on the complete CarrierCompose P
(`g68_g17_g49_selected_program_product_bridge_20260727/SPEC.md:27-42`).

The cure is an additive product version, not deleting the nested-ZIP guard.
Legacy G17/G69 byte values and parsers remain frozen. G77 is the appropriate
shape for that additive version.

### 5. G69 is not the current role-aware actuator

G69 is a strict, useful active-A parser for the frozen population-global
**TSPPV1** wire (`SPEC_g69_g49_active_a_g17_abi_20260727.md:7-18,20-35,
37-69`). G72 proved that TSPPV1 strips the donor role, executes in the wrong
coordinate/stage, and cannot honestly carry the V15 actuator
(`SPEC_g72_fresh_n600_g49_analytic_factor_compiler_20260726.md:8-24`).

G74 adds a distinct counted role-preserving operand and G76 derives an exact
base-preferred camera preimage while retaining the donor scorer tensor
(`SPEC_g74_v15_roleaware_overlay_decoder_20260727.md:7-27,132-142`;
`SPEC_g76_base_preferred_exact_numerator_overlay_20260727.md:41-57,93-108`).
The in-flight TSPPV2 correctly binds one population-global G74RA1 bank and
leaves TSPPV1 untouched.

Therefore:

- G69 remains legacy compatibility evidence.
- The public product must consume TSPPV2 directly or gain a new append-only
  G17 active-A adapter for TSPPV2.
- It must never label G69's TSPPV1 parser as proof that TSPPV2 was consumed.

### 6. G68 owner-specific incidence is structural, not execution evidence

G68's incidence relation names
`physical_group_id + logical_owner_id + receiver_consumer +
receiver_operation`, but expressly does not prove execution. G65 likewise
refused to fabricate local ownership inside a co-deflated monolithic ZIP
(`SPEC_g65_g17_archive_object_operand_linker_20260727.md:23-47,48-65`).

The flat product improves the ontology:

- each independently compressed outer member is a real physical coding group;
- the five semantic groups are consumed by canonical P reconstruction;
- TSPPV2 is consumed by its strict parser and G74 decoder;
- a later conditional packet is consumed only after the Y1 state it names;
- ZIP directory/EOCD overhead is a shared demux group.

An execution receipt must still prove those edges against the exact archive and
decoded state. A structural relation alone is insufficient.

### 7. G7 is exact but not product-generic or globally optimal

G7 correctly rebuilds full objects, double decodes, measures the exact selected
archive, and admits only a negative full score transition. It also says plainly
that its ordered greedy trace is not a global optimum
(`taskspace_whole_archive_allocator.py:704-723`).

For this product it needs an additive archive-strategy seam, not another
section-size estimate:

```text
ArchiveStrategy.build(exact logical state)
  -> finite exact archive candidates
ArchiveStrategy.parse(exact archive bytes)
  -> exact logical state + physical groups
ArchiveStrategy.receiver_request(...)
  -> exact product-bound request
```

The existing G7 measurement, pointer, nonlinear-score, repeat, and rollback
logic can remain. The monolithic strategy remains the legacy default. The new
flat strategy exhaustively races every actual member's STORE/DEFLATE choice
while the member count remains bounded (five P members + TSPPV2 + conditional
Y0|Y1 is `2^7 = 128` exact archives). If later member count makes exhaustive
search unavailable, the result must be labeled bounded/nonoptimal rather than
"cheapest."

Joint endpoint selection must then go to the complete-universe/G33 controller
when proposal interactions or ordering can reverse signs; G7's greedy prefix
must not be presented as a globally optimal codec.

## Smallest coherent public ABI

Call the physical product `FlatV15SelectedPreimagePublicProductV1`; G77 may
retain its current internal name. The exact final `archive.zip` directory is:

```text
manifest.json
predictor.zip
predict/movable_polygon_worldsheet.g1s
render/receiver_realization.ddrp
render/scorer_solved_templates.ddst
taskspace/selected_preimage_program.tsppv2
taskspace/actuators/0001-conditional_y0_given_y1.<typed-magic>  # when admitted
taskspace/actuators/0002-...                                    # future, ordered
```

There is no second outer manifest. P's exact `manifest.json` remains untouched.
The reserved `taskspace/` namespace is the generic demux boundary.

Every actuator packet must self-bind:

- exact type/magic/version;
- source window and population identity;
- predecessor state type and SHA;
- output state type;
- exact operand byte homes and lineage;
- named receiver operation;
- target/source/decoder custody identities that affect interpretation; and
- parse/re-encode identity.

The decoder sorts numeric actuator stages, refuses gaps/duplicates/unknown
types, and requires every predecessor identity before execution. This is the
minimum extensibility needed for the planned two-layer codec:

```text
exact P
  -> TSPPV2 role-aware native prepaint / selected Y1 state
  -> conditional Y0|Y1 enhancement packet
  -> joint pair state
  -> G76 or successor exact preimage realization
  -> generic V10 factor-2/output-camera realization
```

A video-specific selector, factor, learned residual, threshold, exception, or
state delta lives in a counted actuator packet. Generic ZIP reconstruction,
parsers, state machine, inverse/preimage solver, repair, realization, output
writer, checkpoint logic, and public closure code live in `inflate.py` /
`inflate.sh`.

## Exact state-transition DAG

```text
archive.zip exact bytes/SHA/size
  -> public unzip
  -> EXTRACTED_DIRECTORY_CUSTODIED
  -> exact member-set/name/safety/size checks
  -> outer member payload SHA + compression-method receipt
  -> CANONICAL_P_RECONSTRUCTED_FROM_FIXED_CONSTANTS
  -> require P bytes=133941 and SHA=759e... (or packet-bound fresh identity)
  -> strict CarrierCompose parse/re-encode + G74 open
  -> strict TSPPV2 parse/re-encode/source-target-decoder binding
  -> TSPPV2_Y1_STATE
  -> ordered conditional Y0|Y1 parse/bind/execute
  -> JOINT_SELECTED_PREIMAGE_PAIR_i
  -> exact camera/output realization for pair i
  -> pair-i receipt + atomic checkpoint
  -> decode pair i again and require byte/receipt identity
  -> repeat i=0..599 without a dense n600 resident tensor
  -> expected raw video files and names
  -> clean public inflate run 1
  -> clean public inflate run 2; require tree/output identity
  -> archive parse-back and recursive runtime/import/file manifest
  -> upstream/evaluate.py full n600 contest-CPU and/or contest-CUDA
  -> only then score/pointer admission
```

The public runtime should consume the extracted directory it is given. Reading
`../archive.zip` happens to work in the current shell layout but couples the
decoder to an unpassed parent path and does not exercise extracted-directory
closure. It is unnecessary because fixed-constant reconstruction is proven.

For cross-host byte identity, the decoder should prefer a tiny manual ZIP32
STORE constructor (fixed local headers, CRCs, central directory, EOCD) over
assuming all installed Python `zipfile` versions serialize cloned metadata
identically. It must then compare the result to the packet-bound P SHA before
use.

## Migration and implementation order

1. Land and review TSPPV2/G77 as the additive local role-aware product. Preserve
   G17/G69 unchanged.
2. Correct G77 terminology so physical member-payload uniqueness and logical P
   reconstruction are not conflated. Keep the required nested `predictor.zip`.
3. Add extracted-directory parse/reconstruct entrypoints using the fixed P
   member list and manual canonical STORE ZIP32 writer. Prove exact `759e...`
   reconstruction after the public `unzip` command.
4. Generalize the flat builder/parser to independently admit STORE/DEFLATE and
   exact mixed-method price selection. Exhaust all current profiles; retain the
   129,800-byte top-level P profile unless the complete product race finds a
   different whole-object minimum.
5. Add the ordered typed-actuator namespace/state machine. The next type is
   conditional `Y0|Y1`, and its predecessor must be the exact TSPPV2/Y1 state.
6. Emit an executable G68-style owner-specific incidence receipt for every
   semantic/actuator/shared-demux physical group and actual receiver operation.
7. Extract G7's exact evaluation core behind the additive archive-strategy
   interface. Keep monolithic G17 as the compatibility strategy; do not weaken
   its nested-ZIP refusal.
8. Feed only complete same-base endpoints to the global/G33 arbitration surface.
9. Compile real fresh n600 factor operands from the five immutable 120-pair
   stages. G72's fresh margins/base-score inputs and whole-score admission remain
   owed; bounded G74/G76 mechanics are not population score evidence.
10. Build the receiver-closed public archive; parse it back; stream and double
    decode all 600 pairs; run two clean public inflations; then run the exact
    upstream scorer.

## Required tests and falsifiers

1. Public-unzip reconstruction: no read of the original P ZIP or product
   `ZipInfo`; exact P bytes/SHA must still match.
2. Missing, extra, reordered logical, renamed, symlink, traversal, duplicate,
   empty, oversized, or mutated semantic/actuator files must fail closed.
3. Outer STORE/DEFLATE changes must preserve extracted payload and reconstructed
   P identity.
4. Exact exhaustive price receipt must enumerate all actual method profiles and
   bind the selected archive bytes/SHA, zlib version/profile, and deterministic
   tie-break.
5. Recursive-flatten proposals must lose to the measured top-level boundary or
   provide a new exact whole-product counterexample; no assumed recursive win.
6. The complete P ZIP byte string must not occur as a nested physical member;
   each of the five P member payloads must occur at exactly one declared home.
7. TSPPV1 role-colliding bytes must be refused by the TSPPV2/G74 path.
8. TSPPV2 must execute before conditional Y0|Y1; gaps, swaps, duplicated stage
   numbers, wrong predecessor SHA, and mixed incompatible actuator families
   must fail closed.
9. Delete/mutate each actuator and prove the named receiver state/output changes
   or deterministically refuses; structural incidence without liveness is not a
   pass.
10. Pair-streamed decode must equal the corresponding bounded reference without
    retaining dense n600 planes, and two full decodes must be byte-identical.
11. Resume after every 120-pair stage and after an injected interruption; final
    bytes must equal an uninterrupted run.
12. Archive scan must prove scorer weights, target/GT tables, historical selected
    planes, and video-derived decoder constants are absent from free code.
13. Public `inflate.sh` must produce every expected raw video under the exact
    names, size, dtype, and frame order on two clean roots.
14. Recursive public closure and full upstream evaluation must bind the exact
    final archive bytes/SHA. No private decode result can substitute.

## Honest blockers after this review

- G77/TSPPV2 is in flight and has not yet landed a public extracted-directory
  runtime, archive receipt, or public double-inflate proof.
- G77 v1 is STORE-only and leaves at least 4,141 measured semantic-member bytes
  on the table before pricing the actuator.
- G7 has no flat multi-member archive strategy and remains order-dependent.
- G68 owner-specific incidence has no exact flat-product execution/liveness
  receipt.
- The ordered conditional `Y0|Y1` packet/state transition after TSPPV2/Y1 is not
  yet a closed public ABI.
- G72 still lacks fresh matching margins/base scorer-state inputs and therefore
  has not compiled the five real 120-pair role-aware stages.
- G74/G76 prove bounded receiver mechanics only; cross-host Torch/fixed-camera
  portability, Pose/Seg networks, n600 score, and public output remain open.
- No receiver-closed exact archive, public double decode, full-n600 component
  distances, upstream score, or frontier pointer movement exists from this
  composition.

## Pointer-delta honesty

This review found a coherent product boundary and an exact 4,141-byte
whole-semantic rate win, but it did not produce or evaluate a candidate. The
canonical exact pointer is unchanged. The next goal-directed unit is the public
extracted-directory G77/TSPPV2 product with mixed-method exact pricing and an
ordered conditional `Y0|Y1` extension—not another monolithic G17 wrapper and
not recursive flattening.
