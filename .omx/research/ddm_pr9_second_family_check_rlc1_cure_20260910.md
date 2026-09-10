# ddm_pr9 second-family compliance check of the RLC1 cured receiver

Date: 2026-09-10  
Axis: `[review; exact bytes; macOS-CPU scorer-free/advisory evidence]`  
`score_claim=false`  
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdict

**CLEAR-WITH-CONDITIONS.** The RLC1 mechanism clears pr8's rule-118 acceptance test: I did not find a video-selected numeric literal in the changed decoder code. The selected semantic class, row band, all fourteen remaining geometry settings, and forty fitted mixer weights are read from the counted archive rider. The geometry branch actually selected by the public receiver is integer Q16 C, and the retained two-configuration public-entrypoint proof is full-n600 and byte-identical.

The candidate directory is **not fire-cleared in its present state** for two independent conditions:

1. `candidate_runtime/MANIFEST.sha256` is stale. Its lines 10, 11, and 41 contain the move40 hashes for the three edited files, and it has no entries for the three added RLC1 modules. `shasum -a 256 -c MANIFEST.sha256` therefore fails on `inflate.py`, `inflate.sh`, and `runtime/residual_archive.py`. MAIN must not seal or fire a tree whose own dependency manifest does not describe that tree.
2. RLC1 has no timing seal. Its retained cold public runs were refused because competing-process count was unavailable and there is no matched admissible local/T4 calibration. The changed receiver cannot inherit move40's timing.

These are candidate-custody and timing conditions, not evidence that the rule-118 cure failed. Any change beyond a manifest-only refresh reopens this compliance judgment. A manifest-only refresh still changes the runtime-tree digest and therefore requires every downstream receipt to be rebound to the new digest.

## Acceptance table

| Requirement | Verdict | Runtime evidence |
|---|---|---|
| Lane semantic class is counted | CLEAR | `runtime/rlc1_geometry.c:48,67` compare to `cfg[0]`; `runtime/rlc1_mixer.py:25,43,75` passes archive-derived config; direct archive parse gives `target_class=1` (Lane). No literal class `1` selects Lane in the C geometry. |
| Lane row band is counted | CLEAR | `runtime/rlc1_geometry.c:117` gates with `cfg[1]` and `cfg[2]`; direct archive parse gives lower `128`, upper-exclusive `320`, hence rows 128--319. |
| TC4 Movable id | NOT APPLICABLE BY OMISSION | No TC4 map or selector exists in the seven-file candidate delta. The RLC1 rider contains no TC4 map and `NO_SEAL.json` states `tc4_maps: not included`. Movable class id 3 is neither embedded nor consumed. Any TC4 revival must count its class id and map selector in the archive; the current candidate cannot claim a TC4 result. |
| Remaining fitted geometry quantities are counted | CLEAR | `runtime/rlc1_geometry.py:21,24-44` parses exactly `<BHH8B6B>`; `runtime/rlc1_geometry.c:92,95,98-103,107-111,128` consumes only `cfg[3:17]`. |
| Forty fitted mixer weights are counted | CLEAR | `runtime/rlc1_mixer.py:20-32,38-46`; the 35 inherited and 5 new int8 weights are sliced from the rider, never supplied by code. |
| Receiver selects integer geometry | CLEAR, GEOMETRY SCOPE | `runtime/residual_archive.py:527-532,658-663,739-752` routes an RLC1 rider to `LaneMixer`; `runtime/rlc1_mixer.py:74-83` routes geometry to `LaneGeometry`; `runtime/rlc1_geometry.py:62-74,96-117` calls the compiled C library from `inflate.sh:75-76`; `runtime/rlc1_geometry.c:1-130` has no floating type or operation. |
| Whole receiver is integer-only | NOT CLAIMED | `runtime/rlc1_mixer.py:64-65` retains the inherited `np.float64` KT ratio/table update. This is outside the cured geometry, but it is reachable. The public twins are the evidence for that inherited portion; do not restate the result as all-receiver integer or cross-host proof. |
| Two independent public executions | CLEAR, BOUNDED | `experiments/ddm_rlc1_public.py:83-98,138-169` creates separate work roots and invokes `bash candidate_runtime/inflate.sh`; `:199-213` compares all bytes with the retained move40 raw and validates the report. Retained receipts cover cold BLAS settings 1 and 4. |
| Counted bytes and rate accounting | CLEAR | `runtime/rlc1_mixer.py:20-25,38-46`; direct member/rider parses below. |
| Logical imports and scorer cleanliness | CLEAR | `inflate.py:6-16,54`; `runtime/residual_archive.py:529,531,640-643,657,660`; `runtime/rlc1_geometry.py:9-15`; `runtime/rlc1_mixer.py:10-13,27`. No changed file imports or embeds SegNet, PoseNet, evaluator weights, GT labels, source video, or network content. |
| Runtime dependency manifest | CONDITION FAILED | `MANIFEST.sha256:10,11,41` has stale hashes and the file ends at line 46 without any `rlc1_*` entries. |
| Strict scorer | CLEAR, NO RUN | The delta is decode-only and the archive contains exactly stored member `p`; no scorer content was added. I did not run a scorer and make no score claim. |
| Timing | CONDITION OPEN | `inflate.py:56-68` declares CUDA normally and uses advisory CPU only by opt-in. Retained timing is advisory and explicitly unsealed. |

## Complete literal and table provenance

I independently enumerated executable numeric tokens from the current seven-file delta and checked their current source lines against move40. The result matches the retained audit's 443 occurrence rows: 256 occurrences are on exact source lines inherited from move40 and 187 occur in new or edited lines. I also separately inspected numeric text embedded in strings and compiler switches, which Python/C numeric-token scans do not enumerate. The retained audit was therefore a cross-check, not the premise.

The table below accounts for every changed-decoder occurrence by source region. Repeated loop indexes, lengths, masks, and zero/one control values are grouped only where the same named a-priori rule fixes every occurrence.

| Runtime file and lines | Literals/tables covered | Provenance rule |
|---|---|---|
| `inflate.py:13-14,49,56-60,68` | default proof BLAS `4`; public file id string `0`; advisory switch string `1`; T4 label; budget text `1,800`; Torch threads `4` | Generic public-entrypoint/test controls. File id and 1,800-second budget come from the public contest interface; 1/4 select the two proof configurations, not decoded content. None predicts a video label. |
| `inflate.py:18-19,28-31,35-39` | archive SHA and `180173`; one-member `p` shape | Artifact identity and strict container validation. The size is measured from the counted archive, not a reconstructed semantic value. |
| `inflate.sh:4-14,20-35,46-86` | argument count/indexes, exit codes, Brotli `1.2.0`, compiler `-O3`, C standard `c11`, branch/index controls, public id `0` | Generic shell, dependency-version, compiler, and public-interface controls; the 34 numeric occurrences inherited from move40 remain exact-line identical. `inflate.sh:75` adds only the same generic compiler controls to build the counted-config geometry implementation. |
| `runtime/residual_archive.py` unchanged exact lines | 221 occurrences | Exact source-line membership in move40, including format widths, public dimensions, codec framing, masks, sentinels, checkpoint cadence, and inherited algorithms. They are not new RLC1 semantic content. |
| `runtime/residual_archive.py:527-532` | outer bit `0x80`, four-byte magic prefix length `4` | Generic rider framing: bit announces a counted tail and four bytes distinguish its wire-format magic. The actual magic bytes are `RLC1`, not a video statistic. |
| `runtime/residual_archive.py:658-665,743-765` | config length `60`; control/index values; report strings `Q16`, `TC1M-35-int8` | `60 = 1 variant + 40 counted weights + 19 counted geometry bytes`. Other values are branch/control and truthful format labels. They do not supply fitted values. |
| `runtime/rlc1_geometry.py:17-21` | `H=384`, `W=512`, `K=5`; group map `%64 + 2*(%64)`; group count `190`; format `<BHH8B6B>` | A-priori public geometry/alphabet and inherited HPAC schedule. Group ids span 0--189 because `x mod 64 + 2(y mod 64)` on public coordinates has that exact range. The wire schema, not any field value, fixes 19 bytes. |
| `runtime/rlc1_geometry.py:24-44` | size/indexes; class/band domain; bounds `1..16`, `1..4`, `1..255`, `1..255`, `1..2`, `1..255`, `1..255`, final edge `<64`; ordered six-edge check; `int64` dtype | A-priori fail-closed parser/domain rule. These are type/cardinality and overflow-safety bounds, not defaults: the receiver rejects missing config and obtains every accepted field from the archive. No bound selects the observed field value. |
| `runtime/rlc1_geometry.py:50-120` | public shape checks; zero/K class domain; C types; state/counter `0/1`; uint8/int64 dtypes | Generic FFI, type, shape, allocation, causal-order, and class-alphabet checks. No fitted table or seed exists here. |
| `runtime/rlc1_geometry.c:7-19` | `H=384`, `W=512`, `K=5`, slots `S=8`, `Q=65536`; cfg length `17`; six moment arrays | A-priori public geometry/alphabet; eight inherited 64-column sublattices (`512/64`); Q16 scale `2^16`; 17 parsed fields; six sufficient moments of a quadratic centerline fit. No selected parameter value is defined. |
| `runtime/rlc1_geometry.c:21-38` | zero/one; integer-root start shift `62`; two-bit steps; divide-by-two; nearest-even half/tie/parity arithmetic | A-priori signed floor division, 64-bit integer square root, and nearest-even Q16 conversion. These rules are generic arithmetic and independent of video data. |
| `runtime/rlc1_geometry.c:39-73` | zero/one/six loop and state values; cfg index `0` | Generic allocation, six-moment accumulation, run counting, and bitset operations. The semantic class itself is archive field `cfg[0]`, directly parsed as Lane id 1. |
| `runtime/rlc1_geometry.c:75-113` | zero/one/six/eight loop/index/cardinality values; cfg indexes `3..10`; Q16 operations | Generic control structure. Actual history, prior, count, residual, slope, width, multiplier, and span values are all read through the named cfg fields. |
| `runtime/rlc1_geometry.c:114-130` | sentinel bins `8` and `7`; cfg band indexes `1,2`; six edge indexes `11+j`; zero/one loop controls | A-priori nine-context alphabet: six ordered distance intervals plus valid-no-fit and outside-band sentinels. Actual band and six thresholds are archive fields. |
| `runtime/rlc1_mixer.py:15-25` | `K=5`, `H=384`, `W=512`, `BINS=9`; `1+40+FORMAT.size`; magic prefix `4`; versions/masks `1,17,15,16`; config offset `41` | A-priori public geometry/alphabet, nine-context schema, and wire framing. `40 = 35` inherited weights plus `K=5` new feature weights; `41 = 1` variant plus 40 weights. Actual weights/settings are sliced from the rider. |
| `runtime/rlc1_mixer.py:38-54` | config `41+19=60`; variant `1`; slices `1:41`, `:35`, `35:`; array dimensions | A-priori wire/schema cardinality. The compared length and variant validate the packet; neither supplies content. All 60 content bytes originate in the rider. |
| `runtime/rlc1_mixer.py:60-68` | inherited KT pseudocount `0.5`; probability ratio clip `1/16..16`; `False` | Inherited TC3 online adaptive-mixer rule. It operates only on causally decoded counts/expectations and contains no prefit table. This is the reachable float64 portion named above. |
| `runtime/rlc1_mixer.py:70-135` | archive offset `41`; bin sentinel `8`; feature axis `0/-1`; zero; `K`, `BINS`, `K*BINS*K`; key-prefix slice `4` | Generic array axes, no-feature sentinel, cardinalities, causal online-count update, and checkpoint schema. Geometry settings remain `self.config[41:]`; tables start at zero and are learned only from the decoded prefix. |

There is no runtime seed. The seed `20260910` appears only in the retained offline reference proof and candidate provenance. There is no static fitted geometry table: the only persistent video-derived table-like quantities in this delta are the forty counted int8 weights and seventeen counted geometry fields; online counts begin at zero and depend only on prior decoded symbols.

## Archive-read proof and exact configuration

I parsed `candidate_runtime/archive.zip` through the candidate runtime's own `read_residual_archive` and also compared its bytes to the retained rider/config. The archive contains one stored member `p`.

- Variant: `1`.
- Forty counted int8 weights: `[-2, 26, -20, 15, 17, -1, 3, -7, 25, -1, 11, 17, -1, 2, 5, 19, -18, 7, 14, -2, 5, 7, 20, -12, 2, 10, -2, 5, 11, 13, -14, 9, 10, -2, 8, 29, 23, 6, 10, 5]`.
- Nineteen-byte geometry struct `<BHH8B6B>`: target class `1` (Lane); row lower `128`; row upper-exclusive `320`; history window `16`; current/prior multiplier `4`; minimum count `4`; residual maximum `36`; slope maximum `2`; width denominator `2`; width multiplier `3`; span maximum `24`; distance edges `[0, 1, 2, 4, 8, 16]`.
- The complete 60-byte config is byte-identical to `retained/config.bin`, SHA-256 `76f10171e42d27e6a1966a5a4a0a530c18541415b2a60a5fe9f8c2bb8af3d91a`. The geometry suffix is `retained/geometry_config.bin`, 19 bytes, SHA-256 `3f9d0f64699308047b8d90cd51800a1518d0cea39ea449d15681b2efc2b38619`.
- The current 119,534-byte token stream follows the rider and is not reconstructed by code.

The only apparent numeric coincidence between parser bounds and selected values is insufficient to reconstruct the selection: the receiver has no default path, rejects a missing/incorrectly sized config, and reads each actual field from the packet. I did not find a code literal that replaces any of the values above.

## Integer geometry and public proof coverage

The executed geometry call chain is:

`inflate.sh:75-76` -> `inflate.py:63-70` -> `runtime/f26_inflate.py:398-428,604` -> `runtime/residual_archive.py:485-532,633-765` -> `runtime/rlc1_mixer.py:70-95` -> `runtime/rlc1_geometry.py:50-117` -> `runtime/rlc1_geometry.c:39-130`.

There is no alternative Python/float geometry implementation in that RLC1 branch. The native geometry uses signed integer/wide intermediates, Q16 scaling, integer square root, floor division, and nearest-even conversion. The retained independent reference control covers 32 seeded full frames, 6,291,456 positions, zero context mismatches, and byte-identical retained native/reference arrays. This is a library-level geometry oracle check, not the public-entrypoint proof.

The stronger end-to-end evidence consists of two independent cold executions of the public shell entrypoint, one with `RLC1_PROOF_BLAS_THREADS=1` and one with `=4`, each in its own work/output/checkpoint/attempt directory. Both run on host `mac.lan`, Darwin arm64, macOS-CPU advisory, Torch 2.12.1, Torch intra-op threads 4 and inter-op threads 1. Each reports 600 pairs, no checkpoint resume, decoded-token SHA-256 `b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5`, and raw output 3,662,409,600 bytes with SHA-256 `c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5`. The proof script compares every raw byte, not only hashes, against the retained move40 public output; `NO_SEAL.json` records equality across the two configurations.

Coverage boundary: this proves byte identity on one macOS/arm64 host under two BLAS settings and the public entrypoint. It does not establish cross-host, contest-CUDA, or whole-receiver integer identity. I did not rerun either 25-minute decode and did not treat an attempted fresh large-file comparison as evidence because it did not complete during this review.

## Counted-config accounting

All figures are exact serialized bytes.

| Component | move40 | RLC1 | Delta |
|---|---:|---:|---:|
| RC64 token stream | 119,618 | 119,534 | -84 |
| Fitted weight bytes | 35 | 40 | +5 |
| Counted geometry config | 0 | 19 | +19 |
| Rider/member/archive net | -- | -- | **-60** |
| Stored ZIP archive | 180,233 | 180,173 | **-60** |

The RLC1 rider is 119,598 bytes (`4` magic + `1` variant + `40` weights + `19` settings + `119,534` token bytes) and is the exact suffix of archive member `p`. The corresponding move40 rider is 119,658 bytes (`4 + 1 + 35 + 119,618`). The non-rider member prefix and stored-ZIP overhead (100 bytes) are unchanged. Therefore `-84 + 5 + 19 = -60` and `180,233 - 60 = 180,173`. Candidate archive SHA-256 is `8c2eaefa944ca8acba3db80829a5cd7a3d06a17b1fdb7ed5f39ce5a6fd9a124d`; move40 is `986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.

## Dependency and strict-scorer closure

The seven-path delta is exactly `archive.zip`, `inflate.py`, `inflate.sh`, `runtime/residual_archive.py`, and new `runtime/rlc1_geometry.c`, `runtime/rlc1_geometry.py`, `runtime/rlc1_mixer.py`. Python imports added by the cure are stdlib, NumPy, conditional Brotli, and local runtime modules already required by the base receiver. C imports only `stdint.h`, `stdlib.h`, and `string.h`. Public execution reached and completed all 600 pairs, so the logical dependency graph closes on the reviewed host.

No changed source imports the scorer or contains SegNet/PoseNet weights, a GT table, source frames, or a fetch/network path. `archive.zip` contains only member `p`. This satisfies the strict-scorer content rule for the reviewed delta. It does not excuse the stale dependency manifest: current logical closure and manifest closure are separate gates.

Current runtime digest, independently recomputed with `tac.candidate_seal.measure_runtime_digest`, is `32d5e269c310ecec35f7a13349f842e9562daf586e6ce4d3d6553e4f54f9b846` over 51 files and 1,009,263 bytes; move40 is `24d372dcca5ca619b2716a1792a9e38d7e966d0dd1c66c3320e9ced58e12e91a` over 48 files and 994,175 bytes. The retained public receipts bind the current RLC1 digest. Refreshing `MANIFEST.sha256` will necessarily produce a different digest and invalidate that binding until regenerated.

## Timing refusal and exact clearance requirement

I did not measure timing. The retained cold public rows are `[macOS-CPU advisory]`:

- BLAS 1: wall 1,494.551 s; inflate report 1,490.794 s; token stage 1,048.095 s.
- BLAS 4: wall 1,518.197 s; inflate report 1,514.442 s; token stage 1,046.820 s.

Both were correctly refused with `SealContractError: decode_wall_clock: competing process count absent`. There is no RLC1 T4 row, and move40 has different receiver code.

Clearance requires, in order: refresh and validate the runtime manifest; recompute the runtime digest; rerun/rebind the public twin evidence on exactly that runtime; obtain a cold public-entrypoint timing with trusted process/concurrency custody in a controlled environment; bind a matched local/T4 calibration for this receiver and show projected contest-T4 decode at or below 1,260 seconds; create a `decode_wall_clock` seal naming the exact archive/runtime/hardware/command; then have MAIN consume this memo and the new receipts before any fire. An actual contest-T4 cold public decode on the exact tree can replace projection uncertainty if it carries the same custody.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus by content for `RLC1`, `rule 118`, `receiver`, `counted config`, `integer geometry`, `Lane class`, `Movable`, `timing`, `decode_wall_clock`, and the move40/candidate hashes; searched `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED surfaces, design docs/SPECs, and task-ledger rows; and generated/searched the canonical-equations registry.

Beyond the charter seeds, three items changed the review:

1. The current DWC1 timing-contract work says a receiver-changing candidate needs its own timing/calibration and cannot inherit move40. That made the timing condition receiver-specific rather than a generic pending measurement.
2. Canonical integer-decoder and Lane-bound equations describe the intended fixed-point and causality laws but explicitly do not transfer archive-bound or successor-candidate validity. I therefore used them only to classify generic arithmetic, not as proof of RLC1.
3. Direct candidate-manifest verification exposed the stale `MANIFEST.sha256`, a fire blocker absent from the charter's expected prediction. This added a custody condition before timing or seal.

I did not find, in the searched corpus, a compliant precedent that would allow a video-selected semantic class or row band to remain free receiver code. The RLC1 counted fields are therefore necessary, not optional hygiene.

## Follow-on dispositions

- **QUEUED-WITH-A-FIRE-ORDER** — owner: RLC1 artifact owner / MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime/MANIFEST.sha256` plus refreshed RLC1 public receipts; fire trigger: before any new timing, seal, or MAIN fire. Action: refresh all runtime-manifest hashes, add the three RLC1 modules, make `shasum -a 256 -c` pass, recompute the tree digest, then regenerate or rebind public proof to that exact digest.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / DWC1 timing lane; consumer store: `.omx/research/ddm_rlc1_20260910/TIMING_FIRE_ORDER.json` and the candidate seal store; fire trigger: manifest-clean exact runtime and a controlled process-visible timing environment or contest-T4 access. Action: measure the exact post-manifest receiver cold, establish receiver-matched T4 clearance at no more than 1,260 seconds, and create the timing seal.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: canonical frontier pointer and evaluation ledger; fire trigger: both prior conditions clear and the exact archive/runtime pair is sealed. Action: run the authorized exact contest evaluation and move the pointer only if the exact result beats move40.
- **FOLDED** — owner: MAIN; consumer store: `.omx/research/ddm_rlc1_20260910/TIMING_FIRE_ORDER.json`; fire trigger: only if a later, separately priced TC4 successor is chartered. Action: require any TC4 Movable id/map selector to be counted; do not attach TC4 claims to RLC1.

## Live hypotheses

- A clean manifest-only refresh should preserve the decoded bytes because `MANIFEST.sha256` is not in the runtime digest's executable call path; this remains untested until the public receipts are rebound to the post-refresh tree.
- A controlled or direct T4 measurement may clear timing because the retained advisory runs spend roughly 1,047 seconds in token decode and the public path declares CUDA for the renderer; this is plausible but not established by the loaded macOS rows or move40's different receiver.

## Dead ends

- Treating the retained literal audit as proof is closed: an independent tree/token and exact-line census was required and performed.
- Treating the RLC1 receiver as float-free is closed: geometry is integer, but the inherited online KT calibration at `runtime/rlc1_mixer.py:64-65` is reachable float64.
- Inheriting move40 timing is closed because RLC1 changes the receiver.
- Claiming TC4 or Movable evidence from this candidate is closed because TC4 maps are omitted.
- Firing the current candidate directory is closed until its stale manifest, proof binding, and timing seal are cured.

**Live frontier (submittable): move 40, S = 0.13763861019288715 @ 180,233 B `[contest-CUDA T4 n600]`, archive SHA-256 `986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`. RLC1 did not move the pointer.**
