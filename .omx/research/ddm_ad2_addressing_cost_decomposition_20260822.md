# AD2 outcome — NR1 QPAIR has a real 17,957 B lossless ordering win; RC1 addressing, NR1 QCTX, and NR1 QEVENT do not

**Disposition:** `QUEUED-WITH-A-FIRE-ORDER` for one isolated NR1 receiver integration. **Pointer:**
unmoved. **Axis:** `[macOS-CPU scorer-free retained-receipt representation measurement n600]`.
No scorer, Metal job, Modal job, advisory evaluation, live-candidate mutation, or `upstream/` write
occurred.

The candidate-specific prior law survives at **INSTANCE** scope. RC1-K2048 spends 10,900 / 59,884 B
(18.2019%) on its spatial assignment map and 48,920 B (81.6913%) on its codebook; the tested new
assignment layouts save **0 B**. NR1-K32 spends 52,124 / 69,004 B physically (75.5376%) on QPAIR;
changing only its generic serialization order from pair-major raster to tile-major time makes the
real Brotli-q11 stream **34,083 B instead of 52,040 B**, a measured **17,957 B** substream cut.

That is not yet an archive or score. NI1 independently landed a byte-closed 122,250 B archive,
SHA-256 `fe7fe805…ca534e`, with unmeasured distortion. Applying AD2's isolated byte delta to that
archive gives a **projection** of 104,293 B, 33,693 B below the fixed-distortion ceiling, and a
projected rate-term reduction of 0.011956829221. The projection is not promoted: the new QPAIR layout
still needs a receiver representation ID, exact packet/archive rebuild, repeated full-RGB decode
identity, and governed scoring.

The authoritative receipt is
`/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/measurement_v6/RESULT.json`,
SHA-256 `80124acd71ff63d4d9379b87674d1a976e1aa73857b4062a1c9ea2afb1b73511`. The sealed fire order is
`measurement_v6/SEALED_FIRE_ORDER.json`, SHA-256
`acea6ebbece5f5ac4adaeda13160fb7ceadd69d86d08a3fedf7494d7bbfc44c6`.

## Inherited-state verification

All inherited pins matched; there was no custody drift.

| object | bytes | verified SHA-256 |
|---|---:|---|
| RC1 K2048 payload | 59,884 | `eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164` |
| NR1 K32 packet | 69,004 | `a68765dc683fa8302b560ef3db0d4a1507eeeccc695322fb8b69f684ed6dab28` |
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| NI1 landed archive | 122,250 | `fe7fe8058376543d5832912e691214969680fea5d85e125e861e9700c5ca534e` |

Module pins also matched: RC1 `6c2ea6f3…cbfbc9`, NR1 `66500b81…01cbb6`.

The four charter arithmetic anchors reproduce from retained receipts:

- DC1, denominator 190 selected full-n600 sparse-grid groups: 1,819,325 position bits =
  **227,415.625 B**; 814,021 per-block length bits = **101,752.625 B**.
- RC1: **10,900 / 59,884 B = 18.2019%** spatial assignment stream.
- NR1: QPAIR physical attribution **52,124 / 69,004 B = 75.5376%**.

## Addressing versus payload

### RC1 K2048

Denominator: the exact 59,884 B `RC1V` payload.

| counted stream | class | bytes | fraction |
|---|---|---:|---:|
| fixed header, dimensions, methods, lengths, CRCs, decoded hash | addressing / how-much / framing | 64 | 0.1069% |
| spatial assignment IDs at implicit raster sites | addressing | 10,900 | 18.2019% |
| 2,048 × 600 temporal program codebook | payload / what | 48,920 | 81.6913% |

The raster coordinates are generic and free. The assignment ID at each coordinate is video-derived
and counted. The lexicographically canonical codebook order does not determine which program belongs
at any site.

### NR1 K32

Denominator: the exact 69,004 B `NR1Q` packet. Logical coded-stream accounting is separate from the
canonical physical attribution, because the first physical section owns the outer header.

| counted stream | class | coded bytes | packet fraction | physical bytes including assigned headers |
|---|---|---:|---:|---:|
| outer + four section headers | addressing / how-much / framing | 382 | 0.5536% | distributed below |
| QPARAM dictionary | payload / what | 109 | 0.1580% | 239 |
| QCTX baseline ID per implicit tile | addressing | 68 | 0.0985% | 152 |
| QPAIR temporal/context choice per implicit pair×tile | addressing | 52,040 | 75.4159% | 52,124 |
| QEVENT delta-coordinate + class-value corrections | **mixed, not separable after coding** | 16,405 | 23.7740% | 16,489 |

QEVENT cannot be assigned a truthful coded address/value split: each sorted delta-ULEB coordinate is
interleaved with one class byte and the combined raw stream is Brotli-coded. Its address entropy is
priced separately below, but no invented coded-byte allocation is reported.

### DX2 incumbent

Denominator: exact 180,368 B archive. These rows sum exactly.

| physical region | class | bytes | archive fraction |
|---|---|---:|---:|
| ZIP framing | framing / how-much | 100 | 0.0554% |
| RX1 header | addressing / how-much | 14 | 0.0078% |
| learned HPAC probability model | addressing / how-to | 13,515 | 7.4930% |
| semantic renderer | payload / what | 30,856 | 17.1072% |
| carrier stream | **mixed payload + basis/coefficient metadata** | 22,010 | 12.2028% |
| compact residual | payload / what | 96 | 0.0532% |
| semantic tokens at implicit raster sites | payload / what | 113,777 | 63.0805% |

The carrier's compressed bytes cannot be split without changing its representation. The HPAC table
is video-derived and is required before token decoding; deriving it from decoded tokens would be
circular. RB1's same-object coder races remain authoritative: the fixed DX2 representation has **0 B**
measured recode headroom. AD2 does not reopen that result.

## Entropy-model prices

Every row below has retained symbol/context counts. “Context bound” means the ideal empirical data
term for the explicitly named first-order model. It excludes model-description and finite-sample
costs, is not a shippable size, and is not a universal lower bound: a richer-context real coder can
beat it. This distinction is load-bearing; ideal entropy is diagnostic, while admission comes from
the real-coder table in the next section.

| stream | symbols | memoryless plug-in | named causal context | context-model bound | incumbent coded | incumbent gap |
|---|---:|---:|---|---:|---:|---:|
| RC1 assignments | 196,608 | 87,105 B | left ID in each raster row; row-start sentinel | 7,317 B | 10,900 B | 3,583 B (32.87%) |
| NR1 QCTX | 3,072 | 605 B | left baseline ID in each tile row; row-start sentinel | 27 B | 68 B | 41 B (60.29%) |
| NR1 QPAIR | 1,843,200 | 66,293 B | same tile's symbol at previous pair; pair-zero sentinel | 44,995 B | 52,040 B | 7,045 B (13.54%) |
| NR1 QEVENT addresses only | 8,192 | 12,110 B | previous gap's ULEB byte length; initial zero | 11,458 B | not identifiable | not identifiable |
| DX2 HPAC raw bytes | 17,952 | 14,961 B | immediately previous raw byte; stream-start sentinel | 10,488 B | 13,515 B | 3,027 B (22.40%) |

The charter's ≥20% prediction is met by QCTX's context-model gap, and RC1 and DX2 also show such
diagnostic gaps. That does **not** make those bytes available: the actual tested QCTX and RC1 layouts
save 0 B, and RB1 closes DX2's fixed-representation coder set at 0 B. Conversely, tile-time QPAIR's
34,083 B real stream beats the listed 44,995 B first-order reference because Brotli sees longer
tile-local temporal structure after the reorder. This is direct evidence that the reference is
model-scoped rather than universal.

## Real lossless representation races

Every raw form, RAW/zlib9/LZMA1-1MiB/Brotli-q11 output, losing variant, and deterministic repeat is
retained. Every form is inverted to the exact source symbol array before its byte result is admitted.

| candidate / stream | form | winning real coder bytes | delta versus incumbent |
|---|---|---:|---:|
| RC1 assignment | incumbent raster u16 + Brotli q11 | **10,900** | 0 |
| RC1 assignment | 8×8-block u16 | 14,353 | -3,453 |
| RC1 assignment | raster fixed-11 | 19,312 | -8,412 |
| RC1 assignment | 8×8-block fixed-11 | 19,846 | -8,946 |
| NR1 QCTX | incumbent raster u8 + Brotli q11 | **68** | 0 |
| NR1 QCTX | 8×8 block u8 | 97 | -29 |
| NR1 QCTX | raster fixed-5 | 84 | -16 |
| NR1 QCTX | 8×8 block fixed-5 | 112 | -44 |
| NR1 QPAIR | incumbent pair-major raster u8 + Brotli q11 | 52,040 | baseline |
| NR1 QPAIR | **tile-major time u8 + Brotli q11** | **34,083** | **+17,957** |
| NR1 QPAIR | tile-major time fixed-6 | 39,457 | +12,583 |
| NR1 QPAIR | pair-major raster fixed-6 | 55,805 | -3,765 |
| NR1 QPAIR | pair-major 8×8 block u8 | 53,743 | -1,703 |
| NR1 QEVENT | incumbent interleaved + Brotli q11 | **16,405** | 0 |
| NR1 QEVENT | split address bytes then values + Brotli q11 | 16,421 | -16 |

Positive delta means fewer bytes. QCTX and QEVENT are folded at this instance. RC1's tested
address-layout family is closed at this instance; CB2 owns the codebook-side mechanism and AD2 did
not touch it. The QPAIR delta is a standalone exact substream measurement, not a measured packet or
archive delta. The combined NR1 figure is therefore **17,957 B upper-bound supply**, not a composed
archive claim; union is not assumed to equal the sum of legs.

## Receiver derivability and rule 118

The winning QPAIR rule is generic: read `(pair_count, gh, gw)` from the existing QPAIR header, traverse
`tile_y, tile_x, pair` rather than `pair, tile_y, tile_x`, Brotli-decode the paid symbols, then apply
the inverse transpose before the unchanged QPAIR semantics. It needs no contest-video table or learned
side information. The traversal algorithm is free receiver code; all QPAIR symbols remain counted.

No assignment stream is free merely because its coordinates are implicit. Retained ambiguity
witnesses hold the other paid surfaces fixed and change one addressing choice:

| stream | held fixed | changed | decoded-token difference | verdict |
|---|---|---|---:|---|
| RC1 assignment | canonical codebook and shape | swap two adjacent assignment IDs | 4 | IDs not derivable from codebook + raster coordinates |
| NR1 QCTX | QPARAM, QPAIR, QEVENT | one consumed baseline ID | 38,400 | baseline IDs not derivable |
| NR1 QPAIR | QPARAM, QCTX, QEVENT | one pair×tile choice | 64 | pair choices not derivable |
| NR1 QEVENT | QPARAM, QCTX, QPAIR | one class value at the same stored coordinate | 1 | corrections not derivable |

Each alternate packet/payload, all changed-section coder outputs, repeat, and 117,964,800-byte decoded
witness is retained. These are semantic ambiguity proofs, not scorer effects. Moving any of these
video-derived IDs, coordinates, probability tables, or class values into `inflate.py` would be the
rule-118 hide-data-in-code fake.

## Per-candidate routing

| rank | candidate | measured lever | measured delta | disposition |
|---:|---|---|---:|---|
| 1 | NR1 K32 | tile-major time QPAIR | **17,957 B** | `QUEUED-WITH-A-FIRE-ORDER` for isolated receiver integration |
| 2 | NR1 K32 | QEVENT split | 0 B | `FOLDED` at this instance |
| 3 | NR1 K32 | QCTX reorder/packing | 0 B | `FOLDED` at this instance |
| 1 | RC1 K2048 | tested assignment reorder/packing | 0 B | `CLOSED` at instance scope; route to its codebook half |
| 1 | DX2 | fixed-representation coder work | 0 B | `CLOSED` by RB1; not reopened |

The prior-law verdict is **CONFIRMED at INSTANCE scope** for the exact retained RC1-K2048 and
NR1-K32 packets. It is not a family theorem. RC1 remains codebook-dominated; NR1 is addressing-
dominated and has a large, achieved ordering win. The falsifier—every addressing stream within about
5% of its named context-model bound—does not fire.

## RECALL EVIDENCE

The search covered `.omx/research/` memos and receipts, the canonical research index, DAG FEED blocks,
the canonical-equations JSON, task ledgers, current hot state, and retained APDataStore results. Content
queries included `address|position|implicit|support|assignment|QPAIR|quotient|entropy|receiver-derived`,
plus exact section sizes and payload hashes.

Beyond the charter's seeds, the recall changed the measurement in four ways:

- IG1 (`8ec60069…42ce8b`) refutes “implicit always wins” and requires the smallest complete
  receiver-closed action. AD2 therefore treats ordering as a free generic algorithm but keeps every
  video-derived symbol counted, and admits only a same-object real-coder win.
- SE3 (`a61bfb84…63b64`) priced 81,365 B / 100,904 B Road-Lane descriptions but could not derive its
  band coordinates from the receiver; the coordinates came from a stand-in argmax. AD2 therefore
  requires exact receiver inputs for every derivability claim.
- SP1 (`6bf5dcfd…bbda`) closes its explicit support formulation at 421,366–444,394 B. AD2 does not
  reopen explicit position fields; it tests whether existing implicit sites can be traversed better.
- NI1 (`2839ea3f…53b2e`) landed the exact 122,250 B shipping archive while leaving distortion unmeasured.
  Its row replaces the charter's older 135,595 B projection in AD2's downstream arithmetic.

RB1 (`fa26a444…64f09`) remains the fixed-DX2 negative control. No complete free-address result beyond
these scoped surfaces was found in the searched corpus; that is bounded absence, not a universal claim.

## Verification, custody, and boundaries

- `measurement_v6` retains 199 manifest-listed artifacts plus its top-level `RESULT.json`; the tree is
  about 636 MiB. Resume verification rehashed all 199 listed artifacts successfully.
- The run is deterministic and resumable from its completed root. Stage pins, measurement completion,
  source copy, storage preflight, all entropy counts, all real-coder variants/repeats, all ambiguity
  packets, and all decoded witnesses are retained on APDataStore.
- `measurement_v1` stopped on a false local RX1 invariant: the exact pinned header uses Brotli codec ID
  2 and known nonzero feature bits. Its already-written inputs remain retained. The DX2 member it read
  was already durably present inside the pinned archive and is separately retained in every completed
  v2–v6 run. No payload or candidate was mutated.
- v2–v5 are retained complete superseded runs. v6 is authoritative because it serializes the exact
  conditioning rule alongside every entropy count table, includes NI1's landed pins and sealed fire
  order, and makes an interrupted same-root rerun validate rather than trust existing frame streams.
- After full-tree certification, v2–v5 were copy-verified to Vertigo cold store and their APDataStore
  source paths replaced with symlinks, reclaiming 2,470,753,954 B. `COLD_STORE_PLAN.json` SHA-256 is
  `96188a0b…6801b`; `COLD_STORE_EXECUTION.json` SHA-256 is `daf2d1a3…d3ee06`. Each execution row records
  original path, destination, bytes, complete file/tree SHA-256, rebuild command, and false-authority
  flags. v1 and authoritative v6 were not moved.
- No d_seg, d_pose, Lane retention, S, exact-evaluator result, or contest promotion was measured.

## NEXT_IF_RESUMED

- `ad2_qpair_tile_time_receiver_integration`; disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN assigns ddm_ni2_nr1_qpair_tile_time_receiver`; consumer store=`/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/qpair_tile_time_receiver_r1/`; fire trigger=`NI1 build_r4 scorer harvest is terminal or MAIN explicitly forks an isolated successor, NI1 and AD2 pins revalidate, and the shared staged index is empty`; action=add a receiver-recognized tile-time QPAIR representation ID, rebuild the exact NR1 packet and archive, and require two byte-identical full-RGB decodes matching current NI1 K32 output.
- `ad2_qpair_tile_time_main_scorer`; disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN scorer-lane dispatcher`; consumer store=`/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/qpair_tile_time_receiver_r1/harvest/`; fire trigger=`integration holds a byte-closed archive, repeat-identical full-RGB decode, exact equality to NI1 K32 output, and MAIN holds the sole n600 scorer slot`; action=run the governed advisory and per-class repeats on those exact integrated bytes.

## LIVE-HYPOTHESES

- Tile-major QPAIR will realize nearly the full 17,957 B inside NI1 because its representation-ID
  change can reuse the existing one-byte section codec field and the archive stores the packet without
  another entropy layer. This is plausible from the packet grammar, but only an exact rebuilt archive
  can establish the delta.
- The QPAIR win may generalize to K64 because it exposes the same tile-local temporal runs and K64 is
  NI1's named lower-distortion fallback. It remains untested; K64 needs its own exact packet race.
- A richer generic QPAIR context than physical reorder alone may approach or beat 34,083 B because the
  tile-time stream still has repeated temporal regimes. Any video-derived context model must be counted,
  so the next admissible lead is a fixed causal transform with a real receiver and coder race.

## DEAD-ENDS

- RC1 8×8 ordering and fixed-11 packing: all are 3,453–8,946 B worse than the 10,900 B incumbent.
- NR1 QCTX block ordering and fixed-5 packing: all lose; the 60.3% entropy-model gap is not shippable
  headroom by itself.
- NR1 QEVENT address/value splitting: 16 B worse, and the current coded address/value bytes cannot be
  truthfully separated.
- NR1 QPAIR fixed-6 packing and pair-major block ordering: both lose to tile-time u8; packing the
  alphabet before the real coder destroys useful byte-level run structure.
- DX2 HPAC recoding from its diagnostic entropy gap: RB1 measured 0 B at the current fixed
  representation; ideal entropy does not reopen that coder race.
- Free derivation of assignment IDs, baseline IDs, QPAIR choices, QEVENT corrections, or HPAC tables:
  retained ambiguity/circularity evidence closes it. Only their generic coordinate traversal is free.

OWN-VEHICLE FRONTIER: UNMOVED — AD2 S NOT MEASURED; DX2 remains S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600].
