# DDM D3C — Road-rooted class-pyramid peel-order screen

Date: 2026-08-26  
Actor: `ddm_d3c`  
Axis: `[macOS-CPU advisory / scorer-free, n600]`  
Source field: `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`  
Code base: `144219690b2828fd5af2526e45b096298c4a4147`  
Retained store: `/Volumes/APDataStore/pact/ddm_d3c_class_pyramid/`

## Result

All 24 Road-rooted four-peel chains were swept; none were deferred. The best reduced-context
KT screen was:

`Lane -> MyCar -> Undrivable -> Movable` (fine to coarse), decoded in the reverse order,
at **238,610.3 B**. The scoped real RC64 control for that chain retained **238,632 B** across
four independently framed rung payloads, only **+21.7 B** over the KT screen.

This is a negative screen for this context formulation, not a codec or family verdict:

- The best ladder screen is **+111,318.3 B** above the 127,292 B joint token subsystem.
- It is **+40,985.2 B** above the comparable D3 single-peel screen of 197,625.0 B
  (49,696 B real four-symbol stream plus the final Lane-plane KT bound).
- The best-to-worst order span is **13,699.5 B / 5.741%**, falsifying the preregistered
  `>=10%` order-effect prediction.
- No chain bounded below the D3 single-peel screen, falsifying the second preregistered
  prediction.
- The live D3B same-field result found beyond the charter seeds is stronger still:
  127,499 B for its HPAC-conditional Lane factorization, only +207 B against 127,292 B.

The own frontier did not move. No archive, scorer, or evaluator was run.

## Measurement definition and denominators

The retained field has 117,964,800 sites (`600 x 384 x 512`) and exactly these source counts:

| Class | Sites | Fraction |
|---|---:|---:|
| Road | 27,406,888 | 23.233% |
| Lane | 691,095 | 0.586% |
| Undrivable | 58,413,222 | 49.518% |
| Movable | 1,460,458 | 1.238% |
| MyCar | 29,993,137 | 25.425% |
| **Total** | **117,964,800** | **100.000%** |

The charter's 24-chain count is `4!`, so one class must be the implicit base. Road was derived,
not asserted, as that base because it is the spatial-embedding host in the D3/LM1 line. The four
non-Road classes were exhaustively permuted. A listed peel order is fine to coarse; decode runs
in reverse, with Road filling the final residual.

Each binary rung is priced only over its active support: all sites not already assigned to a
coarser decoded class. Its context key contains:

- current binary plane at causal left, up, and prior-frame same-site locations, each with an
  unavailable symbol; and
- the complete already-decoded coarse field at left, right, up, down, prior-frame, and
  next-frame radius-one locations, with an out-of-bounds symbol.

This gives 1,259,712 possible generic contexts. The screen records both an exact hindsight
conditional length and a Jeffreys/KT prequential length from the sufficient counts. The context
was reduced from the PP1 temporal winner to limit sparse-context overhead. It is not the D3B
reference form.

`# FORMALIZATION_PENDING: the reduced-context rank screen is not a canonical rate law; a receiver-closed reference-form row is required before registry promotion.`

## Ranked chain table

All byte columns are `[macOS-CPU advisory / scorer-free KT conditional-bound screen, n600]`.
The base Road residual costs zero additional symbols after the four binary decisions.

| Rank | Peel fine to coarse | Decode coarse to fine | Hindsight B | KT B | delta vs 127,292 B |
|---:|---|---|---:|---:|---:|
| 1 | Lane -> MyCar -> Undrivable -> Movable | Movable -> Undrivable -> MyCar -> Lane | 237,918.3 | 238,610.3 | +111,318.3 |
| 2 | Lane -> Movable -> MyCar -> Undrivable | Undrivable -> MyCar -> Movable -> Lane | 238,419.8 | 239,054.0 | +111,762.0 |
| 3 | Lane -> MyCar -> Movable -> Undrivable | Undrivable -> Movable -> MyCar -> Lane | 238,454.4 | 239,140.5 | +111,848.5 |
| 4 | Lane -> Undrivable -> Movable -> MyCar | MyCar -> Movable -> Undrivable -> Lane | 238,540.1 | 239,162.7 | +111,870.7 |
| 5 | Lane -> Undrivable -> MyCar -> Movable | Movable -> MyCar -> Undrivable -> Lane | 238,862.7 | 239,508.3 | +112,216.3 |
| 6 | Movable -> Lane -> MyCar -> Undrivable | Undrivable -> MyCar -> Lane -> Movable | 238,901.3 | 239,615.5 | +112,323.5 |
| 7 | Lane -> Movable -> Undrivable -> MyCar | MyCar -> Undrivable -> Movable -> Lane | 239,236.0 | 239,846.9 | +112,554.9 |
| 8 | Movable -> Lane -> Undrivable -> MyCar | MyCar -> Undrivable -> Lane -> Movable | 239,717.5 | 240,408.5 | +113,116.5 |
| 9 | MyCar -> Lane -> Undrivable -> Movable | Movable -> Undrivable -> Lane -> MyCar | 244,179.8 | 245,028.5 | +117,736.5 |
| 10 | MyCar -> Lane -> Movable -> Undrivable | Undrivable -> Movable -> Lane -> MyCar | 244,716.0 | 245,558.8 | +118,266.8 |
| 11 | Movable -> MyCar -> Lane -> Undrivable | Undrivable -> Lane -> MyCar -> Movable | 244,978.1 | 245,781.2 | +118,489.2 |
| 12 | MyCar -> Movable -> Lane -> Undrivable | Undrivable -> Lane -> Movable -> MyCar | 245,001.2 | 245,889.9 | +118,597.9 |
| 13 | Undrivable -> Lane -> Movable -> MyCar | MyCar -> Movable -> Lane -> Undrivable | 248,112.1 | 248,853.9 | +121,561.9 |
| 14 | Undrivable -> Movable -> Lane -> MyCar | MyCar -> Lane -> Movable -> Undrivable | 248,189.1 | 248,983.0 | +121,691.0 |
| 15 | Undrivable -> Lane -> MyCar -> Movable | Movable -> MyCar -> Lane -> Undrivable | 248,434.8 | 249,199.4 | +121,907.4 |
| 16 | Movable -> Undrivable -> Lane -> MyCar | MyCar -> Lane -> Undrivable -> Movable | 249,026.4 | 249,786.7 | +122,494.7 |
| 17 | MyCar -> Undrivable -> Lane -> Movable | Movable -> Lane -> Undrivable -> MyCar | 250,011.3 | 250,940.8 | +123,648.8 |
| 18 | MyCar -> Undrivable -> Movable -> Lane | Lane -> Movable -> Undrivable -> MyCar | 250,069.2 | 251,016.3 | +123,724.3 |
| 19 | Undrivable -> Movable -> MyCar -> Lane | Lane -> MyCar -> Movable -> Undrivable | 250,676.8 | 251,506.0 | +124,214.0 |
| 20 | Movable -> MyCar -> Undrivable -> Lane | Lane -> Undrivable -> MyCar -> Movable | 250,712.6 | 251,545.9 | +124,253.9 |
| 21 | MyCar -> Movable -> Undrivable -> Lane | Lane -> Undrivable -> Movable -> MyCar | 250,735.6 | 251,654.6 | +124,362.6 |
| 22 | Undrivable -> MyCar -> Lane -> Movable | Movable -> Lane -> MyCar -> Undrivable | 250,942.3 | 251,806.2 | +124,514.2 |
| 23 | Undrivable -> MyCar -> Movable -> Lane | Lane -> Movable -> MyCar -> Undrivable | 251,000.2 | 251,881.7 | +124,589.7 |
| 24 | Movable -> Undrivable -> MyCar -> Lane | Lane -> MyCar -> Undrivable -> Movable | 251,514.1 | 252,309.7 | +125,017.7 |

Swept: 24/24 chains and 32/32 unique `(decoded class set, target class)` rung states.  
Deferred: 0 chains and 0 rung states.

## Top chain, rung denominators, and real bytes

The D3 pinned RC64 source was compiled with alphabet two. Each rung used an exact KT-adaptive
frequency extension. It emitted a real, retained, independently framed payload and an identical
repeat. Decoder control replayed the source-derived context-key sequence and reproduced every
active symbol exactly.

| Decode rung | Target given decoded coarse classes | Active symbols | Ones | KT screen B | Real RC64 B | Payload SHA-256 |
|---:|---|---:|---:|---:|---:|---|
| 1 | Movable given none | 117,964,800 | 1,460,458 | 39,854.6 | 39,860 | `d18ca904402ef2c9f30f9f8ff49f55ec2230299c4cf9ee663a82ad2db99b02ac` |
| 2 | Undrivable given Movable | 116,504,342 | 58,413,222 | 33,711.0 | 33,716 | `075e342d190ba34b1ada18981fa4a9db9e1e508cb31df297252333ae2bc6e4eb` |
| 3 | MyCar given Undrivable, Movable | 58,091,120 | 29,993,137 | 17,115.6 | 17,120 | `f438a050a25bf228ba69234d7dbf497660aa0258107283ab4a24165b974322bc` |
| 4 | Lane given Undrivable, Movable, MyCar | 28,097,983 | 691,095 | 147,929.0 | 147,936 | `7d91932ffe3c0e8e0d5127f66a9a83e4bd8d99610e0d20915be0d4ff94a35244` |
| **Total** | four independent rung envelopes | — | — | **238,610.3** | **238,632** | — |

The repeat payload SHA equals the primary SHA on every row. Packed decoded active-symbol streams
were also retained. They are support-order symbol sequences, not full spatial planes.

### Real-row boundary

This row is **not receiver closed**. The control decoder consumed context keys replayed from the
retained source/coarse field. It proves the real arithmetic mechanism and byte total at this
screen fidelity, but it does not prove that a standalone receiver regenerates contexts from prior
decoded rungs, parses one packet, or reconstructs the entire five-class field independently.
The four payloads also each pay their own `R6D1` magic and u32 padding; no combined packet/model
framing was priced. Therefore this is not a byte-closed codec row, archive row, or family verdict.

## RECALL EVIDENCE

The recall pass searched content across `.omx/research/`, the canonical research index and
`sub015_DAG_*` FEED surfaces, design/SPEC documents, the canonical equation registry, and the
canonical task ledger. Queries included `class peel`, `pyramid`, `bitplane`, `conditional entropy`,
`spatial embedding`, `Road Lane`, `decoded coarse`, `peel order`, `D3`, `D3B`, `LM1`, `PP1`, and
`token ordering`.

Charter seeds read in full were the D3 and LM1 memos, D3B charter including Amendment 1, and GB1
verdict. Beyond those seeds, the pass found:

- `ddm_pp1_direct_partition_pricing_20260728.md`: a full-field temporal KT screen at 173.6 KB,
  with Road and Lane each contributing about 62 KB, and generic independent binary planes at
  660.5 KB. This changed the screen from class-frequency-only KT to a temporal plus decoded-field
  conditional context.
- `ddm_to2_token_ordering_race_20260822.md`: ordering without changed context is not a rate lever.
  This made every chain use its actual decoded-coarse field rather than merely rearranging symbols.
- `segnet_recursive_fractal_factorization_20260715.md`: Road/Lane is the hard spatial interface.
  This supported Road as the derived implicit base and made the final Lane rung a falsifiable target.
- The live read-only D3B store, not yet represented by a repo memo, contains a completed same-field
  HPAC-conditional row: 64,276 B Lane packet, 127,499 B token subsystem, exact field identity,
  result SHA `b9886a92b351c35aa04db913928a11bc96f170d08eb08c1e2f351c8b08d84457`.
  This raised the comparison standard from the charter's prospective D3B reference form to an
  already-measured real row.
- The canonical equations registry contained conditional-entropy and context rate laws, but no
  prior same-object five-class peel-chain table. No prior D3C task row was found before registration.

The D3, D3B, and LM1 stores were read-only throughout.

## Custody and reproducibility

Commands:

```text
.venv/bin/python experiments/ddm_d3c_class_pyramid_peel_order.py all --resume
.venv/bin/python experiments/ddm_d3c_class_pyramid_peel_order.py confirm
```

The second command regenerated the same real payloads after the receipt language was tightened to
call the unpacked products active-symbol sequences rather than spatial planes.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `PREPARE_RESULT.json` | 3,600 | `d3e3a647463a6133ee4a2c0eaf84e72c76d93d46a7df769b9d2bdd139b6db42f` |
| `SCREEN_RESULT.json` | 84,407 | `8dffa811f01864eaf3c05083be60acf7740927fa4c8bde73a871910da076f976` |
| `CONFIRM_RESULT.json` | 11,151 | `ab4582a32181196035a31f8935f5b7619552563ac87a41b8a415499b4b7660af` |
| `MANIFEST.json` | 31,994 | `d8098367ef90c2634eb436b9f174c61a320c3c1b781285d0f9b25761c5679724` |

The manifest covers 128 retained files and every file's path, byte count, and SHA-256. The retained
store occupies 486 MiB. It includes four full source class planes, 32 screened active planes,
32 sparse sufficient-count tables, 32 bound receipts, four real payloads, four identical repeats,
four decoded active-symbol streams, per-stage checkpoints, generated C, and the compiled library.
All are evidence artifacts, so cleanup is certify-or-block: none was deleted or moved.

## Ledger receipts and typed routing

- `ddm_d3c_class_pyramid_peel_order` was registered at
  `2026-08-26T17:32:02.990693Z`, actor/owner `ddm_d3c`, session `ddm_d3c_20260826`.
- It moved to `in_progress` at `2026-08-26T17:32:03.126133Z` after recall and custody resolution.
- Completion receipt: pending serializer landing; the completion row will name the landing commit
  and green validation state.

`QUEUED-WITH-A-FIRE-ORDER`: the D3B successor owns reference-form confirmation of the rank-1 chain.
Its consumer store is `/Volumes/APDataStore/pact/ddm_d3c_class_pyramid/successor_reference/`. Fire
only when it can apply the D3B Amendment-1 reference form (HPAC/F26 plus D3A geometry/temporal and
an adaptive mixer) to the same source field. First produce an all-costs-in same-object subsystem row;
continue to independent receiver context regeneration and full five-class decode identity only if
that row has a credible path below 127,292 B. If it cannot clear that gate, fold the reduced-context
ranking as a screen-only negative and do not spend an archive or scorer run.

## GESTALT-DELTA

GESTALT-DELTA: The uncomputed ladder does have a stable best order, and keeping Lane as the finest
peel is strongly favored, but order supplies only a 5.741% screen span and the full ladder adds
40,985 B over the D3 single-peel screen. The live problem is therefore reference-form spatial
prediction of the Road/Lane interface, not searching more class permutations.

## Measured, not measured, and verdict boundary

Measured: all 24 reduced-context bounds, all rung denominators, the four retained real RC64
payloads, deterministic repeats, and exact active-symbol replay under source-derived contexts.

Not measured: HPAC/F26 or any D3B Amendment-1 reference form on the full ladder; receiver-generated
context keys; one-packet parse-back; independent exact five-class decode identity; packet/model
overhead; archive bytes; Seg/Pose distortion; exact contest score; CPU/CUDA parity.

Verdict scope: **FORMULATION-SCREEN NEGATIVE** for this Road-rooted radius-one KT/RC64 context. It
does not close the class-pyramid family, the D3B reference form, or any codec. The one real row is a
mechanism control, not promotion evidence.

Own-vehicle frontier: **GB1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**, archive
SHA `ba1f3830...`; unchanged by D3C.
