# DDM QX2 events-section redesign — conditional byte clearance, QX1 still OVER

## Verdict

`OVER` is the terminal QX1 verdict, scoped to this formulation. A real,
address-free conditional event section does clear the byte envelope, but it is
conditioned on a 117,964,800-byte decoded C1 baseline that no QX1 receiver
currently produces. That baseline is not in the 136,553-byte archive. Calling
the archive a complete QX1 candidate would therefore be a fake implementation.

All measurements below are `[scorer-free exact rate and receiver-conditional
parse-back measurement]`, full `n600`. No scorer, Metal, Modal, contest eval,
or `upstream/` mutation occurred.

| Object | Section payload | Exact archive | Delta vs strict `<137,986 B` | Disposition |
|---|---:|---:|---:|---|
| QX1 core, section absent | — | 113,844 B | −24,141 B at the largest legal archive | incomplete control |
| QX1 explicit-address event control | 39,815 B zlib9 | 153,707 B | +15,722 B | inadmissible negative |
| boundary bitmap, radius 0 | 27,848 B lzma9e | 141,740 B | +3,755 B | `OVER` |
| boundary bitmap, radius 1 | 30,192 B lzma9e | 144,084 B | +6,099 B | `OVER` |
| boundary bitmap, radius 2 | 31,804 B lzma9e | 145,696 B | +7,711 B | `OVER` |
| boundary bitmap, radius 4 | 34,028 B lzma9e | 147,920 B | +9,935 B | `OVER` |
| decoded-state distance ranks | 30,216 B lzma9e | 144,108 B | +6,123 B | `OVER` |
| boundary-transition enumerative subset | **22,661 B Brotli q11** | **136,553 B** | **−1,432 B** | conditional byte gate cleared; receiver open |

The exact maximum section payload is 24,093 B: `137,985 - 113,844 - 48`,
where 48 B is the QXE section header. The winning section adds 22,709 B
including that header. Its archive and deterministic repeat are byte-identical,
SHA-256 `da63908dd0babb426f72ae91add5b6db349aa0a670705e441cd4ab0aeda4923d`.

## Stage 0 — actual event population

The denominator is 600 pairs, 117,964,800 raster sites, and all 17,926 retained
S2 events. Every pair has events: minimum 11, median 28, mean 29.8767, maximum
152. The source object is the QX1-retained packet
`/Volumes/VertigoDataTier/pact/ddm_qx1/retained/sections/08_events_exceptions_explicit_address_control/raw.bin`,
39,836 B, SHA-256
`df4c0534537a9919681509a0b44a392d7d4b46c812d7570c534e6b823adae7fc`.

Measured structure:

- 17,691/17,926 events, 98.6891%, lie exactly on the boundary of the
  event-implied decoded C1 baseline. Only 235 events are off that boundary.
- The radius-0 boundary candidate set has 2,551,391 sites. This is the useful
  receiver-conditioned alphabet; transmitting an exact subset of it is much
  cheaper than transmitting raster addresses.
- Exact same-site/same-transition persistence from the immediately preceding
  pair covers 74/17,926 events, 0.4128%. A bounded shift search over `[-8,8]^2`
  reaches only 677/17,926, 3.7766%.
- There are 16,613 horizontal runs, 1.0790 events/run. Transition-separated
  8-connected components number 16,369, with 15,397 singletons and 1.0951
  events/component. The event object is therefore singleton-dominated, not a
  small set of long curves or scanline runs.
- Major transitions are `1→0` 2,791, `0→4` 2,551, `0→2` 2,542, `0→1` 2,520,
  `2→0` 2,267, `4→0` 2,143, `3→2` 1,071, and `2→3` 798. All 18 observed
  directed transitions are preserved exactly.

The first attempt failed closed on one known S2/GT-cache mismatch at
`(pair,row,col)=(11,286,399)`: cached class 0, retained S2 target class 4,
baseline class 0. The S2 source receipt already reports exactly one cache-target
mismatch. QX2 treats the retained S2 event as authoritative, pins the original
GT-cache semantic SHA-256
`f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557`,
and pins the corrected S2 target state as
`36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68`.
The failed 2,162,688-byte partial payload was retained and folded, not deleted.

## Stage 1 — address-free forms and negative crosswalk

| Form | Mechanism | Build disposition | Prior negative avoided or folded |
|---|---|---|---|
| deterministic temporal generator + counted residual | Predict sites from persistence or an ego-shift, then conditionally code misses. | Rejected before build on the full population: 0.4128% exact persistence and 3.7766% best bounded shift cannot remove the required 39,863 B section delta. | Folds LC3's weak predictor result without transferring its Lane-only number; also avoids pretending MA2 dash phase is solved. |
| decoded-state boundary subset | The receiver derives C1 boundary candidates; per pair and baseline class, QX2 enumeratively codes exact target-transition subsets. Only the 235 off-boundary events use decoded-state distance ranks. | Built and exact. The raw representation is 26,666 B; Brotli q11 is 22,661 B. | Avoids LTG1/pincer's explicit `(pair,row,col)` stream and QBW2's plane serialization. It still carries counted conditional subset identity, so it is not “generation alone.” |
| decoded-state distance ranks | Order every raster site by distance to the decoded boundary and transmit event-rank gaps plus transitions. | Built and exact; 30,216 B, `OVER`. | Avoids literal raster coordinates but folds the exact-address pincer into a rank stream; the entropy remains too high. |
| implicit continuous threshold field | Fit curves/components or scanline intervals and let the receiver threshold them. | Rejected before build: 15,397/16,369 components are singletons and horizontal runs average 1.079 events. | Folds MA2 smooth-curve phase failure, LTG1 topology-event cost, and GF1's 5.09× form-and-fit result for this event object. |

No scope or mechanism reduction was used: all 17,926 events are reconstructed,
all candidates were real-coded, and all raw/coded/repeat/packet/archive bytes
were retained. The four radii are tuning configurations of one boundary form;
the two priced form families are the optimized boundary form and distance-rank
conditional form.

## Stage 2 — exact receiver and coder result

The winning receiver regenerates the radius-0 candidate set from the decoded
baseline. For each pair and each of the 20 possible directed class transitions,
it decodes a combinatorial subset rank against the remaining candidates of the
known baseline class. This jointly identifies site subset and target class
without storing raster addresses. The raw section contains:

- 17,691 boundary events in 151,673 exact enumerative rank bits, stored in
  18,960 bytes;
- a 6,385-byte transition-count stream;
- 235 far residual events in a 1,253-byte decoded-distance-rank stream; and
- 68 bytes of self-identifying header and length fields.

The 26,666-byte raw section is SHA-256
`5aa980451fb4c7eadd28bb98186e4e5fd220034f669b8bfc31934cf3dbd4c13c`.
Real-coder results are Brotli q11 22,661 B, lzma9e 23,208 B, and zlib9
24,213 B. Each coder repeat is byte-identical and each decode reproduces the
source `PartitionEvent` tuple exactly, including pair, row, column, target
class, and baseline class for 17,926/17,926 events.

This measurement establishes a conditional representation result, not a QX1
candidate. The conditioning baseline is retained at
`/Volumes/APDataStore/pact/ddm_qx2/retained/baseline/c1_baseline_labels.u8`,
117,964,800 B, SHA-256
`02a2a3f572d6e0abf039d812330962ae8b1a44f02701661136482759e33ccf34`.
It was derived using the GT cache and retained S2 events. QX1 has no receiver
that produces this state, so these bytes cannot be treated as free, omitted,
or hidden in code.

## Stage 3 — close or iterate

The byte subproblem is solved conditionally, but the complete QX1 envelope is
not. `ENVELOPE-CLEARED` is withheld because the required decoded state is not
under QX1 receiver custody and the QX1 pose cap remains unmeasured. The measured
formulation verdict is therefore `OVER`, with one narrowly defined receiver
binding action queued below. No scorer action is authorized by this receipt.

## RECALL EVIDENCE

Recall covered `.omx/research/` memos and receipts, the canonical-equations
registry, `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` feed surface, design
specifications, task/status/queue surfaces, source, and both SSD custody roots.
Queries included `QX1|events_exceptions|17926|39836`, `event|boundary|worldsheet`,
`argmax_cell_identity|known-site`, `predictor|residual|conditional`,
`Wseg|target table|candidate admissible`, `QBW|QBMIX|LC3|pincer`, and
`receiver_projection`. The equation command was
`.venv/bin/python tools/list_canonical_equations.py --json`.

Findings beyond the charter seeds changed the plan:

1. `/Volumes/VertigoDataTier/pact/evidence/s2_compose_20260721/partition_seed/receipt.json`
   records one cache-target mismatch. That converted the initial pair-11
   failure into an exact, source-authorized correction instead of silently
   substituting the GT cache.
2. `argmax_cell_identity_ideal_bytes_v1` gives a 2,724.8733-byte ideal class
   identity floor only when sites are already known; it excludes site grammar,
   candidate-set transport, headers, receiver, and realization. QX2 therefore
   did not use it as a closing price and instead built the finite subset coder.
3. `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.md`
   and `.omx/research/codex_findings_ddm_v13_worldsheet_event_predictor_20260722_codex.md`
   show that prior worldsheet/curve abstractions did not transfer automatically
   through the receiver; V13's exact 29,810-byte Movable grammar still failed
   its receiver-visible projection. That supported rejecting a smooth global
   field after QX2's singleton census rather than repricing the same curve path.
4. The C0B/PBR and premise-falsification receipts separate exact target tables
   from candidate-admissible state and warn that predictor baselines are
   vehicle-specific. That forced the 117,964,800-byte C1 state to remain an
   explicit external conditioning debt and is why conditional byte clearance
   did not become `ENVELOPE-CLEARED`.

Queue/status search found QX1 in its nx1/QX1 handoff and live arm rows, but did
not find a QX1-native receiver producing the pinned C1 baseline in those
searched scopes.

## Custody and reproducibility

- Result: `/Volumes/APDataStore/pact/ddm_qx2/RESULT.json`, 32,628 B, SHA-256
  `b3a63070260ca4d8d6ea23ec7395bb3156b2cbdae91c1a27bca2e0d82b63e234`.
- Run manifest: `/Volumes/APDataStore/pact/ddm_qx2/RUN_MANIFEST.json`, SHA-256
  `fca260fc1f10257ebd12012f488fead8f9e9ed31ebeb2f6ce7c5aa218dfe0aa7`.
- Runner: `experiments/ddm_qx2_events_section_redesign.py`, SHA-256
  `88457037f5cbc272b494306a1613f8c6e2abe3499fdf83164274e3db76b1311c`.
- Command: `.venv/bin/python experiments/ddm_qx2_events_section_redesign.py
  --resume-from /Volumes/APDataStore/pact/ddm_qx2`.
- Failure receipt:
  `/Volumes/APDataStore/pact/ddm_qx2/checkpoints/FAILED_ATTEMPT_20260831_PAIR11.json`.
- All raw forms, all three real-coder outputs, deterministic repeats, QXE
  packets, ZIP archives, the decoded baseline, and the failed partial payload
  remain retained. No cleanup fired.

**Frontier line:** canonical pointer remains **S = 0.14797617125559104 @
180,002 B `[contest-CUDA T4 n600]`**, AFR1 archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
QX2 moved no score or pointer.

## NEXT_IF_RESUMED

- **QUEUED-WITH-FIRE-ORDER** — owner: MAIN-assigned QX1 receiver-binding arm;
  consumer store: `/Volumes/APDataStore/pact/ddm_qx2/RESULT.json`; fire trigger:
  a deterministic, counted QX1 core receiver produces the exact baseline SHA-256
  `02a2a3f572d6e0abf039d812330962ae8b1a44f02701661136482759e33ccf34`
  without reading GT or S2 target tables. Bind the retained QXC1 decoder to that
  state, rebuild the exact archive, and issue an implementation charter to MAIN
  only if parse-back remains exact and the complete archive stays `<137,986 B`.

## LIVE-HYPOTHESES

- A QX1-native continuous latent may produce the same radius-0 boundary
  candidate field. This remains plausible because 98.6891% of the retained
  events lie on that field and the conditional event payload already has 1,432
  bytes of archive headroom; it is untested because QX1 has no receiver.
- A receiver-generated baseline need not match every C1 label if it preserves
  a deterministic candidate ordering and the event subset is re-ranked against
  it. This is plausible because QXC1 depends on candidate identity and order,
  not RGB, but any changed field must be re-coded and re-priced from scratch.

## DEAD-ENDS

- Temporal persistence/ego-shift generation is closed for this retained event
  object: exact persistence is 0.4128% and the best bounded global shift is
  3.7766%.
- Smooth curves, scanline intervals, and component grammars are closed for this
  retained event object at this form: 15,397/16,369 transition-separated
  components are singletons and runs average 1.079 events.
- Boundary dilation beyond radius 0 is closed for this coder: radii 1, 2, and 4
  monotonically worsen the exact archives to 144,084, 145,696, and 147,920 B.
- Dense boundary occupancy and decoded-distance ranks are closed against the
  QX1 gate: their best exact archives are 141,740 B and 144,108 B.
- Do not promote or score the 136,553-byte conditional archive. Its required
  117,964,800-byte GT/S2-derived baseline is external, no QX1 receiver exists,
  no QX1 pose cap was measured, and no score or pointer moved.
