# ddm_rb1 — the 42,382-byte bound has no incumbent-stream supplier

`date_utc: 2026-08-22` · `disposition: QUEUED-WITH-A-FIRE-ORDER` ·
`axis: [contest-CUDA T4 n600]` for the shipped row and distortion components;
`[macOS-CPU scorer-free exact byte/receiver parse-back]` for this arm ·
`score_claim: false` · `frontier_moved: false` · `[no-triality] [p0-ledger-ok]`

## Answer first

The exact DX2 archive is **180,368 B**. At its measured distortion,
`d_seg = 0.00020139` and `d_pose = 0.00000637`, the strict sub-0.12 ceiling is
**137,986 B**; the complete archive must therefore lose **42,382 B**.

The shipped archive has seven physical byte regions. The 113,777-byte RC64 token stream is
the only single region large enough to contain the whole cut, but size is not measured
headroom. On the current representation, every retained lossless coder win is already shipped:
FX5's token-context win contributes 70 B and DX2's carrier-coefficient win contributes 18 B.
The current-body admissible headroom vector is therefore **zero in every region**. The honest
waterfill allocates **0 B**, leaving an explicit **42,382 B residual** assigned to
`UNASSIGNED-NEW-REPRESENTATION`, not optimistically spread across incumbent streams.

This is not a mathematical proof that no future codec can save one more byte. It is the measured
floor of the tested, receiver-closed fixed-representation coder set. Older memoryless bounds are
diagnostics on other bodies, not transferable source-entropy floors. The campaign needs a new
representation whose complete archive is priced against 137,986 B at its own measured distortion.
NR1's task-cell quotient is the only named standing candidate with enough affected byte mass; it
has **0 measured bytes today**.

No scorer, render, trainer, Modal job, or paid action ran. No new payload was materialized or
discarded. The shipped archive and receiver bytes were read only.

## Exact bound

The authority receipt is
`/Volumes/APDataStore/pact/ddm_dx2/r7/t4_row_r1/MODAL_REMOTE_RESULT.json`; the exact archive is
`/Volumes/APDataStore/pact/ddm_dx2/r7/retained/candidate_dx2_cabac.zip`, SHA-256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.
The retained repeat has the same bytes and SHA-256.

```text
distortion = 100*0.00020139 + sqrt(10*0.00000637)
           = 0.028120227975693968
rate/byte  = 25/37,545,489
           = 6.658589531221714e-7 S/B
S(180,368) = 0.14821987563243377
S(137,986) = 0.11999944148120990   PASS
S(137,987) = 0.12000010734016303   FAIL
demand     = 180,368 - 137,986 = 42,382 B
```

Even impossible zero distortion does not save the incumbent representation: the strict rate-only
ceiling is 180,218 B, so DX2 still needs a 150 B representation cut when distortion is set to zero.

## Shipped physical inventory

I parsed the retained member `p` with the shipping receiver's
`RX1_MODEL_HEADER = struct.Struct("<4sBBBBHHH")`. The lengths below are physical bytes in the
exact archive, not decoded sizes or compression-factor projections. The member is ZIP-STORED, so
its 180,268 B plus 100 B of ZIP framing equals the complete 180,368 B archive with no remainder.

| physical region | shipped B | archive share | shipped coder / receiver action |
|---|---:|---:|---|
| ZIP framing | 100 | 0.0554% | one STORED member `p`; local header + central directory + EOCD |
| RX1 header | 14 | 0.0078% | fixed `RX1M` v1 length/flag grammar |
| HPAC probability model | 13,515 | 7.4930% | Brotli q9/lgwin16 outer stream; receiver restores canonical IHS1 |
| semantic renderer | 30,856 | 17.1072% | Brotli q9/lgwin16 outer stream; receiver restores the 36,130 B SM3R packet |
| frame-0 carrier | 22,010 | 12.2028% | Brotli q9/lgwin16 over the 22,008 B compact body; RR5 adaptive-arithmetic basis plus DX2 adaptive-context Rice/CABAC-prefix coefficients restore the canonical carrier |
| fixed residual table | 96 | 0.0532% | stored compact six-bit `RCF1` table body; the four-byte magic is receiver-restored |
| semantic token stream | 113,777 | 63.0805% | RC64 arithmetic stream under the shipped 19-member FX5 context law |
| **total** | **180,368** | **100.0000%** | exact; unaccounted bytes **0** |

The 22,008 B carrier-body figure in the DX2 receipt and the 22,010 B physical carrier stream are
different objects: the latter is the counted Brotli stream. Allocation uses 22,010 B.

## What the established floors do and do not mean

The PR130 four-section audit is useful precisely because it shows why an order-0 or explicit-model
bound cannot be promoted to a current-object information floor:

| PR130-lineage section | achieved B | diagnostic comparison B | achieved minus comparison | classification |
|---|---:|---:|---:|---|
| token stream | 114,860 | 114,852 explicit-model cross entropy | +8 | real coder/model gap on that old object only; not current source entropy |
| semantic model | 35,033 | 36,805 order-0 H0 | -1,772 | H0 is a memoryless coder-choice diagnostic; context/LZ legitimately beats it |
| carrier / pose model | 23,054 | 22,989 order-0 H0 | +65 | old-object coder diagnostic; representation and body changed afterward |
| HPAC model | 14,962 | 16,567 order-0 H0 | -1,605 | H0 is not a lower bound for a contextual coder |

Source: `.omx/research/ddm_pr130_reproduce_20260809/SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md`.
Those numbers cannot be subtracted from DX2's live streams. Doing so would mix bodies and, for two
sections, would call an achieved value below H0 impossible.

The closer-lineage `ddm_bp1_section_coding_axis_closed_20260818.md` independently raced real coders
on a 179,930 B descendant: semantic matched its shipped 34,243 B; carrier's best local alternative
was only 1 B smaller; the shipped HPAC beat the local recode by 39 B; joint and byte-plane variants
lost; token alternatives lost. Its available same-object total was 5 B, below admission. Later FX5
and DX2 then landed their distinct 70 B and 18 B wins on the current lineage. R012's 88 B ceiling is
therefore fully present in the exact archive inventoried here.

For the waterfill, “measured floor” means the smallest retained, real-coder, receiver-closed byte
count on the current fixed representation. It is an achieved floor within the tested coder set, not
a Shannon or Kolmogorov lower bound:

| region | live B | current tested-set floor B | distance / admissible headroom B | floor status |
|---|---:|---:|---:|---|
| ZIP framing | 100 | 100 | 0 | structural for the shipped one-member STORED ZIP grammar |
| RX1 header | 14 | 14 | 0 | structural for the shipped RX1 grammar |
| HPAC model | 13,515 | 13,515 | 0 | achieved coder-set floor; no transferable mathematical floor |
| semantic renderer | 30,856 | 30,856 | 0 | achieved current incumbent; MZ2's older-body +340 B recode result is negative evidence, not a transferred current floor |
| carrier | 22,010 | 22,010 | 0 | achieved coder-set floor after the shipped RR5 and DX2 riders |
| fixed residual | 96 | 96 | 0 | achieved incumbent representation; no measured smaller same-decode row |
| token stream | 113,777 | 113,777 | 0 | achieved coder-set floor after FX5; fixed-field coder axis exhausted |
| **total** | **180,368** | **180,368** | **0** | tested current-representation set only |

## Corrected-bound waterfill

An admissible supplier must be a real coded payload, receiver-decodable, composable with the exact
DX2 object, and priced by the complete score. Because the bound holds distortion fixed, a lossy row
does not supply bytes merely because its ZIP is smaller.

| allocation surface | measured admissible headroom B | allocated demand B | supplier verdict |
|---|---:|---:|---|
| ZIP + RX1 header | 0 | 0 | cannot supply: already at the shipped grammar floor; 114 B total mass |
| HPAC model | 0 | 0 | cannot supply: current-lineage coder races do not beat shipped bytes |
| semantic renderer | 0 | 0 | cannot supply: exact recodes lose; lossy edits are score-negative |
| carrier | 0 | 0 | cannot supply: 88 B campaign ceiling already includes DX2; rank/refit is distortion-dead |
| fixed residual | 0 | 0 | cannot supply: only 96 B live and no smaller measured same-decode row |
| token stream | 0 | 0 | cannot supply as the exact field: FX5 is shipped; real token-drop rows are score-negative |
| **measured incumbent-stream supply** | **0** | **0** | no positive supplier |
| **UNASSIGNED-NEW-REPRESENTATION residual** | — | **42,382** | requires a representation not yet measured |

Thus the allocation sum across shipped streams is **0 B**, and the honest residual is
**42,382 B**. A table that forces the stream allocations to sum to the demand would be a table of
hopes, not a measured waterfill.

The prior-law prediction is **not adjudicated**. The measured headroom vector is all zero, so it is
not roughly proportional to stream size and the stated falsifier did not fire. But there is also no
positive supplier with which to confirm concentration or the “largest stream is not the largest
supplier” clause. Rejected gross cuts are visibly non-proportional, which is qualitative support for
placement effects, not a bankable result.

## Standing candidates re-priced against their slot

The only measured allocation slot is the 42,382 B new-representation residual. The rows below show
why none of the standing candidates may book a piece of it today.

| candidate | affected current slot | measured row | complete price | allocation result / disposition |
|---|---|---|---|---|
| NR1 task-cell quotient | 42,382 B unassigned new-representation residual; if every other DX2 byte stayed fixed, tokens would have to fall from 113,777 B to **71,395 B** (37.2501%) | no complete candidate and no coded payload | unmeasured | **0 B booked; QUEUED-WITH-A-FIRE-ORDER.** Its actual acceptance test is complete archive `<=137,986 B` at DX2 distortion, or the recomputed strict ceiling at its own measured distortion. The 71,395 B token figure is a derived one-stream bar, not a prediction. |
| carrier rank/refit | 22,010 B physical carrier; it cannot contain the whole 42,382 B cut | on hv1, rank-4 gross cut 14,709 B; feasible rungs 913–1,847 B | the best realized trust-region refit still missed break-even by **35.5x**; the sphere-wide price misses by 1,498x–3,139x | **0 B booked; FOLDED at FAMILY scope for post-hoc rank/refit.** Gross archive cuts are not fixed-distortion supply. Only a from-scratch jointly trained carrier is outside that closure. |
| width distillation | 30,856 B physical semantic stream | WD4's trained width-64 descendant is a retained 166,459 B archive, **13,927 B** below its 180,386 B FX5 parent | `[macOS-CPU advisory]` n600: `d_seg=0.03182023`, `d_pose=13.43292999`, `S=14.8829`; catastrophic loss | **0 B booked; FOLDED for this warm width-64 slice.** Its gross cut is 32.86% of the demand, but the complete price has the wrong sign. It does not close every possible newly trained student. |
| token drop | 113,777 B physical token stream | FS2 real re-encode: 1,022 B; FS3 real re-encode: 664 B | FS2 net **+0.0055153 S**; FS3 net **+0.03579520 S** | **0 B booked; FOLDED at FORMULATION scope for the measured token-drop mechanisms.** The cuts are 2.41% and 1.57% of the demand before their score losses. |

Two tempting semantic credits are also excluded from the table. MZ2's exact dense/sparse/
row-dictionary/hybrid recodes are all **+340 B** and all 38/38 tensors are consumed. SF1's reachable
2,051 B post-hoc FiLM reduction is not composable with the 823 B alternative and prices at
**+0.062227 S**; its honest supply is 0 B.

## Adversarial assumption check

The shared assumption behind the exhausted campaign is that the receiver must reproduce the four
logical fields independently and exactly, so each physical stream can be optimized in isolation.
If that assumption is wrong, the allocation problem changes: semantic model, HPAC context, tokens,
and carrier may share a smaller sufficient statistic whose receiver reconstructs evaluator-equivalent
behavior rather than the incumbent fields. That is the plausible breakthrough seam and the reason
NR1 remains live. It is inherited-by-design, not empirically established; no complete quotient row
exists. The opposite failure mode is equally important: calling cross-stream jointness a saving
before real serialization would merely move uncounted payload between labels. Only complete archive
bytes and measured distortion can adjudicate it.

## Boundaries

- **Measured here:** exact archive identity and size; physical RX1/ZIP section lengths and shares;
  recomposed exact bound; arithmetic distances; read-only verification of retained coder and
  candidate receipts.
- **Recalled from retained measured receipts:** current-lineage coder closures, carrier rank/refit,
  WD4, FS2/FS3, MZ2, and SF1 prices. Their axes and vehicle scopes remain attached above.
- **Not measured:** any new quotient, entropy, distortion, scorer output, render, joint composition,
  or current-DX2 candidate beyond the shipped archive. Cross-body gross cuts were not transferred.
- **Negative scope:** zero current measured headroom is an INSTANCE verdict on DX2 within the tested
  fixed-representation coder set. It is not a FAMILY proof against every possible compressor or new
  representation.
- **Custody:** no payload was materialized by RB1. Existing DX2, WD4, carrier, and token-drop
  payloads remain in their retained stores. Nothing under the live JO run directory was read or
  touched; `upstream/` remained read-only.

## RECALL EVIDENCE

The recall searched `.omx/research/` memos and retained receipts by content for `42,382`,
`113,777`, `rate bound`, `memoryless`, `coder axis`, `task-cell quotient`, `carrier rank`,
`width distillation`, `token drop`, `RX1M`, `RC64`, `SM3R`, and `CABAC`; the canonical-equations
JSON for rate/archive/entropy equations; `CANONICAL_RESEARCH_INDEX*`; the `sub015_DAG_*` FEED
blocks; and canonical task-status/bridge/ledger surfaces for the same mechanisms.

Findings beyond the charter seeds changed the result:

- Direct parsing of the exact DX2 archive replaced ES1's design categories with seven physical
  regions whose bytes sum exactly. In particular, the counted carrier is 22,010 B, not its
  22,008 B decoded compact body, and the current semantic stream is 30,856 B.
- The PR130 section memo showed that two achieved coders beat their order-0 H0 diagnostics. That
  prevented treating memoryless comparisons as mathematical floors or subtracting them across
  vehicles.
- BP1 supplied a closer-lineage same-object coder race and R012/FX5/DX2 supplied the later
  composable closure. Together they changed the waterfill from a nominal section-size allocation
  to zero admissible incumbent supply.
- WD4's later n600 advisory gate changed width-64 from a 13,927 B gross candidate into a measured
  complete-score failure. FS2 and FS3 likewise changed token-drop projections into real re-encode
  losses. RA2/RA3 priced the carrier rank/refit family rather than merely measuring its byte ceiling.
- The canonical equations include the standard exact rate-per-byte law and warn that entropy-coded
  placement is not a uniform raw-byte surface; none supplies a current-DX2 headroom number or
  overrides the retained real-coder prices.
- I did not find a prior DX2 42,382 B per-physical-stream decomposition in the searched index, DAG,
  task-status, bridge, or Markdown corpus. ES1 derived the scalar bound and proposed a design
  envelope, but explicitly did not bank its category allocations.

## Landing status

The required serializer was invoked with the post-edit working-tree SHA-256, but its internal
`git add` failed before staging with `unable to create temporary file: Operation not permitted` and
`failed to insert into database`. This sandbox exposes `.git` read-only. The memo remains an
untracked working-tree artifact; the staged index is empty. No commit is claimed.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN/operator`; consumer store:
  `.omx/research/ddm_rb1_rate_bound_decomposition_20260822.md`; fire trigger: a workspace with Git
  index/object write permission is available; action: recompute the memo's post-edit SHA-256 and
  commit only this file through `tools/subagent_commit_serializer.py` with
  `[no-triality] [p0-ledger-ok]`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_nr1_taskcell_quotient_prebuild`; consumer store:
  `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/`; fire trigger: the governed r9
  endpoint is frozen and sealed, NR1 has a deterministic actual-coder receiver-closed candidate with
  all payloads retained, and MAIN grants the unique scorer lane; action: price the complete archive
  against 137,986 B at matched DX2 distortion or recompute the strict ceiling from the candidate's
  measured distortion.

## LIVE-HYPOTHESES

- **A task-cell quotient can remove the residual that exact-field coding cannot.** It is plausible
  because reproducing DX2's 113,777-byte token field exactly is stricter than reproducing its frozen
  evaluator cells, and that one field has enough mass to contain 42,382 B. It remains wholly
  untested as a complete coded archive and therefore carries zero credit.

## DEAD-ENDS

- **Another lossless recode of the fixed DX2 representation:** closed at INSTANCE scope within the
  tested coder set because the composable 88 B ceiling is already shipped.
- **Using PR130 memoryless bounds as DX2 floors:** closed as a cross-object inference because H0 is
  a coder-choice diagnostic and is already beaten by contextual coders in two sections.
- **MZ2 exact semantic recoding as a supplier:** closed on its measured instance because all 38/38
  tensors are consumed and every tested exact form adds 340 B.
- **SF1 post-hoc semantic pruning as a supplier:** closed at FAMILY scope for the measured post-hoc
  edits because the reachable 2,051 B row costs +0.062227 S and the larger sum is undecodable.
- **Post-hoc carrier rank/refit:** closed at FAMILY scope because its best realized price misses
  break-even by 35.5x despite a byte ceiling that can clear an older bar.
- **WD4 warm width-64 and FS2/FS3 token drops:** closed only on their named measured scopes; their
  retained archives are smaller, but their complete score prices have the wrong sign.

Own-vehicle frontier remains **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; pointer unmoved.
