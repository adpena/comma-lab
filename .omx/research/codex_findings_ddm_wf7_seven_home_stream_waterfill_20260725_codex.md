# Codex findings — DDM WF7 seven-home stream waterfill

Date: 2026-07-25
Axis: `[macOS-CPU frozen-scorer advisory]`
Verdict: `STREAM_PRICE_DOMAIN_NONEMPTY_RATE_ONLY;NO_613_BOX_MEMBER`
Verdict scope: `INSTANCE x exact 134211-byte seeded C1 state x settled five-coder menu x exact CC3 composition and current E4 receiver endpoints`

## Answer to the delegated question

**The seven-home granularity is actionable for lossless rate reduction, but it
does not produce an actionable #613 box member at this endpoint.**

The exact seeded C1 state has five strictly improving physical-home rows. Their
joint receiver container is 132,435 B, down 1,776 B from 134,211 B, and restores
the state byte-for-byte. The rate-only action is
`delta_S = -0.0011825655007449763`.

That is a real finite price domain, so the per-cell 0/162 null table was partly
a granularity issue. It is not a distortion cure: lossless recoding leaves
`d_seg` and `d_pose` unchanged, and every measured receiver-closed endpoint
available to this arm remains far outside the #613 distortion box.

Full receipt:
`.omx/research/ddm_wf7_seven_home_stream_waterfill_20260725T203257Z/receipt.json`
SHA-256: `4ca7ba5075ff93dc1b42b8c469c5154fe32ca7c90c03885603c37ceca2ae0e94`

## CC3 was seated first, without double counting

CC3 remains the first finite stream-level falsifier:

- source/candidate: 139,538 B -> 136,116 B;
- measured delta: -3,422 B;
- measured mixed-versus-raw receiver output identity;
- 135/135 coder frames parse back;
- `delta_d_seg = delta_d_pose = 0`;
- measured endpoint: `(0.024731920030381944, 163.0492342914382, 136116)`.

The full -3,422 B cannot be added to WF7. Only -2,302 B comes from three
physical leaves nested inside v15; the remaining -1,120 B belongs to outer
PC1/WS1/v16/application wrappers. CC3 and WF7 attack the same describe pool on
different exact objects. They are alternative finite-price falsifiers, not
additive credits.

## The seven logical rows needed one physical correction

EV2/LP1's `lane_program_seed = 270 B` is a lawful accounting delta, not one
contiguous stream. Reconstructing the sealed seeded state
`3d5ab9786cc3d3eedd9a5fd1d878aea8186fbcf450ffcb781862db63ac2ca0cd`
shows where those bytes physically live:

| physical home | source B | seeded-state B | delta B |
|---|---:|---:|---:|
| manifest | 3,345 | 3,379 | +34 |
| lane member | 0 | 155 | +155 |
| central directory/EOCD | 383 | 464 | +81 |
| all other homes | 130,213 | 130,213 | 0 |
| total | 133,941 | 134,211 | +270 |

WF7 therefore races the seven non-overlapping physical homes. This preserves
LP1's accounting equality without pretending that its 270-byte logical delta
is directly decodable as one frame.

## Measured whole-home rows

Each row used the settled five-arm CC2/MS7 coder menu. Every arm parsed back to
the exact original home. The final container has one 21-byte counted directory.

| physical home | current B | winner | framed B | delta B |
|---|---:|---|---:|---:|
| manifest | 3,379 | G4 decoder context | 2,019 | -1,360 |
| predictor ZIP | 100,099 | Willems CTW | 99,821 | -278 |
| G1 worldsheet | 29,878 | raw | 29,878 | 0 |
| receiver profile | 85 | raw | 85 | 0 |
| solved template | 151 | Bellard mixture | 149 | -2 |
| lane program | 155 | Bellard mixture | 140 | -15 |
| central directory/EOCD | 464 | Bellard mixture | 322 | -142 |
| selected payload subtotal | 134,211 | — | 132,414 | -1,797 |
| counted WF7 directory | 0 | — | 21 | +21 |
| **candidate** | **134,211** | — | **132,435** | **-1,776** |

Candidate SSD artifact:
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_wf7_seven_home_stream_waterfill_20260725T203257Z/wf7_state.dwf7`
SHA-256: `4b0c686c8cf2976d8961593669636fa4f5f7f2f25fea58a4d5285c2df01ad615`

The receiver validates canonical varints, codec IDs, exact frame consumption,
each selected coder's canonical decode/re-encode, the restored ZIP, seven home
boundaries, and the sealed final state SHA.

## #613 box adjudication

The box is `d_seg <= 0.00116`, `d_pose <= 0.00161`,
`archive_bytes <= 200000`.

| row | d_seg | d_pose | counted bytes | box |
|---|---:|---:|---:|---|
| CC3 measured receiver-closed | 0.0247319200 | 163.0492343 | 136,116 | fail Seg + Pose |
| E4 measured receiver-closed | 0.02861482 | 147.49104309 | 344,203 | fail all three |
| WF7 state diagnostic | 0.027470296224 | 163.061327281443 | 132,435 state-container B | fail Seg + Pose |

The WF7 diagnostic is not presented as an E4 or contest packet triple. It
combines exact restored-state identity with the measured C1 state endpoint only
to show that distortion already rejects the member. The missing
state-container-to-E4 runtime binding remains explicit; no fake packet or score
claim was made.

`cheapest_measured_box_member = NULL`.

## Pointer and authority

- Effective competitive frontier: official leaderboard displayed `0.172`
  (official-best-aware correction consumed).
- `0.1910828242 [contest-CPU]` is a custody-specific local baseline only.
- Pointer moved: false.
- No score claim, promotion, dispatch, campaign fire, paid work, or heavy
  launch occurred.
- PF3/PF3b prices were not added.

## STORES CONSULTED

- `CLAUDE.md` and byte-identical `AGENTS.md`
- `PROGRAM.md`
- `docs/operating_manual_craft_handoff.md`
- Fable eureka memo §A1
- EV2 exact seven-home allocation receipt
- LP1 corrected C1 typed-home receipt
- CC2 five-coder race receipt
- CC3 mixed-coder receiver and 135-frame replay receipts
- C1 composed candidate ledger and #613 box
- MS4D direct metric completion receipt
- MS2R/MS2RP box-tolerance receipts
- E4 Brotli runtime and upstream harness receipts
- routing card §§4–6
- operator inbox through `2026-07-25T19:52:29Z`

## Round-1 self-review

PASS with one material correction applied before this memo: the first
accounting read treated the 270-byte lane seed as if it were a contiguous
frame. Reconstructing the exact state disproved that shape and produced the
34/155/81 physical reconciliation above.

Review checks:

1. cc3 -3,422 B is seated first and never added to WF7;
2. all seven home bytes plus the 21-byte directory reconcile exactly;
3. candidate decode restores the sealed 134,211-byte state;
4. null is used for the absent box member;
5. the WF7 state diagnostic is not labeled an E4/contest packet;
6. official `0.172` is the competitive reference; pointer remains unmoved;
7. negative verdict remains INSTANCE-scoped and leaves lossy homes, other
   endpoints, and the describe family open.

MAIN landing review is required, especially for the logical-to-physical
lane-seed correction, cc3 nonadditivity firewall, container parser, and the
decision not to promote the state diagnostic into a packet triple.
