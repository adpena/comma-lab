# D3B lossless Lane factorization — receiver-closed n600 verdict

## Result

**Verdict: REFUSED / formulation closed.** The best exact factorization is the shipped HPAC
Road/Lane conditional with a 64,208 B RC64 body and a 64,276 B retained packet. Fully counted with
the verified D3 four-class stream, model, and factor framing, the token subsystem is **127,499 B**:
**207 B above** the 127,292 B joint-token bar and **42,435 B above** the 85,064 B demand-closing bar.
Its actual research factor archive is 180,575 B, 360 B larger than GB1. All nine candidates decode
the exact source field, so the failure is rate-only. No seal, scorer run, public fire order, or Modal
dispatch is admissible.

Axis for every row below: **[macOS-CPU advisory / scorer-free exact rate and receiver measurement,
n600]**. These are real RC64 payload bytes, not entropy estimates or projected archive sizes.

| Context design | Lane RC64 body | Counted Lane packet | Token subsystem | vs 127,292 | vs 85,064 | Actual factor archive | vs GB1 archive | Identity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hpac_conditional` | 64,208 B | 64,276 B | **127,499 B** | **+207 B** | +42,435 B | 180,575 B | +360 B | PASS |
| `reference_d3a_q8_mixer_r2` | 64,508 B | 92,024 B | 155,247 B | +27,955 B | +70,183 B | 208,323 B | +28,108 B | PASS |
| `reference_d3a_q1_mixer_r2` | **63,476 B** | 106,584 B | 169,807 B | +42,515 B | +84,743 B | 222,883 B | +42,668 B | PASS |
| `field_geometry_temporal_r2` | 358,520 B | 358,588 B | 421,811 B | +294,519 B | +336,747 B | 474,887 B | +294,672 B | PASS |
| `field_geometry_temporal_r1` | 367,808 B | 367,876 B | 431,099 B | +303,807 B | +346,035 B | 484,175 B | +303,960 B | PASS |
| `field_geometry_r2` | 471,928 B | 471,996 B | 535,219 B | +407,927 B | +450,155 B | 588,295 B | +408,080 B | PASS |
| `field_geometry_r1` | 477,264 B | 477,332 B | 540,555 B | +413,263 B | +455,491 B | 593,631 B | +413,416 B | PASS |
| `field_r2` | 529,184 B | 529,252 B | 592,475 B | +465,183 B | +507,411 B | 645,551 B | +465,336 B | PASS |
| `field_r1` | 529,732 B | 529,800 B | 593,023 B | +465,731 B | +507,959 B | 646,099 B | +465,884 B | PASS |

The corresponding rate-only changes relative to GB1 would be +0.000239709223124 S for the base
conditional, +0.018715963454358 S for q8, and +0.028410869811817 S for q1. They are arithmetic
consequences of archive bytes under exact field identity, not scored or promotable rows.

## Verdict scope and prior-law falsification

`[verdict_scope: FORMULATION -- D3 four-class quotient plus lossless conditional Lane; shipped HPAC
base; q8/q1 counted D3A AA-SDF geometry; r2 decoded-whole-field, cross-frame, and causal Lane
context; online fixed-point log-odds mixing; n600]`.

The charter's falsifiable prediction was a conditional Lane plane plus counted parameters at no more
than 40,000 B. It failed on every verdict-bearing member:

- the unaugmented shipped conditional packet is 64,276 B;
- q8's mixer makes the RC64 body 300 B worse and charges a 27,440 B chart;
- q1 proves that better analytic geometry carries real information—the RC64 body is 732 B smaller
  than the shipped conditional—but its 43,032 B chart overwhelms that gain;
- the complete q8 and q1 token subsystems miss the bar by 27,955 B and 42,515 B respectively.

This closes the chartered factorization at FORMULATION scope. It is not a claim that no possible
lossless conditional coder exists. A materially different generic predictor or integrated packet
format may reopen only through the explicit byte trigger below.

## Optimal form

The reference rows are not the crude-table screens. They compose the shipped HPAC/F26 Road/Lane
conditional with D3A analytic lane-band distance, threshold, and integer angle features; decoded
whole-quotient context including both adjacent frames; already-decoded previous-frame and current
causal Lane neighbors; and an online integer log-odds mixer. The mixer starts exactly at HPAC weight
zero and must earn every change causally. No learned table is transmitted.

The scoped choices were derived or raced:

- radius 1 and 2 were raced; r2 won every matched generic pair and was used by the reference rows;
- 18 context bits give about 107 support observations per bucket before collision at the measured
  28,097,983-site denominator, keeping the fixed state bounded while allowing specialization;
- eight dyadic coverage/distance bins and eight integer direction codes preserve the AA-SDF and
  tangent-sign structure without platform-dependent `atan2`;
- D3A q8 and q1 quantization endpoints were both raced. q1 is the better raw predictor but the worse
  counted packet, proving the chart-fidelity/rate exchange rather than presuming it;
- the fixed-point learning shift is raced against the exact zero-weight HPAC member in the same pass;
  q8 loses raw bytes while q1 wins 732 raw bytes, so the mixer can both reject and admit signal.

The generic field-only and square-geometry rows are retained screens, not family-closing evidence.
Their absolute empirical-probability mixer overwrites a strong HPAC prior and is dominated.

## Decode identity and HC1 binary-question decomposition

The independent receiver reopened each candidate archive, decoded the 49,696 B four-class stream,
parsed its Lane packet, decoded the binary Lane plane, and reconstructed the canonical five-class
field. The quotient receiver produced 117,964,800 B at SHA-256
`deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07`. Every one of the nine
reconstructed fields produced 117,964,800 B at the source SHA-256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. Every decoded Lane
bitplane matched SHA-256 `6ca82a7883411d0eb27addac7dcf662e84d2f9cc66404c299da2e15761c0e0cf`.

The q8 receiver independently recovered 600 pairs and 159,384 LBND2 bytes from its 27,440 B carrier.
The q1 receiver independently recovered 600 pairs and 159,386 LBND2 bytes from its 43,032 B carrier.
Neither receiver read the retained encoder-side coverage arrays.

Measured HC1 denominators:

| Question | Right | Wrong | Denominator | Right fraction |
|---|---:|---:|---:|---:|
| joint HPAC argmax equals five-class symbol | 117,737,191 | 227,609 | 117,964,800 | 0.998070534600 |
| pooled quotient argmax equals four-class symbol | 117,867,252 | 97,548 | 117,964,800 | 0.999173075358 |
| conditional Road/Lane argmax equals Lane bit on quotient-Road support | 27,967,568 | 130,415 | 28,097,983 | 0.995358563638 |

There are 691,095 Lane ones on 28,097,983 quotient-Road sites: 2.459589359% Lane within support,
and support is 23.818955315% of all 117,964,800 positions. The quotient and conditional wrong counts
sum to 227,963, 354 more than the joint wrong count; these are marginal binary questions, not a strict
partition of joint errors.

## Rule-118 accounting

Counted bytes for every row are exactly:

`49,696 four-class RC64 + 13,515 four-class model + 12 factor framing + Lane packet`.

Each Lane packet includes a 68 B self-checking header. q8 additionally includes an 8 B reference
prefix plus the exact 27,440 B video-derived D3A carrier; q1 includes the same prefix plus its exact
43,032 B carrier. The RC64 body follows those counted bytes. Generic field hashing, integer AA-SDF
feature extraction, causal context construction, the fixed-point logit table, and the arithmetic
receiver are code and rule-118-free. Encoder-side q8/q1 coverage arrays are uncounted conveniences,
not receiver inputs; successful independent regeneration and arithmetic decode prove that boundary.

No GT mask, scorer weights, per-pixel table, or donor coverage field is in code. No fitted context
table is omitted from accounting. Actual zip sizes close exactly as `token subsystem + 53,076 B` for
all nine archives.

## RECALL EVIDENCE

Sources and queries searched before construction:

- content search over `.omx/research/`, receipts, designs, specs, and task status for
  `Lane|Road|lossless|factor|conditional|HPAC|reorder|context|Wyner|decoder side information`;
- `.venv/bin/python tools/list_canonical_equations.py --json`, then the Wyner-Ziv and conditional
  coding entries touching decoder-side information;
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design docs, and canonical
  ledger rows for the same terms;
- the charter seeds at D3, D3A, LM1, HC1, NO1, TO2/AD2, and GB1, plus their retained stores and real
  receiver/coder implementations.

Beyond the charter seeds, recall found the canonical Wyner-Ziv decoder-side-information equation and
the TO2/AD2 law that reordering only counts when a real coder and exact inverse win bytes. That changed
the design from a rearrangement probe to whole-quotient decoder conditioning with complete packet
parse-back. Recall also found D3's exact Lane representations—195,582 B Brotli-q11 mask, 273,464 B
lossless XOR, and 163,304 B pixel-major—so none was rerun as a supposedly new price. Finally, D3A's
retained q8–q1 chart/coverage endpoints changed the reference plan from generic coordinate context to
a counted q8/q1 quantization race. No cheaper n600 receiver-closed Lane sufficient statistic was found
in the searched corpus.

## Payload custody and reproducibility

Primary store: `/Volumes/APDataStore/pact/ddm_d3b_lossless_lane_factorization/`.

- `RESULT.json`: 42,025 B, SHA-256
  `ce901a9ce63fcaa01ae16fff28c19f8779f14ce27b77cc2c48d852e810a5e396`.
- `ENCODE_RESULT.json`: 22,011 B, SHA-256
  `d0381f64b3db2f98a025bd5afd49017f6deb6a9abcfa27df85c5039e346daa55`.
- `DECODE_RESULT.json`: 42,025 B, same SHA as `RESULT.json`.
- all nine primary and deterministic-repeat Lane packets are under `retained/lane_streams/`; all nine
  parsed research archives are under `retained/candidates/`; all nine reconstructed fields and Lane
  bitplanes are under `retained/decode/`.
- checkpoint `encode_latest.npz`: 49,709,262 B, SHA-256
  `1ecb47a687645dd569bde19e7493a7afc5cffc8b28b29a93b0d20e3be2688a39`.
- checkpoint `quotient_decode_latest.npz`: 10,540,671 B, SHA-256
  `72ecf9d0f8578d10e8ae1848ba351f71f2bcfbcea435478b00e4310a02b71d4e`.
- checkpoint `lane_decode_latest.npz`: 30,203,421 B, SHA-256
  `472acb13ef43b3b7354b5020e56a1731eb6f9839648b3726a7a9c53e00964275`.
- pre-reference v1 payloads and receipts are retained under `measurement_v1/`, manifest SHA
  `b0bec239a3478a5775faef40a8710d788b306fe292b6575dbeddd4d0cfc322cf`; q8-complete v2 is under
  `measurement_v2/`, manifest SHA
  `8562578cf8624063f213f284a3c816ac7dcd96a1a23a772de547c14d5848657a`.

The first encode attempt reached frame 20 but failed before a verdict artifact because the checkpoint
parent directory did not yet exist. That was a resumability defect, not a research result. All three
checkpoint writers were corrected to create their parent atomically; the n600 encode and both decode
stages then completed with checkpoints every 20 frames. A completed `--stage all --resume` replay
validated every retained fact and returned in under a second without re-encoding.

## Ledger receipts

- `ddm_no1_row3_alphabet_merge::d3b_lossless_lane_factorization`: registered, advanced, and completed
  by actor `ddm_d3b`, session `ddm_d3b_20260826`, at 2026-08-26T18:17:50Z; test status green.
- `ddm_no1_row3_alphabet_merge::d3b_fr0_zero_sideinfo_reopen`: pending under MAIN with an explicit
  byte-derived fire trigger and consumer store; no job has fired.

## GESTALT-DELTA

Before D3B, the working model was that Road embedding plus non-causal quotient geometry would make the
Lane plane at most 40 KB. After real coding, the base conditional is 64,208 B; generic absolute
context mixing destroys calibration; q8 adds no raw information; and q1 contributes only 732 B of
raw-stream value for a 43,032 B chart. The bottleneck is not whether Lane geometry is predictable—it
is—but whether decoder side information can be supplied generically or in at most a few hundred
counted bytes.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — disposition: pending task
  `ddm_no1_row3_alphabet_merge::d3b_fr0_zero_sideinfo_reopen`; owner: MAIN; consumer store:
  `/Volumes/APDataStore/pact/ddm_d3b_lossless_lane_factorization/followons/fr0/`; fire trigger: derive
  a generic video-independent predictor or integrated framing plan that proves RC64 body at most
  64,000 B with the current 80 B framing, or at most 64,080 B with zero framing, and zero counted side
  information. Only then run one retained n600 encode/decode-identity row; otherwise do not launch.

## LIVE-HYPOTHESES

- A zero-side-information calibration of the exact HPAC Road/Lane question may still save the needed
  208 B because the incumbent packet misses narrowly and the tested hashed model failed by replacing,
  not gently calibrating, the strong prior.
- q1's measured 732 B raw-stream gain proves analytic geometry contains coder-relevant information.
  If a generic quotient-derived generator or a video-derived sufficient statistic can reproduce that
  signal in at most 516 counted bytes, the q1 exchange would cross the strict subsystem bar.
- Integrating factor framing into an existing public member can remove at most 80 B; it cannot win
  alone, but paired with at least 128 B of zero-side-information coder gain it could clear the bar.

## DEAD-ENDS

- The q8 counted D3A reference mixer is closed: its RC64 body is 300 B worse than HPAC and its chart
  adds 27,440 B.
- The q1 counted D3A reference mixer is closed: its real 732 B body gain cannot amortize a 43,032 B
  chart; total misses by 42,515 B.
- Strength-64 absolute empirical mixing over field-only, square-geometry, and temporal hashes is
  closed as a screen family: every row is 294,519–465,731 B above the bar because it overwrites HPAC.
- Exact Lane mask Brotli, lossless XOR, and pixel-major streams are already closed by retained D3
  measurements at 195,582 B, 273,464 B, and 163,304 B; no successor should reprice them as new work.
- Framing-only consolidation is closed: removing all 80 counted framing bytes leaves 127,419 B, still
  127 B above the strict bar.
- No scorer, seal, or Modal path exists for these candidates because every receiver-closed archive is
  larger than GB1; firing one would only spend authority to confirm a pure rate loss.

Own-vehicle frontier unchanged: **S = 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]** (GB1;
D3B produced no lower exact score).
