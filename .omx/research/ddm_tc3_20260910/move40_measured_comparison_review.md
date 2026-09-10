# TC3 move40 measured comparison audit

Independent bounded audit of retained full-population results and actual
payloads. Axis: `[exact serialized bytes, scorer-free macOS-CPU, n600]`.
No encode, optimizer, native build, receiver, scorer or long job was launched;
no source, canonical state or index was changed. This review does not certify
the pending literal-shell public proof or create a measured contest score.

All current results use **117,964,800 positions = 600 × 384 × 512**, archive
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`
at 180,233 bytes and field
`b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5`.
Store:
`/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/`.

## Current result

| Move40 quantity | A: TC2 map, five new fitted coefficients | B: run tracks, joint 40-coefficient fit |
| --- | ---: | ---: |
| Full continuous codelength saving, bits/8 | 83.426814 | 48.946654 |
| Full integer-probability codelength saving, bits/8 | 83.309006 | 41.409311 |
| Raw RC64 bytes | 119,534 | 119,576 |
| Raw saving against source 119,618 bytes | 84 | 42 |
| Counted rider bytes | 119,579 | 119,621 |
| Rider saving against source 119,658 bytes | **79** | **37** |
| Smallest actual archive bytes | **180,154** | **180,196** |
| Archive saving against source 180,233 bytes | **79** | **37** |

The predeclared menu is exactly 2 variants × 10 rider representations × 4 ZIP
settings = 80 samples. Both winners use the plain rider in a stored ZIP.
B is 42 archive bytes worse than A, so it fails the replacement condition
requiring B to beat A by at least 300 bytes. A clears the 30-byte admission bar
by 49 bytes. The one staged candidate is A, archive SHA
`299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936`.

The recorded score projection recomputes to
`0.13763861019288715 - 79 × 25 / 37,545,489 = 0.13758600733559048`,
delta `-0.00005260285729666303`. It remains conditional on public output
identity and MAIN's later exact contest evaluation. No new score was measured
by this audit. The live measured anchor in the captured pointer is
S 0.13763861019288715 at 180,233 bytes, contest-CUDA T4 n600.

## Historical scale comparisons

The mandated TC2 denominators are historical move37 results: causal continuous
82.771457 bytes, causal real 78 bytes, and the noncausal GT-distance reference
5,480.397216 ideal bytes. They describe another field. These ratios compare
scale; they do not measure current-field oracle headroom or a matched treatment
effect. No GT-distance oracle was rerun on move40.

| Move40 numerator / historical denominator | A | B |
| --- | ---: | ---: |
| Continuous / TC2 continuous 82.771457 B | 100.791767% | 59.134702% |
| Real archive saving / TC2 real 78 B | 101.282051% | 47.435897% |
| Continuous / GT-reference ideal 5,480.397216 B | 1.522277% | 0.893122% |
| Integer ideal / GT-reference ideal 5,480.397216 B | 1.520127% | 0.755590% |
| Real archive saving / GT-reference ideal 5,480.397216 B | 1.441501% | 0.675134% |

The last row explicitly mixes physical bytes with a historical ideal-byte
denominator and is only a normalized scale comparison. A's 79-byte result
versus historical TC2's 78 bytes cannot be attributed to an improved predictor.

## Fit denominators and numerical gap

An independent count of both stored keep masks across all 600 prepared frames
matches the fitting receipts: A retains 18,214,360 positions (15.440504% of the
full population), and B retains 15,383,355 (13.040632%). These are selected by
base uncertainty or base prediction error, not contiguous prefixes. The final
continuous and deployed integer paths are evaluated on all 117,964,800
positions during the retained encode.

A fits five added coefficients while retaining the source's 35. B fits all 40
against fixed pre-TC1 frequencies and eight fixed features per class bank. The
complete counted rider includes 40 int8 coefficients for either variant; the
source includes 35. No per-frame fitted table is introduced by these recipes.

Both variants have four saved optimization iterations with decreasing losses.
Recomputing the final supporting-tangent gap from each saved gradient and
coefficient matrix matches its RESULT: A `0.000005032247` ideal bytes;
B `0.000039724340` ideal bytes. The retained-subset continuous improvements
are 79.062924 and 48.783257 bytes respectively. These are numerical convergence
facts for the selected subsets and fixed bounded coefficients. They do not
certify full-population coefficient optimality or the best int8 matrix.

The existing optimistic calculation that grants zero candidate loss on all
omitted positions yields full continuous bounds of 162.274327 bytes for A and
171.301370 for B:
`source_full_bits/8 - retained_loss_nats/(8*ln(2)) + retained_gap_bytes`.
This numerical fixed-feature calculation has the same exclusions as the
registered supporting-tangent argument; B also extends the coefficient family
beyond the original five-added-coefficient equation. Neither number certifies
a sub-150-byte ceiling, another predictor, or an integer-code bound.

| Difference in measured gain | A | B |
| --- | ---: | ---: |
| Continuous minus integer-probability ideal B | 0.117809 | 7.537343 |
| Integer-probability ideal minus actual raw saving B | -0.690994 | -0.590689 |
| Raw saving minus counted-rider saving B | 5 | 5 |
| Continuous minus counted-rider saving B | 4.426814 | 11.946654 |

B loses 15.399097% of its measured continuous gain on the integer path. That
path includes coefficient quantization and deployed probability arithmetic;
the evidence does not isolate coefficient rounding alone. Actual raw-stream
savings are differences between two terminated integer-length streams and
can exceed the ideal difference by a fraction of a byte. Therefore B's raw
saving is 42 bytes, not the rounded-down 41-byte ideal difference. The five
additional counted coefficients then reduce it to 37 bytes. Recovering the
entire measured 7.537343-byte integer-path loss would not bridge B's 113-byte
shortfall to the real-byte threshold.

## Verdict scope

**The charter's FORMULATION-scoped operational stopping rule fires:** the
tested coherent run-track plus joint-refit recipe saves 37 actual archive
bytes on move40, below the predeclared 150-byte threshold. Its 0.5–1.5 KB
central prediction fails for this implementation. The tested TC2/TC3 fixed
context-map mechanisms, bounded shared weights, fit selection procedure and
counted precision did not supply the proposed large gain.

This is not a proof that every lane predictor, run dynamic, richer causal
context, full-population or discrete coefficient optimizer, or task-space
generator must fail. Even the optimistic fixed-feature numerical bound above
exceeds 150 bytes. A and B change both geometry and fit scope, so their
comparison cannot isolate tracking's contribution from joint recalibration.
No broad geometry FAMILY closure is supported.

## Evidence checked

Verified both source trace envelope payloads against their receipts and the
retained source-control envelope; source full n600 field/control receipts
agree. Rehashed the source-control ZIP and confirmed exact equality to the
owned source ZIP. Verified raw RC64, padded envelope and rider twin equality
for A and B; checked rider reconstruction from header, counted weights and raw
stream. Serialized fitted coefficients and saved float coefficients match the
fit receipts, and all encode gain arithmetic recomputes from the recorded
total-bit sums.

Rehashed all 80 archive facts and 20 distinct rider facts, checked the exact
predeclared sample grid, read every ZIP's sole `p` member and verified its
unchanged source prefix plus retained rider. Each compressed rider decodes to
the same variant config and raw stream. Both per-variant minima, the selected
A archive and its recorded SHA/size agree. Fit/encode/prepare joins refer to
the same current trace binding. Stored keep masks were counted independently;
this audit did not rerun the native objective or rehash every multi-megabyte
prepared feature shard. The earlier source and live-binding reviews cover the
frozen implementation and source manifest.

| Receipt relative to move40 store | SHA-256 |
| --- | --- |
| trace/RESULT.json | `2590a835e402afddc54e7ed6c26b6608a188b7350e5e11b5a08ab7d8fa3538b9` |
| prepare_v2/RESULT.json | `0faa96cb79e4ea6886ecfb312ccfaa7b450771dac4ac86e312212f8e4daf67a3` |
| fit/A/RESULT.json | `ab00f060d7ba4ab60ec39a40385740d9e4154f4866f3497e3cc73a7529443e11` |
| fit/B/RESULT.json | `42cf01b5d32c1f1e1d7642fe229fcdb93d8a0eb09d7ce9f3de30922a93006d31` |
| encode/A/RESULT.json | `cc91726f44ba77361268648836d9b91ad8312a6c184d7d7e5ccb53ae0691caee` |
| encode/B/RESULT.json | `57ee1a43e7f60e622c89bcbd45ae4df01ed2b6a60542f0de5071252ab7faea32` |
| CONTAINERS.json | `97c1d1515ef847820e0ee5b125f2f7bc33e18e5eecfb0636312dd7fcf52041f2` |
| CANDIDATE.json | `0ca8c172610355d4a294127455785809d2a599151e753246d3c937bdf46b4833` |

## RECALL EVIDENCE

Re-read the previous measured-comparison review, current fit/encode source,
current rider parser and receiver container-selection source. The prior
original-recall and historical TC2 denominator checks remain recorded in
`measured_comparison_review.md` and the geometry/main-arm recall records.
Bounded source queries covered
`binding|ref.prepare|source_frame|inherited|ROOT|LIVE|FIELD|resume` and
`def split_member|def read_archive_member`; direct reads covered the fixed
feature objective, probability-selected keep mask, coefficient rounding,
actual envelope/raw/rider serialization and finite container menu. Current
receipts and retained bytes supplied every new numeric claim. The same
historical-field and numerical-bound scope restrictions still apply.

Disposition: FOLDED into the parent's active current-field memo and public-proof
harvest; owner parent ddm_tc3; consumer current move40 store and final TC3 memo.
This audit introduces no new job or independent follow-on queue row.

LIVE-HYPOTHESES: A's measured 79-byte archive saving may survive the current
public receiver; the implemented causal decoder matches its encoded mechanism,
but this audit supplies no completed literal-shell proof. Other geometry or
generator mechanisms remain outside the tested formulation and have no gain
claimed here.

DEAD-ENDS: repeating this unchanged move40 B recipe to seek the charter's
large gain is closed by its 37-byte result. A universal closure of lane
prediction is unsupported. Historical move37 oracle ratios cannot be promoted
to matched current-field oracle recovery, and old move39 public evidence cannot
stand in for the move40 public proof.
