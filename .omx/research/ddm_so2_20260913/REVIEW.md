# SO2 design review receipt

Scope: one design memo and one rung specification. Review owner: ddm_so2, self-review.
No independent-agent review, implementation test, trained model, encoded candidate or scorer result
is claimed. The charter explicitly restricts this arm to design; the MAIN-owned adapter remains
unbuilt. This receipt is not a Python review-gate mark for that future adapter.

## Numerical and mechanism review

* Re-derived current archive census from the actual RX1M header and the SJ1 subset7 pricing
  receipt. Found the charter's move-48 stream total was 42 B stale. Kept the stricter charter
  pass ceiling 123,998 B, corrected current comparison to 130,567 B.
* Re-derived conditional target using the real evaluate.py arithmetic order. The minimum
  saving for sub-0.12 is 24,515 integer bytes; 5% only buys 6,569 B against move 49. The
  memo cannot call that pass a solved cross. No score is measured by this arm.
* Checked the lifting inverse algebra and the 600x384x512 five-label cache schema. All detail
  symbols, including zero/nonzero decisions, must be coded. Every original token proposal remains
  representable; an anchor edit changes up to four transformed symbols. This is a proof of the
  specified map, not execution of that map on the real data.
* Rejected a forecast inferred from class shares, patch area or conditional-information ranking.
  The learned price is UNKNOWN until MAIN trains and physically packs/encodes it. The numerical
  intervals in the memo are admission targets, not a fitted-prior estimate.
* Rejected a naive Lane split after the independently recalled D3B exact-rate receipt. NO1's
  existing broad learned-model row is folded into this one concrete measurement, not duplicated.

## Interface and custody review

* Parsed the actual trainer's isolated `_build_parser` definition and fed it the exact memo argv.
  All flags accepted; all 29 CL2 reference configuration fields matched. Only argparse ran,
  not the trainer entry point, model construction, cache materialization or any training.
* Checked epoch selection and resume source: checkpoint state_dict is the EMA shadow; continuous
  phase ends at 29 and terminal QAT at 60; latest.pt is the resume source. No entropy-best
  checkpoint or drift waiver is selected. DPI1's initializer includes the omitted depth buffers.
* Traced current model container through HPR1: IHS1 -> counted RC3H -> CK2 -> Brotli q10/lgwin22.
  The old CL2 direct-Brotli pack is explicitly excluded as today's model price.
* Found the stock RLC1 pricer accepts a new FIELD only, while HPR1 accepts a new PRIOR. No stock
  end-to-end command joins them. The memo now specifies the adapter and its exact readiness
  boundary rather than inventing flags or modifying pointer guards to manufacture a control.
* Required true independent decode; known-symbol injection and twin encoders alone do not prove it.
  Both stream processes, model intermediates and final envelopes must be retained and hashed.
* The eight-byte SO2 prefix is charged inside the sole stored ZIP member. No unpriced support,
  class-map, video-derived table or learned constant is placed in receiver code.
* Public receiver, raw identity, timing, native parity and candidate-contract proof are confined
  to the conditional MAIN receiver branch. No archive/inflate or scorer validity is claimed today.

## Final check

After correcting a malformed line break in the falsifier paragraph, rechecked the source pins,
thresholds, CLI parse, reference configuration and the three follow-on dispositions. All future
work has owner, consumer and trigger. No new code or contract mutation was introduced. Remaining
uncertainties are explicit: learned price, adapter implementation, real inverse decode and public
receiver timing. These prevent a measurement or readiness claim, not delivery of a design memo.
