# ddm_wh1_wrong_half_decomposition — decompose the ~76,600 B "is-my-argmax-wrong" half of the token stream (the ONE undecomposed surface; harness tasks #1259/#1263/#1204, owning memo ddm_hc1 arc / token-stream-is-one-binary-question)

## MANDATE

Operator 20260825: *"Believe in yourself and feel free to be creative and weird and think
divergently"* + full-authority frontier-score-lowering GO. hc1 measured (memory
[[token-stream-is-one-binary-question]]): the dx2-lineage token stream is 97.80% ONE binary
question — "is my argmax right?" costs 111,276 B while "which class instead" costs 2,501 B
(2.69 bits to say wrong vs 0.09 to say which). The ~76,600 B WRONG half is 181% of the
42,382 B demand and is UNDECOMPOSED — six arms measured the stream in aggregate, nobody
asked WHERE those bits go (#1204: "the arithmetic coder already knows exactly"). Stage B of
the live diagonal (jg2 re-encode on the moved object) will re-ask this exact question on
the NEW field; a decomposition instrument built NOW both (a) probes the one open surface
and (b) becomes stage B's diagnostic the day the moved object exists.

## SCOPE

1. BUILD the per-position bit ledger: instrument the live entropy-coding path (the
   F26/HPAC conditioned coder on the gb1 body — runtime custody at the shipped tree, model
   receipts in the gb1/jt21 verdict memos `ddm_gb1_groupbin8_verdict_20260824.md` +
   `ddm_jt21_joint_21family_reencode_verdict_20260825.md`) to emit −log2 p per coded token
   decision, split into the WRONG-indicator bits vs WHICH-class bits, tagged with
   {pair, cell position, class, top1-runnerup margin bucket, g4 spatial-stationarity
   stratum}. The coder computes these probabilities already — this is READ-OUT, not a new
   model (the #1266 law: a paid model misses break-even 47.4×; this arm ships ZERO new
   counted bytes).
2. DECOMPOSE: where do the ~76,600 B concentrate? Emit the measured marginal tables:
   by-class (Lane's 33.56%-of-model-bits law, lane_is_0.59pct_area memory, predicts Lane
   dominance) · by-margin-bucket · by-spatial-stratum · by-pair-rank. Name the top
   concentration cells with receipts.
3. PRICE the exposed structure honestly: for the top concentration cells, compute the
   CONDITIONAL entropy gap (actual bits paid vs an oracle-conditioned bound) — an upper
   bound on what ANY better conditioning could recover, compared against the measured
   model-axis ceiling (~2,009 B remaining per [[dx2-block-ceilings-are-measured-and-sum-to-5-percent]]).
   If the gap concentrates where the ceiling table already priced it, say so plainly — the
   finding is then CONFIRMATORY, not a new lever.
4. PACKAGE as a reusable instrument (`tools/token_wrong_half_ledger.py`-class) that stage B
   runs on the MOVED object's re-encoded stream — the stage-B consumer contract is one
   paragraph in the memo.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_wh1_wrong_half_decomposition/`.
- The gb1/dx2 custody trees + shipped runtime are READ-ONLY; instrument a COPY of the coder
  path, never the sealed tree. Do NOT edit the live wd3 trainer (builder pin 0b976d0d0a).
- Every ratio carries NAMED numerator+denominator with a receipt path (the wa1
  phantom-ratio pattern).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- [[dx2-block-ceilings-are-measured-and-sum-to-5-percent]] — model-axis ceiling 2,162 B
  gross (z=11.89), coder 88 B: this arm does NOT expect a byte lever inside the dx2 body;
  its value is the MAP for stage B's new object. Framing any dx2-body recovery > ~2 KB as a
  candidate would contradict the measured ceiling — treat that outcome as a red flag on the
  instrument, not a win.
- `ddm_hc1` arc (task #1259) — calibration CLOSED at 237× below bar; do not rebuild the
  calibration probe.
- Task #1204's own framing — the aggregate measurements exist ×6; do not re-measure
  aggregates, decompose.
- [[perfect-localization-is-worthless-the-address-is-the-tax]] (×3 measured) — any recovery
  scheme that must NAME positions pays the address tax (df1: address floor 3.15× the
  prize); the only admissible exploitation is CONDITIONING (address-free), which is why the
  output feeds the stage-B re-encode rather than a sidecar design.

## OPTIMAL FORM

- Family exemplar: the mi1 conditioning-family instrument arc (reference: task #1263/#1266
  rows — the groupbin8 decode-scan family that produced the gb1 pointer move, commit
  884bb65f1e verdict memo `ddm_gb1_groupbin8_verdict_20260824.md`) — same shape: read the
  coder's own probabilities → find conditional structure → zero-transmitted-byte
  conditioning. wh1 is that family's DIAGNOSTIC generalization.
- SCOPE reductions declared per row (n600 full stream required for the marginal tables — no
  prefix subsets per the m88 prefix-bias law; a bounded per-pair sample is legal ONLY for
  the instrument smoke, never for the tables). MECHANISM reductions FORBIDDEN.
- **PRIOR-LAW PREDICTION (falsifiable):** the lane×rate join law (Lane = 33.56% of model
  bits at 0.59% area) + the cost-hurts join (what costs most is what hurts most, ar1b)
  predict the WRONG-half bits concentrate ≥3× over area-uniform in Lane cells and in the
  low-margin bucket. FALSIFIER: a near-uniform decomposition (no cell class reaching 2×
  enrichment) — count it plainly; it would mean the wrong-half is conditioning-saturated
  and stage B's re-encode gains come only from the field move itself.

## DELIVERABLE

`.omx/research/ddm_wh1_wrong_half_decomposition_20260825.md` — the instrument (committed,
2 review passes) · the n600 decomposition tables with receipts · the conditional-entropy-gap
pricing vs the measured ceiling · the one-paragraph stage-B consumer contract. Commit via the
serializer. End with the own-vehicle frontier line.
