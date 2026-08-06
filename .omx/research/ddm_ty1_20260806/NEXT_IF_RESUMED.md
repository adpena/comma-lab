# ddm_ty1 next if resumed

1. Run TY1-R2 over the machine corpora before claiming global exhaustion:
   AU1 `au1_corrections_index.jsonl`, AU1 `au1_headline_vs_body.jsonl`, VO2
   registry rows, and `probe_outcomes.jsonl`. Output should be adjudicated rows,
   not raw string hits.
2. If scorer-free coding work is the next best EV, implement a PR130/HPAC-class
   semantic-stream coder for the TK1 5-class maps and compare against the
   142,001 B PP1 KT baseline with counted bytes and exact decode.
3. If a scorer lane opens and semantic renderer risk is the blocker, run TK2 D1
   C0/C1/C2 n600 one candidate at a time, chunked with `--resume`, after claiming
   the lane.
4. Do not cite lane C12/C14/C15 as in-training Lane negatives. Any Lane burn
   must use fixed epoch-0 gates and read Lane Betti-0 plus per-class d_seg.
5. Do not promote ancestor PR101/int8-symbol coder negatives to current
   semantic/IX2 token streams. Every coder family verdict must name the object
   it encoded.

No scorer slot was owned or consumed by TY1. No exact score moved.
