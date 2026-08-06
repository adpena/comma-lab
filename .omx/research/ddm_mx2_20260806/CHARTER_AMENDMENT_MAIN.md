# CHARTER AMENDMENT (MAIN, binding at landing review) — ddm_mx1 + ddm_mx2

Read BEFORE claiming any parity verdict. These are ESTABLISHED MEASURED findings (m44:
recall from stores, not working memory) that the charters did not name explicitly:

1. PARITY ON REAL FRAMES ONLY (NO-FAKE #3): every parity gate (forward max-abs, argmax,
   loss-at-one-step) MUST run on REAL decoded pairs from upstream/videos/0.mkv (or the
   built label caches), NEVER random/synthetic tensors. A synthetic-fixture parity PASS
   is the canonical fake — argmax-tie structure only exists on real content. Landing
   review will refuse synthetic-only parity receipts.
2. #855 HAZARD (MEASURED): the DEFAULT MLX conv adapter flips 76 argmax pixels on REAL
   frames, systematically. If you bind our MLX SegNet surfaces, verify which adapter you
   inherit and report the argmax-diff count on real frames explicitly (0 or the measured
   number + tolerance rationale). Do not assume the default path is parity-clean.
3. #903 LESSON (MEASURED + CURED): upsample-VJP scatter × Adam sign(g) made 40/41 arrays
   diverge while the LOSS SCALAR was identical — loss-parity at one step is NOT gradient
   parity. If your port trains, add ONE gradient-parity check (per-tensor max rel-diff,
   real input) torch-CPU vs MLX-CPU, or explicitly scope the claim to forward-parity.
4. BATCH SHAPE IS PART OF THE INSTRUMENT (08-06, batch_shape_is_part_of_the_forward_
   instrument): pin (code, weights, threads, batch shape) in every parity receipt; a
   batch-16 reference vs batch-1 consumer flips ties.
5. hb1-lineage note (mx2 esp.): their e2e.py:1222 self-compress stage warm-starts from
   hpac_p64 — continuation-init is THEIR form. If any leg of your port chooses an init,
   cite the e2e.py line for it (calibration-lineage recursion; no invented defaults).
