# DDM HP3 exact-evaluation queue

**Disposition:** QUEUED-WITH-FIRE-ORDER  
**Owner:** MAIN scorer scheduler, or the successor that first claims the released `ddm_ai1` n600 scorer slot  
**Consumer store:** `.omx/state/main_hot_state.md` plus
`.omx/research/ddm_hp3_20260810/EXACT_EVAL_RECEIPT.json`  
**Fire trigger:** `ddm_ai1` has released the sole n600 scorer slot, the lane claim is recorded, and the exact
archive/runtime hashes below still match. Fire this candidate before any further HP3 mutation.

## Immutable input

- Archive: `/Volumes/VertigoDataTier/pact/ddm_hp3_20260810/retained/candidates/requant_frame_embed_step2_hpm300/archive.zip`
- Bytes: `191044`
- SHA-256: `004436ea59780708e446392b33ab8d8ab5ce287622f5dd919a75208abee638ae`
- Receiver-closed runtime:
  `/Volumes/VertigoDataTier/pact/ddm_hp3_20260810/retained/winner_submissions/requant_frame_embed_step2_hpm300_004436ea59780708_f59bf2e8fe46/`
- Decoded-token SHA-256:
  `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`
- Inflated raw SHA-256:
  `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`
- Full scorer-free receipt:
  `.omx/research/ddm_hp3_20260810/FINAL_RECEIPT.json`

## Required row

Run exactly one full n600 `upstream/evaluate.py` row on the immutable archive/runtime under the
available authority axis, record exact `d_seg`, `d_pose`, archive bytes, recomputed score, hardware,
commands, and hashes, then update the live pointer only if the row qualifies. The unchanged raw output
and eight-byte rate reduction imply `S = 0.1721359706202714696` if the established CPR1 components
reproduce; this number is DERIVED, not an exact-eval claim.

