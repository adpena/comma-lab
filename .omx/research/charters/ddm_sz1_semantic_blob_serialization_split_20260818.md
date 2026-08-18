# ddm_sz1 — semantic-blob serialization SPLIT: receiver-side un-split + byte-close (the −515 B lever)

**Spawned:** 2026-08-18 by MAIN, from fx2's R5 finding (`.omx/research/ddm_fx2_model_axis_all_sections_20260818.md`,
"separate_finding_not_in_these_candidates" in `ddm_fx2_t4_sealed_fire_order_20260818.json`).
**Owner model:** Opus arm. **Task:** successor row under #1116.

## The finding you are converting into a candidate

The archive's semantic blob ships as ONE Brotli-coded section of 34,763 B. Inside it, 8,284 B of
fp16 metadata (scale/offset arrays) are stored interleaved and un-entropy-coded. Byte-SPLITTING
that metadata (planar/SoA layout: high bytes grouped, low bytes grouped) BEFORE the container's
own Brotli yields 34,248 B — **−515 B, measured through the real coder at the container's own
parameters, with a control that reproduced the shipped 34,763 B at delta +0.**
Receipts (SHA-pinned):
- `/Volumes/APDataStore/pact/ddm_fx2/probe/r5c_scale_split_f12.json`
  sha256 `88aee37e349f73b01fedb14dacabc2c3ea21271436fbd5bf7d7e03852e0aa2d6`
  (`{"shipped": 34763, "control_rebrotli": 34763, "byte_split": 34248, "delta": -515}`)
- `/Volumes/APDataStore/pact/ddm_fx2/probe/r5b_main_relay_rows.json`
  sha256 `84d51ab1f15e3a631d17e6a79ea019896862dd40e4ca1ee37f99eb568dcc5eb7`
- pd1's corpus-wide prior: weight SERIALIZATION is unsaturated (PR133 semantic renderer −13.0%,
  HPAC model −25.9% standalone re-code; nobody entropy-codes a scale array) while entropy-coded
  sections are saturated — store at `/Volumes/APDataStore/pact/ddm_pd1/`.

This is a SERIALIZATION change, not a probability model: the decoded VALUES are unchanged; only
the on-disk LAYOUT moves. The receiver must therefore gain a deterministic UN-SPLIT (rule-118
free receiver code, zero counted bytes).

## RECALL FIRST (apparatus clause — run before designing)

Grep `.omx/research/` + the corrections index for: semantic blob serialization, byte-split,
planar fp16, mz2 q3/q4 receiver-close (its 2 sub-KB candidates live on the SAME section family —
check for overlap/composition), sv2's IX2TOK01 LZ-match-structure law (#859 — the live coder pays
for match structure; your split must WIN through the real Brotli, which the probe already shows).
Consult `/Volumes/APDataStore/pact/ddm_fx2/RETENTION_MANIFEST.json` for the retained section
payloads. Never recall from working memory alone.

## Deliverables

1. **Encoder-side split + receiver-side un-split**, versioned: a section-format tag such that an
   archive WITHOUT the tag decodes exactly as today (inactive = byte-identity — the RG1 pattern),
   and an archive WITH it un-splits deterministically before value decode.
2. **Round-trip proof:** decoded tensors from the split section == decoded tensors from the
   shipped section, bit-exact; full-archive rebuild; determinism repeat byte-identical.
3. **Composition candidate:** the split composes with the fx2 candidate A token win on DISJOINT
   sections — build the composed archive (fx2 tokens + split semantic blob), measure total bytes,
   verify the seven untouched sections byte-identical and the decoded token field
   `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` unchanged.
4. **Sealed fire-order** (MAIN fires; you fire nothing, spend nothing): archive sha+bytes,
   re-pinned runtime tree carrying the NEW receiver, entry-point smoke receipt, pre-registered
   falsifiers (d_seg AND d_pose EQUAL to base to all digits — values unchanged by construction;
   archive bytes within ±2 B of projection; decode wall-clock delta ~0 — an un-split is O(n) copy).
5. **Bounded extension (measure-only, do not ship):** apply the same split transform to the other
   sections' fp16 metadata (carrier_blob, table_codes) and report per-section deltas — pd1 says
   the axis is corpus-wide unsaturated; ship ONLY the proven semantic split this arm.

## OPTIMAL FORM

Reference form: the byte-split transform measured by fx2's r5c probe through the REAL Brotli at
the container's own parameters with a delta-+0 control (receipt pinned above) — your encoder MUST
reproduce −515 B ±2 on the same section before any receiver work counts. Deltas: n/a (full
section, full archive — no scope reduction). MECHANISM reductions: NONE permitted — an
entropy-ESTIMATE byte claim is a TOY; only real-coder section bytes and real-archive totals are
admissible. Provenance pins: the two probe receipts above; the fx2 corrector commit
`85880c77a6`; the incumbent archive `65c75d7f097df930760b6611209f9caf66f5b10914cefa8d954b6d7834f6b0c4`
(180,601 B) and rr4 `35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956` (181,161 B).

## Prior-law prediction (m38 — adjudicated at landing)

Standalone on the rr4 base: −515 B ⇒ ΔS ≈ −3.43e-4. Composed with fx2 candidate A (−711 B
tokens): ≈ −1,226 B vs rr4 ⇒ ΔS ≈ −8.16e-4 vs rr4, ≈ −4.4e-4 vs the fx1 pointer. Falsifiers:
(a) receiver un-split fails bit-exact round-trip ⇒ report the exact divergence, do NOT force;
(b) composed archive misses the additive projection by >10 B ⇒ the sections interact through the
container — report the interaction, do not paper over it; (c) decode wall-clock delta >5 s ⇒
report (should be ~0).

## Constraints (binding)

- NO Modal, NO paid dispatch, NO Metal/MLX-GPU, NO n600 scorer jobs. All measurement is
  byte-level + bounded local decode.
- ALWAYS KEEP THE PAYLOAD: every candidate section + archive retained with sha256 under
  `/Volumes/APDataStore/pact/ddm_sz1/` + RETENTION_MANIFEST.
- Commits via `tools/subagent_commit_serializer.py` with POST-EDIT working-tree sha;
  .py files need 2 review-tracker passes; NEVER REVIEW_GATE_OVERRIDE on .py.
- No co-author trailers; no backticks in commit messages. Cite
  `docs/operating_manual_craft_handoff.md`. `upstream/` is READ-ONLY.
- Final message: durable memo path + verdicts vs the prior-law prediction + fire-order path.
