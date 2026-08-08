# ddm_tr2p1 — TROT joint-from-marginals residual stream RACED vs CR1's edge-conditioned incumbent

**Fire-order (2026-08-08, from tr2 row 1 `tr2_r1_joint_from_marginals_edge_support`,
QUEUED-WITH-FIRE-ORDER → FIRING at free slot):** race the CR1 edge-graph incumbent against a
TROT (Tsallis-regularized OT / ecological-inference) joint-from-marginals residual stream on
the SAME payload with the SAME coder harness. Axis: `[byte-only scorer-free]`. $0, CPU-only.

**Measured incumbent (recall-first — cite, do NOT re-measure):**
- Payload: CR1 selected edge-labeled n600 support (5 edges: Road↔Lane 1,143,497 px ·
  Road↔MyCar 606,290 · Road↔Undriv 511,976 · Undriv↔Movable 150,302 · Road↔Movable 142,295;
  total 2,554,360 px, 506,837 cx1 direct flips).
- Incumbent: **edge-conditioned lzma1-raw 464,557 B**
  (sha `0a53f649768c61912399ccab14e4d3323998e47235992091e2a9e28cf7259fe1`,
  artifact `/Volumes/VertigoDataTier/pact/ddm_cr1_20260808/payloads/p2_edge_conditioned_support.lzma1-raw.bin`).
- Baseline: pooled lzma1-raw 575,095 B → incumbent delta −110,538 B (−19.221%).
- Inputs (SHA-pinned):
  GT argmax `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy`
  (`b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d`) ·
  cx1 argmax `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy`
  (`5e903de650e60ec6a64b34eb455fa1bc911223551d0b31e9ae45cc906e1490be`).
- Selection mode: `n600_all_pairs_no_prefix` (KEEP — never a prefix, m88/m96).
- Coder primitives: reuse the landed surfaces in `experiments/ddm_bd1_class_field_receiver.py`
  + `experiments/ddm_pe1_per_edge_partition_race.py` (brotli-q11 / raw-LZMA1 / zlib-9 /
  smevr-r7-nibble with decode checks). Same-coder discipline per #940 races-not-reputation.
- Full receipts: `.omx/research/ddm_cr1_20260808/CR1_FINDINGS.md` + `CR1_RECEIPT.json` +
  `.omx/research/ddm_tr2_20260808/TR2_ROWS.jsonl` rows 1 and 6 (this probe's registration).

**Arms (per the registered fire-order, all racing to reconstruct the SAME selected
edge-labeled support arrays with EXACT decode equality):**
1. **Incumbent control:** the 464,557 B edge-conditioned lzma1-raw stream (re-verified decode,
   not re-derived).
2. **TROT residual stream (the treatment):** carry {per-edge/per-class MARGINALS (counted) +
   side-information matrix (counted IF video-derived; FREE only if derivable at decode from
   already-carried/decoded state — apply the three-way test: operator-property/generic = FREE ·
   scorer-weights = economic · this-clip = COUNTED) + q/λ/config tags (counted) + RESIDUAL of
   the true support vs the TROT-reconstructed joint (counted) + framing (counted)}. Decode runs
   the TROT solve FREE (rule-118 generic algorithm) then applies the residual. Sweep q over a
   small ladder (e.g. q ∈ {0.5, 0.8, 1.0, 1.2, 2.0} or as the math suggests) — q=1.0 IS the
   Sinkhorn arm.
3. **Marginals-only control:** marginals + residual, NO OT solve (does the joint-from-marginals
   PREDICTION earn its bytes beyond what marginals alone give the residual coder as context?).
4. **Identity/container control:** the raw selected support through the same coder set
   (container/framing overhead isolation).

**Pass condition (pre-registered):** total counted bytes < 464,557 with exact decode equality
to the selected edge-labeled support arrays. **Falsifier:** no TROT-q arm beats 464,557 B under
exact decode equality, OR the win depends on omitting counted video-derived side information.
An honest LOSS-w/-bytes row is a full success of the probe — record it and close the
formulation (verdict_scope: FORMULATION — q-family joint-from-marginals residual coding of the
CR1 selected edge-labeled n600 support payload).

**Implementation route (tr2 row 6):** author a minimal deterministic local TROT solver (POT has
NO native Tsallis solver — checked; POT usable as q=1 Sinkhorn baseline/reference only, no
unpinned production dependency). VALIDATE the solver on small deterministic fixtures FIRST
(reproduce the q-family balancing: q→1 recovers Sinkhorn/KL; marginal constraints satisfied to
tolerance) — fixture validation is solver-parity ONLY, never a family verdict. Deterministic:
seeded, fixed iteration counts or recorded convergence tolerances in the receipt.

## OPTIMAL FORM

Race at reference form: the REAL CR1 payload (SHA-pinned caches above), the SAME landed coder
harness, exact decode equality — zero mechanism reduction. Fixtures appear ONLY as solver
implementation validation (declared TOY-BRACKET: fixture rows cannot produce a family verdict).
SCOPE reductions allowed and declared: the q ladder may be coarse (≥4 values incl. q=1);
solver iteration budget may be capped IF the residual is coded against whatever joint the
capped solve actually produced at decode (decode reproducibility > solver optimality — the
decode rerun must reproduce the SAME joint bit-exactly, same seed/iterations/tolerance).
Provenance pins: incumbent stream sha
`0a53f649768c61912399ccab14e4d3323998e47235992091e2a9e28cf7259fe1` · gt argmax cache sha
`b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d` · cx1 argmax cache sha
`5e903de650e60ec6a64b34eb455fa1bc911223551d0b31e9ae45cc906e1490be` · CR1 receipts commit
lineage per `.omx/research/ddm_cr1_20260808/CR1_RECEIPT.json` · repo HEAD at arm start ·
solver config in receipt.

**Boundaries:** CPU-only (OMP/MKL ≤4 threads), NO Metal, NO scorer slot (ARM-VEH n32 leg 2 is
LIVE on Metal — read-only toward ddm_mx1e/ddm_mx1g run dirs). No archive promotion, no score
claim (`score_claim=false`, `promotion_eligible=false`). Findings:
`.omx/research/ddm_tr2p1_20260808/TR2P1_FINDINGS.md` + typed rows
`TR2P1_ROWS.jsonl` (per-arm: codec, counted-bytes breakdown {marginals, side-info, tags,
residual, framing}, sha256, decode-equality bool, q, solver config). Bulky payloads →
`/Volumes/VertigoDataTier/pact/ddm_tr2p1_20260808/` (certify-or-block).

## AMENDMENT 2026-08-08 (operator correction: "still naive and toy engineering basis")

The residual-then-LZ design above is a CONVENIENCE basis, not the family's reference form.
Reference form for label/support-stream coding = **prediction as PROBABILITY MODEL +
conditional arithmetic/range coding** (the form hb2 just measured on our own labels at
2.1% over its model's ideal: bpp 0.006641 vs ideal 0.006502). And #859/sv2 MEASURED that
generic-LZ outcomes are governed by LZ MATCH STRUCTURE, not symbol probability — a
transform/prior raced only through LZ coders yields an INSTRUMENT-scoped verdict, never a
family verdict. Binding changes:
1. The TROT treatment MUST include a conditional-AC leg: TROT joint → per-symbol conditional
   probabilities → deterministic arithmetic/range coder (own minimal coder or the repro_repo
   AC pattern as parity reference; counted = marginals + counted side-info + tags + AC stream
   + framing). The LZ-residual leg becomes a BASELINE, never the verdict carrier.
2. The INCUMBENT also gets a context-model AC leg (edge-conditioned per-symbol probabilities
   → AC) so the race is reference-form vs reference-form, not naive vs naive.
3. Any TROT LOSS verdict drawn only from LZ legs is scoped INSTRUMENT(LZ-coder), not
   FORMULATION. The pass bar stays: beat the best measured incumbent leg (≥464,557 B or its
   AC leg if lower) under exact decode equality.

**Discipline:** pact commits via `tools/subagent_commit_serializer.py` with POST-EDIT
`--expected-content-sha256` per file; tags `[no-triality] [p0-ledger-ok]`; review_tracker ×2
per pact .py; NO Claude/AI attribution or Co-Authored-By trailer. Recall-first before building
(memories + .omx/research + canonical equations — cite, never re-derive). If serializer hits
sandbox git-perms, write artifacts + say so. End-of-run: follow-on disposition table
(FIRED / FOLDED / QUEUED-W/-FIRE-ORDER — no orphans).
