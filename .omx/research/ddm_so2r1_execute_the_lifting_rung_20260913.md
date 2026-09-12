# SO2R1 — the lossless modulo-five lifting of move 49, trained, packed, encoded: **J_Z = 429,447 B, FAIL**

Owner/checkpoint: `ddm_so2r1`. Charter: `.omx/research/ddm_so2r1_execute_the_lifting_rung_charter_20260913.md`
(commit 6a50275d0). Specification: `.omx/research/ddm_so2_successor_object_priced_with_the_learned_prior_20260913.md`.
Lane `ddm_so2r1_execute_lifting_rung_20260913`. `research_only=true`, `score_claim=false`,
`promotion_eligible=false`, `pointer_moved=false`. Axis
`[macOS-CPU advisory; exact bytes, scorer-free]` for every byte here; no scorer, no renderer
forward, no Modal, no evaluator row.

**Verdict: SO2's single pre-registered scientific falsifier FIRED.** The specified lifting, one
prior, 60-epoch cl2 law prices at **J_Z = 429,447 B** against the 123,998 B pass ceiling — over
by **305,449 B**, which is **3.463×** the ceiling and **8,777** standard deviations of the
34.8 B container-break lottery. This is not a near miss and no amount of encoder luck reaches it.
The instance and formulation close. Every gate the charter required passed, so this is a
scientific negative and not an INSTRUMENT-REFUSED.

Labels: **MEASURED** = a run in this store with a retained artifact; **DERIVED** = arithmetic on
measured values, derivation shown; **ASSUMED** = awaiting verification. Store root
`/Volumes/APDataStore/pact/ddm_so2_first_rung/`, **1,079 files, 1.426 GiB** retained against the
charter's 8 GiB cap, every payload sha'd on the ExFAT destination itself (`RETENTION.json`, from
the retained `retention_manifest.py`), 1,165 `._` resource-fork stubs skipped as non-payloads.
Nothing was deleted and no run measured a payload and threw it away.

## The headline table (all MEASURED unless marked)

| row | packed prior | RC64 stream | +8 B | **J** | whole archive |
|---|---:|---:|---:|---:|---:|
| move 49, the incumbent (control) | 11,629 | 118,938 | — | **130,567** | 179,153 |
| the ceiling SO2's charter set | — | — | — | **123,998** | 172,584 (D) |
| incumbent prior on Z, no retrain (control 2) | 11,629 | 866,516 | 8 | **878,153** | 926,731 |
| **Z under the prior retrained on Z (the rung)** | **11,139** | **418,300** | **8** | **429,447** | **478,033** |

Margin against the ceiling: **−305,449 B** = **−8,777** × the 34.8 B lottery sd
(`[[container_break_delta_is_a_one_sample_lottery…]]`). To pass, the stream would have had to be
112,851 B at this prior size; it is 418,300 B, so the rung is short by a factor of **3.707** on
its own coded leg (D).

Conditional S, using the source evaluator's expression and denominator, division before
multiplication, D49 = 100·0.00010287 + sqrt(10·0.00000455) = 0.01703236878161602 (D):
`S(J) = D49 + 25·((48,586 + J)/37,545,489)`. At J = 130,567 this returns
**0.13632299781031237**, the stored pointer to its last digit — an independent re-derivation of
SO2's decomposition, not a restatement of it. At the measured J_Z = 429,447 it returns
**0.33533492**. The rung would have raised the score by 0.199, not lowered it.

## What the two controls establish

**Control 1 — the loop is the shipped loop.** `ddm_sj1_rlc1_price.py encode --field control`, in
this arm's own store, re-encoded move 49's own field in two independent processes and produced
the live archive **byte-identically**: 179,153 B, sha
`73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`, stream 118,938 B, twins
equal. Its own ideal (cross-entropy) bytes are 118,937.5 against 118,938 emitted — the shipped
integer-CDF coder sits within **0.5 B** of its model's cross-entropy on F. MEASURED.

**Control 2 — the HPAC member-identity control.** Repacking the archive's OWN IHS1 body through
this adapter's RC3 → CK2 → Brotli(q10, lgwin22) sequence reproduces the archive's own HPAC member
exactly: 18,929 B body → 11,629 B member, sha `925adb48dca8a77e…`, and the reconstructed rider is
byte-equal to `parts.hpac_blob`. Without this, the candidate's 11,139 B would only be "what this
code emits", not "what the shipped container charges". MEASURED.

**Control 3 — the incumbent prior on Z.** The unretrained move-49 prior encodes Z to 866,516 B,
**7.29×** its 118,938 B on F. The refit then recovers **51.7%** of that (866,516 → 418,300), so
the split is unambiguous: the representation costs 7.29×, a full 60-epoch refit under the
identical law buys back about half, and what remains is still 3.517× the incumbent's stream.
MEASURED.

## The rung's own receipts

* **Inverse equality.** All **117,964,800** bytes. The forward map works in the patch frame
  (reshape/transpose to (600, 6, 8, 64, 64)); the inverse is written independently in image
  coordinates with explicit strided slices, so agreement is evidence of the bijection rather
  than of one routine's self-consistency. `sha256(inverse(Z)) =
  fdf2255f60364dcd1e67fb7de107c0640f5fe3a39a7b575c76c86636660efd1d` = the source field's sha;
  0 mismatched bytes. Z: 117,964,800 B, sha `8f5b5cb87ff430f98efe14708c60dc53fc549fb8ddb1f3386274b9d245b451c5`.
  The npz and u8 custody forms of the source were independently re-hashed and agree. MEASURED.
* **Training.** 60 epochs, cl2_shipped_ladder, λ=1.0, seed 20260716, warm start from dpi1's
  pinned `init_depths.pt` (`f05bae5b…`), `--min-free-bytes 21,474,836,480` (20 GiB; APDataStore
  held 49.4 GiB free, a 29.4 GiB margin). **3,073.10 s wall, peak RSS 1,801 MiB, exit 0**, under
  a 7,200 s cap. Terminal EMA at epoch 60 was the pre-registered selection; no `best.pt` was
  consulted. Terminal bit-depth histogram over the 517 rows:
  `{0:77, 1:13, 2:46, 3:58, 4:99, 5:137, 6:67, 7:18, 8:2}`, mean depth **3.683** bits against the
  shipped prior's 5.890. Layout held exactly: 517 rows, 20,416 stored values, no row's value
  count moved. MEASURED.
* **Pack twins.** Candidate IHS1 body 16,650 B → rider 15,056 B → CK2 15,056 B → member
  **11,139 B**, sha `6b8be4dafdb650d0…`; both twins byte-equal; `rc3.restore_hpac(rider, counts)`
  returns the packed body. The candidate prior is **490 B SMALLER** than the incumbent's — the
  model leg is not where this rung fails. MEASURED.
* **Adapter identity controls.** The rebuilt archive parses back to the emitted stream and to the
  candidate rider; `semantic_blob`, `carrier_blob`, `tc1_weights` and `residual_payload` are
  unchanged; the RX1M header struct was asserted against the shipped header's size and its
  field 5 against the shipped HPAC length before being rewritten. Two independent encode
  PROCESSES produced the same archive sha `fc623943a0a484bdc7b9…` and the same J_Z. The receiver
  loop reconstructed Z exactly, frame by frame, for all 600 frames. MEASURED.
* **Independent decode.** In a fresh process, with no known-symbol injection and no encoder
  caches — asserted in code, not claimed: the module refuses unless
  `NativeDecoder.decode.__module__ == "runtime.entropy.rc64"` — the research member was unwrapped
  past its declared `SO2L` prefix, decoded under the deserialized candidate prior over all 600
  pairs, and the SO2L inverse applied. The decode returned
  `8f5b5cb87ff430f98efe14708c60dc53fc549fb8ddb1f3386274b9d245b451c5` = Z, and its inverse returned
  `fdf2255f60364dcd1e67fb7de107c0640f5fe3a39a7b575c76c86636660efd1d` = move 49's field, over all
  117,964,800 bytes. So the archive this rung built really does carry the pointer's field, at
  429,447 B instead of 130,567 B. Receipt: `candidate/decode/primary/DECODE.json`. MEASURED.
* **Cross-entropy against the real price.** The trained prior's own
  `−Σ log2 p(Z_i | decoded history)/8` is **418,299.6 B** against **418,300 B** actually emitted:
  **+0.4 B**. DPI1's 809 B surrogate-to-real reversal does not recur here, because this is the
  same model's own probabilities driving the same integer CDFs rather than a proxy. The
  surrogate was honest; the object is simply expensive.

## Why it fails — the mechanism, located (MEASURED, n600)

The single J number says the rung lost. Two $0 n600 diagnostics say **where**, and they are the
part of this negative worth carrying forward. Fixed 4-tap causal context (left, up, up-left, and
the same position in the previous plane), all 600 planes, no prefix
(`diagnostics/CONTEXT_ENTROPY.json`, `diagnostics/INPLACE_SPLIT.json`):

| field | bits/symbol | × F |
|---|---:|---:|
| F (move 49) | 0.016513301984900846 | 1.000 |
| in-place lifting — same differences, NO quadrant packing | 0.051851375076264870 | **3.140** |
| Z — SO2's differencing + 32×32 quadrant packing | 0.041354787344236170 | **2.504** |
| Z, anchor quadrant only | 0.030174668048946203 | 1.827 |
| Z, detail quadrants only | 0.042438076983813490 | 2.570 |

1. **The quadrant packing is not the defect — it is a 20% recovery** (3.140 → 2.504). The cost is
   the **modulo-five differencing itself**. No layout refinement rescues this family, so the
   obvious "try a different tiling" successor is already answered.
2. **The 99.171%-zero detail quadrants are worse per symbol than the anchors** (2.570× vs
   1.827×), even though 87,740,119 of 88,473,600 detail symbols are zero (MEASURED). SO2's
   plausibility argument — a class-independent shared zero event — is real and it does not pay.
   The differencing converts a smooth, contour-structured class boundary, which a causal model
   predicts almost for free, into a scattered set of isolated, essentially context-free nonzeros.
   The zeros are cheap; the contour was cheaper.
3. The anchors carry their own 1.827×: decimating F by two in each direction roughly doubles the
   boundary density per stored symbol, so even the part of Z that is "just F" costs more.

Caveat that travels with these three numbers: this is a **fixed 4-tap diagnostic, not the shipped
prior**. It scores F at 243,499 B against the real 118,938 B stream, so the learned prior is
2.05× stronger than the diagnostic on F; only the RATIO transfers, never the absolute. A 60-frame
prefix gave 2.393 rather than 2.504 — a 4.6% understatement, so the n600 run was not ceremony.

## Scope of the closure

SO2 pre-registered: *"after a passing original-field control, terminal-EMA pack, independent twin
encodes, and full inverse decode equality, J_Z > 123,998 B closes this specified lifting +
one-prior + 60-epoch law at INSTANCE/FORMULATION scope."* All four preconditions passed, so the
closure is the scientific one, at that scope and no wider:

* **CLOSED:** this 2×2 modulo-5 lifting inside 64×64 patches, quadrant-packed, one prior, 60
  epochs at λ=1. Also closed by diagnostic (2) above: re-tiling or re-packing the same
  differences, since packing was already the helpful half.
* **NOT closed:** lossless recodes of this field in general; learned-probability successors that
  do not destroy boundary contours; NO1 row 2's broader lossless-learned-probability lead. One
  formulation's death is not the family's (Catalog #307).
* **Not evidence about distortion at all.** The construction is an integer bijection; Δd_seg and
  Δd_pose are exactly [0,0] by construction, proved here by full inverse equality. The rung
  failed purely on rate.

## What a successor would have to beat

To reach the ceiling from here the coded leg must fall 3.707× at a held prior size; to reach
SO2's sub-0.12 subsystem ceiling of 106,052 B it must beat the incumbent's own 118,938 B stream
outright. Any successor representation should be screened by the cheap n600 conditional-entropy
diagnostic in `diagnostics/` BEFORE a training slot is claimed: it cost minutes, it predicted
this outcome's shape correctly, and it would have said "no" here for the price of two CPU-minutes
instead of 3,073 s of Metal plus four 600-frame encodes. That screen is this rung's most reusable
output.

## Boundaries honoured

No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`, no evaluator dispatch, no scorer, no
renderer forward, no seal, no packet, no pointer write. `upstream/`, the PR tree, sealed trees,
contract code, receiver code, the renderer, the carrier and the live field were read-only; Z is
a new artifact under this arm's own root. sj1/cb1/cr1/pp1/dpi1/rq1 directories were read only,
`init_depths.pt` by sha. Neither stock CLI was patched: `ddm_sj1_rlc1_price` binds its own source
sha into every encoder checkpoint receipt, so an edit would refuse every in-flight sister resume
(`[[binding_hash_whole_module_kills_checkpoints_20260909]]`); the second control therefore ran in
a SEPARATE store root rather than by mutating the first root's `INPUTS.json` under two live
encodes. `sys.dont_write_bytecode` is set against the read-only trees. Vertigo was never touched
for writes and its 40 GiB reserve was never lowered: at charter time it read 38 GiB free, under
its reserve, so everything was rooted on APDataStore, which held 49.4 GiB and finished at 0.948
GiB used. All heavy stages ran through `tools/launch_detached_process.py` with `--nice 0` and
done receipts; every wait was bound to a completion artifact, never to a clock.

## Observability surface

Inspectable per layer: per-stage JSON (`TRAIN_INPUTS.json`, `pack/PACK.json`,
`encode/<tag>/ENCODE.json`, `decode/<tag>/DECODE.json`, `RESULT.json`) plus every intermediate
payload (IHS1 body, rider, range payload, CK2, Brotli member, RC64 stream, both archives).
Decomposable per signal: J splits into prior / stream / prefix, and the stream into a 600-entry
per-frame bit ledger. Diff-able across runs: two independent encode processes with byte-compared
archives, plus in-process twins at every pack and encode. Queryable post-hoc: all of the above is
machine-readable JSON. Cite-able: every receipt carries producer sha, input shas, live-archive
sha and the checkpoint's causal-state sha. Counterfactual-able: the two diagnostics answer
"what if the packing were removed?" and "what does each quadrant cost?" without re-running.

## Wire-in hooks (Catalog #125)

Sensitivity map — N/A, no new score-relevant actuator exists. Pareto constraint — N/A, the row
is off the frontier by 3.463×. Bit allocator — N/A. Cathedral autopilot dispatch — N/A, nothing
promotable. Continual-learning posterior — the closure and its located mechanism are this memo
plus the retained `RESULT.json`. Probe disambiguator — ACTIVE: `diagnostics/context_entropy.py`
and `diagnostics/inplace_split.py` are the reusable n600 screen for any successor representation.

Canonical equations recalled, not appended: `coder_strength_substitutes_for_capacity_v1` (the
refit recovered 51.7% and the prior got 490 B smaller while the stream stayed 3.517× — coder
strength did not substitute for the representation's lost context), `hpac_prior_capacity_slope_v1`,
`hpr1_counted_section_refit_debt_v1`. No new law is registered on one instance.

<!-- # FORMALIZATION_PENDING: one instance/formulation negative plus two n600 diagnostics; the "differencing destroys contour context faster than shared-zero events repay it" observation is a located mechanism on ONE representation, not yet a law, and registering it off a single rung is exactly the over-generalisation the equations registry exists to refuse. -->

composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49), unchanged.
