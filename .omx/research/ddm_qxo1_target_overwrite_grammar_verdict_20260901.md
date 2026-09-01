---
schema: ddm_qxo1_target_overwrite_grammar_verdict.v1
date: 2026-09-01
arm: ddm_qxo1_target_overwrite_grammar
status: UNDER-GATE
axis: "[scorer-free exact receiver/rate measurement]"
score_claim: false
pointer_moved: false
selection_mode: seeded_random_n1000_then_full_n17926
custody: /Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar
---

# QXO1 target-overwrite grammar — semantic proof passes and the 15,417 B section puts the exact archive 8,676 B under gate

**Verdict: `UNDER-GATE`.** This changes the QX **rate-shape** map only. It does not
change BR2's standing measured distortion refusal: the retained born instance scored
`S=51.18372088274943` on `[macOS-CPU advisory]`, with realized distortion
1,045.9968 times its allowance. QXO1 loaded no scorer and measured no Seg, Pose, or
contest score. Its exact 129,309-byte representation row therefore supplies one Cross
half; BR2's adverse distortion receipt remains the named mechanism/evidence for the
other half, and the own-vehicle frontier is unmoved.

The new semantic object is receiver-valid and below the complete byte gate. Against
the freshly decoded QBT field, 9,177 of 17,926 historical events are target no-ops and
only 8,749 writes change state. QXO1 omits the no-ops, drops every per-event target and
historical-baseline label, and places each remaining site in one of five fixed target
stream slots. The target is **implicit in the counted stream slot**, not information-free.
Brotli q11 prices that exact section at **15,417 B**. With QX1's 113,844-byte core and
the 48-byte section envelope, the physical archive is **129,309 B**, or **8,676 B below**
the largest legal 137,985-byte archive.

All numbers below are full-population `[scorer-free exact receiver/rate measurement]`
unless explicitly labeled as the seeded proof sample. No scorer, `upstream/evaluate.py`,
Metal, MPS, Modal, remote job, or training process ran.

## Consumer-semantics proof

The source contracts distinguish the old and new objects:

1. The retained S2 parser requires strictly increasing, unique sites
   (`src/tac/optimization/s2_partition_seed.py:130-145`). Its historical consumer checks
   the event's old baseline class before writing the target
   (`src/tac/optimization/s2_partition_seed.py:379-394`), and the predictor-bound public
   consumer additionally pins the predictor semantic SHA before calling that path
   (`src/tac/witness_dsl/predictor_bound_residual.py:323-357`). That is the historical
   C1-syndrome ABI; QXO1 does not claim to reproduce it.
2. QX3 reconstructs QBT from counted QX1 sections, evaluates all 600 pair identifiers,
   and verifies the fresh field against the retained native field
   (`experiments/ddm_qx3_receiver_closure.py:308-377`). QXO1 executed this source path:
   **0 / 117,964,800 mismatches**, field SHA-256
   `afeb8c94d5181b03992aefad1daef49ee7aaf1f768d11aa5964dacbfa1e22dbd`.
3. QX3's old event consumer requires historical baseline identity before every write
   (`experiments/ddm_qx3_receiver_closure.py:754-780`). QX4's changed consumer instead
   copies QBT state and unconditionally overwrites each event site with its target
   (`experiments/ddm_qx4_decodable_conditioning_reprice.py:778-818`). QXO1 proves and
   prices only this latter **target-overwrite output** contract.

Because the 17,926 retained events have 17,926 unique sites, last-writer-wins has no
collisions. Omitting a write whose target already equals QBT cannot change a byte, and
regrouping the remaining unique writes by target cannot change their final state. This
is a construction proof with executed field identity, not a summary-derived claim.

| proof population | selection | source events | target no-ops | actual overwrites | writer collisions | field mismatches / 117,964,800 |
|---|---|---:|---:|---:|---:|---:|
| sample | seeded hash-random without replacement, seed `20260901`, n=1,000, never a prefix | 1,000 | 532 | 468 | 0 | **0** |
| full confirmation | all retained events | 17,926 | 9,177 | 8,749 | 0 | **0** |

The sample index payload is retained at
`retained/semantic_proof/sample_event_indices.u32be`, 4,000 B, SHA-256
`210dc1e9a439844296c7d5dbc5f4598d11e6a0ced78375a0c3e8e332a82a0699`.
Its reference and target-implicit fields are byte-identical at SHA-256
`3452d929d95b5b174d251ac1f3c9bbc84b7ab744fe0d536243c66fb533b31fcf`.
The full target-implicit field is 117,964,800 B, SHA-256
`9079929d004cc9638a80159d61371c2982c198f0eb2b19eac4084da981ababc7`,
byte-identical to QX4's retained overwrite output. There is no counterexample.

Receipt:
`/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar/checkpoints/STAGE2_SEMANTICS_PROOF.json`,
SHA-256 `7a60472f129587062f6815ab857b84fb7d712a2c7d8b440e3611f73230369747`.

## Grammar v1 and byte table

`target_implicit_distance_rank_overwrite_v1` is a new semantic object, not a seventh
QX4 explicit-event form. For each pair and target class `0..4`, the decoder derives a
distance order from QBT, removes sites already equal to that target, reads the group
count and rank gaps, and overwrites the selected sites with the loop's target. The body
contains no per-event target byte and no historical C1 baseline byte. Target identity is
still counted through membership in a fixed group stream.

| grammar object | bytes | note |
|---|---:|---|
| fixed header | 128 | geometry, denominators, QBT/event/reference SHA bindings |
| 3,000 target-group counts | 3,000 | one ULEB count per 600 pairs x 5 fixed target slots |
| 8,749 decoder-derived distance-rank gaps | 15,431 | site identity within target-specific QBT alphabet |
| explicit per-event target labels | **0** | target is implicit in the stream slot |
| historical baseline labels | **0** | old C1-syndrome identity is not the consumer object |
| raw grammar | **18,559** | exact parse-back |
| Brotli q11 | **15,417** | selected real coder |
| LZMA-9-extreme | 15,892 | real coder |
| zlib-9 | 16,550 | real coder |

Every coder payload has a byte-identical deterministic repeat and decompresses to raw
SHA-256 `bb6c1b8626f06632ee1b3f2d6088a25d85e6d7db3c4d00b258686418b67c85ea`.
The selected Brotli payload SHA-256 is
`b0c68d2226febf336521d454fa13a9c0fa324a14d2b1cb14ab54038b89de34f2`.

Primary and repeat archives are byte-identical at SHA-256
`2487f5150fd3c38087fb5ada48d00e953c7d88a8a7219e29fbf53420657bb07f`.
Each archive independently parses the eight QXE sections, reproduces the pinned QX1
core, freshly supplies decoder-QBT conditioning, decodes exactly 8,749 overwrite
records, and produces the 117,964,800-byte reference field at SHA-256 `9079929d...abc7`.
The decoded overwrite-record payloads are also identical at SHA-256
`13e5b7419a1873c6543075d1fde4347644247fae25fc46d281450ad244cd2ee9`.

An independent read-only verification freshly re-encoded the raw grammar, reproduced
all three real-coder payloads byte-for-byte, parsed both archives, recovered all 8,749
overwrites, and rehashed all three retained full outputs to the reference SHA. The
grammar checkpoint is
`/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar/checkpoints/STAGE3_GRAMMAR.json`,
SHA-256 `45e78a5f60808ee5ca94732fee07f3bfdaf19c8fd32a3581918e14d53f987edf`.

## Decodable-conditioning re-price and complete arithmetic

| comparison | section payload | exact archive | delta |
|---|---:|---:|---:|
| QX2 C1-conditioned enumerative form bar | 22,661 B | 136,553 B | QXO1 section **-7,244 B**; QX2 baseline remains encoder-only |
| QX4 best exact-tuple form on decoded QBT | 33,435 B | 147,327 B | QXO1 **-18,018 B** |
| QXO1 target-implicit overwrite v1 | **15,417 B** | **129,309 B** | **-8,676 B vs section/archive gate** |

Exact physical arithmetic:

`113,844 B QX1 core archive + 48 B QXE section + 15,417 B payload = 129,309 B`.

The events cap is 24,093 B and the strict complete-archive condition is
`archive_bytes < 137,986`; both pass by 8,676 B. The prior-law prediction is supported:
removing 9,177 target-equal historical events and both explicit label streams recovers
18,018 B from QX4's cheapest exact-tuple form, exceeding the required 9,342 B recovery.

## Cross statement and verdict boundary

- **Held here — rate-shape:** one decoder-conditioned, exact-parse-back QX section and
  complete representation archive are under the fixed-distortion byte gate.
- **Other half — distortion:** BR2's retained born instance remains
  `DISTORTION-REFUSED` at INSTANCE scope. QXO1 does not compose its new bytes with BR2's
  measured distortion and does not claim a score.
- **What changed:** QX is no longer closed on rate representation at this pinned decoded
  field; the old full-event ABI was carrying 9,177 semantically dead writes and historical
  baseline identity the overwrite consumer does not require.
- **What did not change:** no realized renderer, Seg/Pose component, contest runtime,
  CPU/CUDA parity, exact score, or frontier pointer was measured.

Verdict scope is **INSTANCE** for the under-gate grammar row, not a distortion or public
candidate promotion. The previous QX4 verdict remains closed at **FORMULATION** scope for
its six exact historical-event forms. No qxo2 is opened inside this arm.

## RECALL EVIDENCE

Recall searched the full `.omx/research/` corpus by content, the canonical-equation
registry, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, source and SPEC/design
surfaces, hot state, task/queue/final-message ledgers, and both SSD roots. Query families
included `target-free|target overwrite|overwrite semantics|last-writer`,
`context_equals_target|target no-op|9177|17,926`, `QX event|events section`,
`decodable conditioning|receiver-produced QBT`, and `#1374|#1182`. The equation command
was `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
`decoder|condition|event|partition`.

Findings beyond the charter's named receipts changed the implementation:

1. `s2_partition_seed.py` proves global site uniqueness. That made target regrouping safe;
   without it, last-writer order would be load-bearing and the proof would fail.
2. `predictor_bound_residual.py` and QX3's historical receiver bind the old baseline and
   target semantic SHA. QXO1 therefore labels the old C1-syndrome ABI as **not preserved**
   and proves QX4's changed overwrite-output contract instead of laundering the old ABI.
3. Registered `decoder_causal_condition_transport_v1` requires every parse/CDF context to
   exist before consumption. QXO1 therefore derives QBT, distance order, target alphabet,
   and position from decoded state only; the retained S2 object is encoder-side
   verification/grammar construction input, never free receiver conditioning.
4. DCC1/LV3/CCS1 agree that only a new target-overwrite object is admissible. QXO1 prices
   one such object and never reruns QX4's six closed forms.
5. BR2 closes cross-object score composition: a byte-feasible representation cannot inherit
   another object's distortion. This changed the verdict from candidate language to the
   explicit rate-half-only boundary above.

No additional QXO-specific target-overwrite grammar or cheaper receiver-valid price was
found beyond the same-day QX/DCC/LV3/CCS1 lineage in the searched index, DAG, design,
task, source, and retained-store scopes. This is a bounded absence statement, not a
global nonexistence claim.

## Custody, reviews, and reproducibility

- Result:
  `/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar/RESULT.json`,
  SHA-256 `b7b9dd4fb1dbb70aa6dd41a32a6b998c30588103c0d2a8184d71c6ff9147a80a`.
- Run manifest:
  `/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar/RUN_MANIFEST.json`,
  SHA-256 `319b5a24e0267c5408a23f3562ea0b7dd3c0a1c57fb5a23b3dcf312d4648d3cc`.
- Runner: `experiments/ddm_qxo1_target_overwrite_grammar.py`, post-run SHA-256
  `d71be64b20083593ff9615b55d232ec9ac753b03d24941921a5b69b613cf08c0`.
- Landing dependency: the runner SHA-pins the QX2/QX3/QX4 predecessor sources. Those
  sources remain in their own verified predecessor fallback-bundle chain and are not
  absorbed into QXO1; land that chain before the QXO1 bundle.
- Command: `.venv/bin/python experiments/ddm_qxo1_target_overwrite_grammar.py
  --resume-from /Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar`.
- Retention: 30 QXO1-owned files, 708,703,108 logical bytes. Every materialized
  conditioning/proof/output field, raw grammar, coder candidate/repeat, packet,
  archive, and decoded record remains under AP custody; no cleanup fired.
- Review: two clean `review_tracker.py` passes after the final Python edit; Ruff, Python
  compilation, `git diff --check`, and
  `check_no_measure_and_discard_payload(..., strict=False)` all passed.

## Authority boundaries

- **Measured:** fresh QBT decoder identity; unique-writer/event-state census; seeded n1,000
  and full n17,926 semantic output identity; raw grammar bytes; three real-coder prices and
  deterministic repeats; exact packets/archives; exact parse-back; two independent archive
  decodes; complete archive arithmetic.
- **Not measured:** SegNet, PoseNet, distortion, contest score, public inflate runtime,
  contest-CPU/CUDA parity, or promotion eligibility. The 129,309-byte archive is a
  representation receipt, not a submit-ready contest archive.
- `upstream/` remained read-only. Scorer jobs, Modal calls, Metal jobs, training runs, and
  remote dispatches were all zero.

## NEXT_IF_RESUMED

- **Disposition: `QUEUED-WITH-A-FIRE-ORDER`; owner: MAIN n600 scorer-realization scheduler; consumer store: `/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar/RESULT.json`; fire trigger: MAIN binds exact archive SHA `2487f515...bb07f` and decoded field SHA `9079929d...abc7` to the retained born-object realization path under BR2's payload-retaining n600 protocol, preserving every grammar/core hash and explicitly carrying BR2's standing distortion refusal.** Measure the realized QX object's Seg/Pose components; do not compose QXO1 bytes with BR2 distortion or claim a score before that same-object binding.

## LIVE-HYPOTHESES

- The under-gate QX overwrite field may realize differently from BR2's exact 106,832-byte
  born instance because QXO1 applies 8,749 retained semantic mutations absent from the raw
  QBT baseline. This is plausible enough to justify the typed same-object scorer fire, but
  BR2 makes the prior strongly adverse and no distortion transfer is allowed.
- Jointly shaping the QBT core for overwrite-site distance alphabets may reduce the 15,417 B
  section further. It is plausible because 15,431 of 18,431 raw body bytes are rank gaps,
  but it is a changed core with new model/latent bytes and distortion, not a qxo2 tuning row.

## DEAD-ENDS

- Preserving all 17,926 historical C1-syndrome tuples is closed on decoded QBT: QX4's best
  form costs 33,435 B and produces a 147,327 B archive. Do not rerun its six forms unchanged.
- Treating target labels as free is closed: QXO1 removes explicit label bytes but counts
  target information through fixed stream membership. Any claim of zero target information
  would be fake.
- Reordering would be inadmissible with duplicate writers. The retained population has
  0 / 17,926 collisions; successors must re-prove this if the event source changes.
- QX2's 22,661 B C1-conditioned section is not a receiver-valid price for QBT; its hidden
  baseline remains closed by QX3's 510,404 B bridge.
- BR2's distortion cannot be combined with QXO1's 129,309 B archive. A cross-object score is
  closed; only a same-object retained realization can open the score question.
- This arm opens no qxo2 and fires no scorer itself. The sole remaining action is the typed
  MAIN-owned same-object realization above.

Own-vehicle frontier: **AFR1 S `0.14797617125559104` @ `180,002 B` `[contest-CUDA T4 n600]`, archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` — UNMOVED.**
