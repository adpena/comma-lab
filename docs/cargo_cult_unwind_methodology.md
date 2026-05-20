# Cargo-cult unwind methodology

A reusable engineering discipline for rescuing substrate paradigms that fail their first empirical anchor by an order of magnitude or more, without prematurely killing the paradigm.

Validated empirically on one substrate refactoring (NSCS06 v6 → v7): a `0.4413` reduction on the contest-CUDA T4 axis (`105.15` → `58.89`, 44% improvement) in one iteration. Promising as a general recipe; not yet universally proven.

Related: [`docs/asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md) Section C.10. Sister library: [`adpena/tac`](https://github.com/adpena/tac). Submission packet: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

---

## What it is

Most substrate paradigms (predictive-coding, cooperative-receiver, information-bottleneck, hierarchical-prior, composition-stacking) inherit a stack of implicit design assumptions from their canonical reference paper. When the paradigm is ported to a new domain (the comma video compression contest scorer), some of those assumptions transfer cleanly and some do not. The ones that do not transfer are the cargo-culted assumptions: copied along with the canonical formulation because they were structurally inseparable in the original paper, but never tested in the new domain.

Cargo-cult unwind is the systematic procedure of:

1. Enumerating every implicit assumption in the substrate's current implementation.
2. Per assumption, classifying it as **hard-earned** (empirically verified or derived from first principles in the new domain) or **cargo-culted** (inherited from the canonical reference without verification).
3. Per cargo-culted assumption, designing and running an unwind test that either falsifies the assumption (forcing a redesign at that layer) or promotes it to hard-earned.

The lineage of this discipline traces to interpretable-ML practice — Wang & Rudin's *Falling Rule Lists* (AISTATS 2015) is the canonical reference for per-assumption-classification rigor, and Rudin's *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions* (Nat Mach Intell, 2019) is the canonical reference for why structural extinction of CARGO-CULTED assumptions beats post-hoc explanation of opaque failures.

## The empirical anchor

NSCS06 (Carmack-Hotz Strip-Everything composition substrate) v6 returned `105.15 [contest-CUDA T4]` against a design-time predicted band of `[0.10, 0.20]` — `553x outside` the band. A naive verdict would kill the paradigm: composition substrates do not work at this scale, return to within-class bolt-ons.

Instead, the v6 → v7 audit enumerated seven implicit assumptions in the v6 composition stack (per-class chroma anchor, grayscale-LUT replication, np.roll global translation, arithmetic-coded delta, N=1 sample composition, fixed Huffman selector indices, integer-bit allocation via closed-form CDF). Four classified as cargo-culted (per-class chroma anchor unwound via Daubechies-style per-class scaling; grayscale-LUT replication unwound via per-channel chroma preservation; np.roll global translation unwound via per-pair pose-relative translation; closed-form CDF allocation unwound via empirical bit-spend measurement). Three classified as hard-earned (arithmetic-coded delta, N=1 composition, fixed Huffman) and preserved.

v7 implementation against the four unwinds landed at `58.89 [contest-CUDA T4]` — a `0.4413` reduction in one iteration. The paradigm was intact; v6's failure was implementation-level falsification at the cargo-culted assumption layer, not paradigm-level falsification.

## The general recipe

1. **Inventory the implementation's assumptions.** Walk the substrate's code top-down. Every magic number, every architectural choice copied from a reference paper, every "the standard way" comment, every loss-function term — each is an assumption candidate. Aim for 5–15 assumptions per substrate.

2. **Classify each per the hard-earned-vs-cargo-culted contract.** Per assumption, ask: *was this empirically verified in the contest domain (or derived from first principles applicable here)? Or was it copied from the canonical reference because that is what the paper did?* The Assumption-Adversary council role (a dedicated reviewer whose only job is challenging the framing) is structurally protective at this step.

3. **Per cargo-culted assumption, design an unwind test.** The unwind test does not have to be a full substrate retrain. It can be a smoke probe that swaps the cargo-culted choice for the alternative and checks whether the rate / distortion / score moves in the expected direction.

4. **Iterate.** Apply the unwinds that the smoke probes validate. Re-run the full substrate. Compare against the design-time predicted band. If still outside, run a second round of unwinds on assumptions that the first round did not touch.

5. **Promote unwound assumptions to hard-earned.** Record the empirical anchor that promoted each cargo-culted assumption. The classification is monotonic: once hard-earned, the assumption does not require re-unwinding on the next iteration.

## Where it does NOT apply

Cargo-cult unwind is a paradigm-rescue methodology. It is the right discipline when an empirical anchor falsifies a substrate by an order of magnitude or more AND the paradigm itself remains plausible. It is the wrong discipline when:

- The empirical anchor is within the design-time predicted band (no rescue needed).
- The empirical anchor is moderately outside the band (1.5x – 3x) — that range is typically calibration drift, addressable by hyperparameter sweep rather than structural reclassification.
- The paradigm itself is structurally incompatible with the contest scorer's constraints (e.g., a substrate that fundamentally cannot export into the contest archive grammar; no amount of cargo-cult unwinding will rescue the export contract).

## Internal tooling that operationalizes it

- **Canonical equations registry.** A small append-only ledger of empirically-calibrated equations (Brotli cascade bounded per stream, per-pair master-gradient Taylor + Cauchy-Schwarz bound, MPS-vs-CUDA drift architecture-class dependent, and three sisters) that codify hard-earned anchors as first-class artifacts rather than tribal knowledge. See [`canonical_equations_tour.md`](canonical_equations_tour.md).
- **Per-deliberation explicit assumption surfacing.** The council pact for any non-trivial design decision requires every member to explicitly state the assumption they are operating within before the council reaches consensus. The Assumption-Adversary role then classifies each surfaced assumption against the hard-earned-vs-cargo-culted contract. This makes the inventory step (1 above) structural rather than discretionary.

## Honest scope

The discipline is engineering rigor for the substrate-design loop, not a contest-score primitive by itself. It does not produce score reductions directly; it reduces the cost of paradigm-class research by extinguishing the failure mode where a single bad implementation gets misread as a paradigm kill. The 44% improvement on NSCS06 v6 → v7 is one data point; broader generalization is open.
