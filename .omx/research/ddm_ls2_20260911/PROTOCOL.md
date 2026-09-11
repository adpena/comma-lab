# LS2 predeclaration

Axis: [macOS-CPU advisory / scorer-free full-resolution categorical probability measurement].
No scorer, training, candidate archive, or paid dispatch. Fixed move44 archive, shipped integer probability rows,
decoded field, all inherited model bytes and adaptive state. Read LS1 rows/cells/checkpoints; never modify them.

## Finite models (frozen before fitting)
Use log2_fixed Q10 clipped log2((count+0.5)/(expected_mass/2^31+0.5)) from completed earlier planes only.
Each count bank is crossed with the SHIPPED integer-row winning class, uses all five outcomes, and is frozen
for the entire next plane. Expected masses are the SHIPPED integer frequencies, independent of added weights.
Receiver context codes are recovered exactly from LS1 receiver_lane keys: past=(key//8)%8, visible=key%8.
Phase is the exact public HPAC group x%64+2*(y%64), 190 possibilities; no phase fit or free selector.
Separate: [past distance, causal preceding-row distance, phase], 3 coefficients per winner = 15 signed int8.
Joint: [8*past+visible, phase], 2 coefficients per winner = 10 signed int8.
Each weight is int8/32, box [-4,127/32]. Header 8 bytes: magic LS2P, version=1, family=0/1,
scale_log2=5, selected_class=1. The selected Lane class is counted, not embedded learned code.
Physical parameter charges: separate 23 bytes, joint 18 bytes; no empirical count table is transmitted.
These are raw added parameter payload costs; no archive size or coder framing saving is asserted.
At zero correction return exact original integer frequencies, with no float round trip.
Nonzero: Q10 features times int8 weights, TC1 deterministic power-table normalization, float32 then
actual shipped RC64 frequencies conversion. All baseline model/config bytes remain counted and unchanged.

## Fit, bound, validation
Seed 20260911. Uniform random 120 of 600 pair IDs are held out before fitting; remaining 480 fit weights.
Held-out scores assess fixed-weight prediction under ordinary causal online counts. Earlier held-out
outcomes may enter later causal count banks; they never enter the coefficient-fit objective. This is
prequential receiver-state validation, not independent untouched-label statistical generalization.
Then separately fit full600 for achievable whole-object pricing. No architecture or subset selection sweep.
Reuse TC1 7-feature C categorical objective padded with zero features; only active coordinates optimize.
Retain all trial and accepted vectors, rounded parameters, per-stage state and per-pair price arrays.
Fit KEEP=wrong winner or miss mass>=2^-14. Supporting-plane lower bound on retained loss plus zero
loss for omitted symbols; explicit omitted baseline loss credit. Numerical float64 certificate, not interval
proof. All rounded final models are priced on EVERY symbol through real RC64 conversion. Report total,
class and row-mod64==0 vs other rows, plus per-row and pair results. Pair fit/held-out comparisons report
raw totals and per-pair rates without hiding different denominators. In-sample full fit is labeled biased.
No verdict from a reduced smoke. No transfer of HC/MI/TC prior byte magnitudes.

## Stop and handoff
No positive exact-parameter-charged integer log-loss reduction in either family: FORMULATION refusal.
Positive reduction without native timing admission: QUEUED-WITH-A-FIRE-ORDER for MAIN timing feasibility
before one recode; never automatic build. Recode authorization remains MAIN's separate build charter.
Re-derive 0.12 rate demand from live exact components. Numerical bound alone never fires candidate.

## Retention and resume
Owned SSD tree /Volumes/VertigoDataTier/pact/ddm_ls2_full_resolution_lane_probability_bound.
16 GiB free reserve plus projected remaining output required. Atomic per-frame feature/checkpoint payloads;
fit resume from last accepted iterate, all trials retained. Existing artifacts hash checked before reuse.
Automatic hygiene certifies every new retained artifact at stage completion and blocks deletion/move;
all produced bulk is measurement evidence, so no success cleanup is justified. No temporary raw renders.

<!-- # FORMALIZATION_PENDING: measurement memo; no score row -->
