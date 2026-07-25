# Papers-checked — arXiv 2607.18236 "Patch Policy: Efficient Embodied Control via Dense Visual Representations" (Zhou, Cui, Langford, Tan, LeCun, Pinto)

UTC: 2026-07-24 · Harvested by: MAIN (Fable, inline per operator "Fable to harvest") · $0
Evidence class: MEASURED_EXTERNAL (their benchmarks, never our contest axis). Lessons-only —
no external vehicle adopted (no-old-lineage discipline extends to external architectures).

## What the paper measures

Transformer robot policies consuming DENSE frozen-ViT patch tokens directly (no VLM) via a
block-causal attention mask (causal across time steps, full attention within each observation's
patch set + state tokens). Headline external numbers: **+40% relative over global-pooled
representations** (7 environment suites); **beats fine-tuned OpenVLA-OFT by 18% at ~0.7% of its
parameters**.

## Crosswalk vs live surfaces (4 rows)

| # | Their lesson | Our surface | Disposition |
|---|---|---|---|
| 1 | Frozen pretrained encoder's DENSE patch features + tiny head ≫ global-pooled features (+40% rel); never fine-tune the backbone, never pool | **co4 Road-local reformulation gap** (FEED-603-co4): the two failed ridge forms ranked Road pairs from bucket-AGGREGATE atlas features — the "global-pooled" analog. Our frozen SegNet IS the pretrained encoder; its per-cell intermediate features (pre-SE locus #484 · at1 atlas · sn1 telemetry) ARE the dense patch tokens | **ADOPT-AS-REFORMULATION-SHAPE**: the post-J8F Road-local rerun should be a TINY head over DENSE frozen per-cell SegNet features (+ J8F realized telemetry as target), not a third ridge over aggregates. Amends the FEED-603-co4 candidate; stays gated on J8F |
| 2 | 0.7% of params beats the fine-tuned giant | Train-least / surgical-Kolmogorov-projection doctrine; #366 re-aim = 706-param lift through frozen scorers | **CORROBORATION** (external, independent). No action |
| 3 | Block-causal mask: causal across time, dense within observation | Our temporal axis: causal across pairs, dense within pair (ξ-keyed temporal coder #574 · worldsheet event stream); cc1 race 3 (context-model race) | **NOTE-FOR-RACE**: if a learned context model ever wins cc1's J8F-gated race, block-causal over per-pair token sets is the architecture shape (COUNTED params, same gate) |
| 4 | Pooling loses 40% relative — a measured price of aggregation in an adjacent domain | g3 currency law (flip/margin-weighted, never pooled L2) · decompose-every-headline | **CORROBORATION**. No action |

## Verdict

`LESSONS_HARVESTED_INLINE; ONE_REFORMULATION_SHAPE_ADOPTED (Road-local, post-J8F); NO_ARM_SPAWNED`.
Coherence-contract check: NOVELTY — the dense-vs-pooled quantification is new to us (we had the
doctrine, not the external price); DERIVATION — row 1 maps to co4's NAMED gap, not a hunt;
DISTANCE — rows 1/3 join the existing post-J8F wave (no new critical path). Pointer
0.1910828242 [contest-CPU] UNMOVED — this is means.

STORES CONSULTED: co4 receipt + findings memo (FEED-603-co4), cc1 FEED (race 3), #484 pre-SE
locus, at1/sn1 rows, train-least memory, g3 currency law, papers_checked_* precedent memos.
