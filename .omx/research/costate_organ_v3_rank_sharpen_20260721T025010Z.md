# Costate organ v3 rank-sharpening result

UTC: 2026-07-21T02:50:10Z

Lane: `lane_costate_v3_rank_sharpen_20260721`

Axis: `[macOS-CPU advisory]`

Scope: `RETROSPECTIVE_DEVELOPMENT`, fixed n=24 only

Authority SHA-256: `d3ec1543d1c656f506d2c4817771795ca8cb9ac738560d76c7be6f536d6eb2a9`

## Verdict

**NO ADMISSION / NO PROMOTION.** The zero-parameter v3 organ is implemented,
typed, and receipt-wired, but it does not demonstrate rank improvement over v2
on this sealed corpus. Final ordinary Spearman moved from `0.6978255654` to
`0.7033293051` (`delta=+0.0055037397`, 95% CI
`[-0.1508454963, +0.1841447773]`), while decision NDCG@8 moved from
`0.6493498443` to `0.6178159241` (`delta=-0.0315339202`, 95% CI
`[-0.2372876029, +0.0092573247]`). Both are inside bootstrap noise and their
point directions disagree. Top-8 positive-benefit precision stayed `1.0`
(`delta=0`, 95% CI `[0, 0.125]`).

The apparatus-variance weighted Spearman is `0.7254293768`, versus the v2
unweighted/identity-weight baseline `0.6978255654` (`delta=+0.0276038114`, 95%
CI `[-0.1568404639, +0.1972046584]`). This is also inside noise and is not an
improvement claim.

## Additive readback

| Stage | Spearman | weighted Spearman | top8 precision | NDCG@8 | tie pairs |
|---|---:|---:|---:|---:|---:|
| v2 baseline | 0.697826 | 0.697826 | 1.000 | 0.649350 | 105 |
| graded realizability | 0.707803 | 0.707803 | 1.000 | 0.649350 | 42 |
| pool interaction | 0.687223 | 0.687223 | 1.000 | 0.623846 | 42 |
| EMA target/variance weighting | 0.703329 | 0.725429 | 1.000 | 0.617816 | 42 |
| typed receipt emission | 0.703329 | 0.725429 | 1.000 | 0.617816 | 42 |

Graded realizability breaks structural ties but its Spearman delta
`+0.0099778447` has 95% CI `[-0.1339548720, +0.1558007503]`. Pool competition
covers 4 shared pools / 13 unique rows, but the staged Spearman delta is
`-0.0205808830` with CI `[-0.0752955067, 0]`. Pool-to-target Spearman improves
`+0.0161067780` inside noise, whereas NDCG@8 degrades `-0.0060301659` outside
zero with CI `[-0.0123911982, -0.0001236525]`. Therefore the pool formulation
is retained as measured advisory structure, not admitted as a decision rule.

## Evidence custody

- v2 receipt SHA-256:
  `f733187fbc8e69e03d4854a8f45baa0951af4a2807a01bfc1841cffca8d59410`
- r1b7 stage-autopsy receipt SHA-256:
  `61f3d03930ac765b3ad5a287cbff29a3073c800eb5a5f2b98b8a701bc086d03c`
- v3 machine receipt SHA-256:
  `450ce34a34b6d3d549eb7a29c871a456bfcba243335315cbbc00adb5b5b82c23`
- canonical 24-row corpus SHA-256:
  `db101ed2ecef2f3d5570e8b33f94df9fb75fdd1bc20fa97c7aa834ffbfbcd06c`
- bootstrap: 10,000 paired, stratum-preserving replicates; deterministic
  registered seed schedule.
- identical command replay reproduced the machine receipt and corpus SHA-256
  byte-for-byte.
- `witness_measured_reverse_waterfill_v1` is present in the canonical equation
  registry. `ema_decay_run_geometry_v1` has an executable evaluator and typed
  DSL consumer, but no canonical-equations JSONL row; no row was fabricated.

## Receipt and actuation boundary

The append-only typed corpus was materialized with identity rows for the exact
v2 snapshot: 24 requested, 24 present/reloaded, IDs identical. Every seeded
row has zero byte delta, so this is backtest custody, not byte-price evidence.
The M1 producer hook only appends a future first byte-paying row when its
repo-relative receipt has a fully realized `costate_realized_delta` block,
nonzero byte delta, source SHA custody, and byte-close authority. The current
M1 dry manifest remains `NOT_EMITTED`; no fake row was appended.

Actuation `NONE`; learned parameters `0`; pointer unchanged; no score claim;
no live-run, provider, archive, scorer, or trainer mutation.

## STORES CONSULTED

- delegated authority file and SHA custody
- `CLAUDE.md`, `AGENTS.md`, top-10 Claude memory entries
- latest Codex findings/session and Claude design/council surfaces selected by
  the session preflight
- `.omx/state/lane_registry.json`, `subagent_progress.jsonl`, inbox broadcast
- v2 receipt, r1b7 autopsy receipt, canonical-equations registry

## Remaining blockers and MAIN review

1. The corpus has no realized nonzero-byte row; byte-price rank quality remains
   untested.
2. The fixed n=24 retrospective corpus is too small for an admission claim.
3. NDCG and Spearman disagree, and pool interaction is adverse on NDCG.
4. MAIN must inspect the prediction/target separation, source SHA guards,
   corpus lock/idempotency, and M1 fail-closed hook before landing.
5. Global `lane_maturity.py validate` still reports 110 pre-existing missing
   evidence paths outside this lane; this lane's own L1 record is internally
   consistent and deliberately does not claim a real-archive gate.

Verdict scope is this exact v3 formulation and sealed 24-row corpus only; it is
not a negative verdict on costate organs, pool-aware allocation, or EMA-aware
target estimation as families.
