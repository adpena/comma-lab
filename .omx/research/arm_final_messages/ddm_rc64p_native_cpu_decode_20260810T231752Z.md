Executed and landed in commit `c6c4dd4ae5`. Full evidence is in [ddm_rc64p_native_cpu_decode_20260810.md](/Users/adpena/Projects/pact/.omx/research/ddm_rc64p_native_cpu_decode_20260810.md).

Measured `[macOS-CPU, scorer-free, n600]`:

| Receiver | 1 thread | 4 threads |
|---|---:|---:|
| Constriction ANS control | 1,383.08s | 635.94s |
| Native C ANS | 1,477.08s | 647.08s |
| RC64 | 1,461.83s | 812.56s |
| RC64 + cached HPAC | 1,437.42s | 716.94s |

All cells reconstructed 117,964,800 symbols exactly and ended with empty coder state. Entropy calls consumed only 1.11–3.13 seconds; causal HPAC probability generation is the actual wall.

Route B is a real rate win:

- Archive: 187,222 bytes, four bytes below lc2.
- SHA-256: `b3365410a423fa6ae4d53e9a86fc2bd38bc59793ea2b437fc161bdcca11712b0`
- Literal parse-back: 3,662,409,600 bytes, canonical SHA `a18eb42a…`
- Fresh ANS/RC64 compiler bootstrap: 0.59 seconds.
- Rate-only projected score: `0.16959633225649604`.

That projected score is not promoted: no scorer, `upstream/evaluate.py`, Modal, or contest-CPU authority ran. Timing cells also lack repeats and showed host drift, so no stable cross-cell speed ordering is claimed.

Verification: 7 focused tests, payload-retention gate, Python compilation, JSON/whitespace checks, two review passes per Python file, serializer post-commit verification.

## NEXT_IF_RESUMED

- **QUEUED** — owner: MAIN; consumer store: contest-CPU feasibility ledger; fire trigger: claim the single-flight lane and authorize an exact contest-CPU replay of archive `b3365410…`.
- **QUEUED** — owner: MAIN; consumer store: canonical frontier pointer; fire trigger: scorer lane is free and the exact Route-B runtime bundle is admitted.
- **QUEUED** — owner: successor runtime arm; consumer store: `/Volumes/VertigoDataTier/pact/ddm_rc64p_20260810/receipts/`; fire trigger: contest CPU misses 1,800 seconds or MAIN explicitly prioritizes additional CPU headroom.

## LIVE-HYPOTHESES

- A fused C/Rust sparse-HPAC kernel may still pay because over 99.5% of token wall remains outside entropy coding.
- Direct gathered-one-hot construction may remove repeated whole-frame tensor materialization; the tested cache-only formulation did not address that work.
- Route B should score exactly four bytes better than lc2 because literal raw output is identical; an exact evaluator run is still required for pointer authority.

## DEAD-ENDS

- Native ANS as the CPU cure: closed for this instance because symbol recovery was already about one second and native calls did not reduce wall time.
- Cached immutable HPAC plans: closed for this implementation because both full timing axes failed to beat the constriction control.
- APDataStore for growing progress checkpoints: closed after reproducible open stalls; Vertigo checkpoints completed normally.

Own-vehicle frontier remains **lc2 S 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.