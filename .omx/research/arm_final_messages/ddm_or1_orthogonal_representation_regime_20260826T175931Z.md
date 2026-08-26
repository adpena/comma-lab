Executed and canonically closed OR1. The main conclusion is that the orthogonal remainder is thin: generic standalone representations do not beat the current token coders, so near-term weight belongs on exact factorization and field-for-coder work.

| Exact n600 packet | Bytes | vs GB1 tokens | vs D3 tokens |
|---|---:|---:|---:|
| D3 reference F26/HPAC | 49,696 | −63,928 | — |
| GB1 reference F26/HPAC | 113,624 | — | +63,928 |
| SP1 boundary/support | 140,377 | +26,753 | +90,681 |
| Dense RAW-LZMA2 | 200,943 | +87,319 | +151,247 |
| Temporal-XOR RAW-LZMA2 | 259,617 | +145,993 | +209,921 |

All three OR1 packets parse back exactly and their determinism repeats are byte-identical. They remain standalone screens, not shipped F26/HPAC reference-form candidates. No scorer slot was consumed, no archive was mutated, and the pointer remains GB1 at `S=0.14811799921260607`, 180,215 B.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_or1_orthogonal_representation_regime_20260826.md:1)
- [Measurement source](/Users/adpena/Projects/pact/experiments/ddm_or1_orthogonal_representation_sweep.py:1)
- [Retained result](/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/RESULT.json), SHA-256 `ed1763ed…91ab`
- [Payload manifest](/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/PAYLOAD_MANIFEST.json), SHA-256 `61fe5f95…b56f`

Commits: `e493c73054` and closure receipt `5599b0e28b`. Ruff, Python compilation, repository hooks, SHA checks, decode identity, and canonical-ledger validation passed. The ledger reports 673 served rows; its two pre-existing unreadable-history warnings are unrelated to OR1. The unrelated untracked D3B experiment was preserved untouched.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN renderer-training successor`; consumer store: `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/next_renderer_born_small/`; fire trigger: a deterministic, from-scratch smaller renderer retains every stage checkpoint and learned payload, integrates with the shipped receiver, earns at least 2,000 B complete-archive credit, and passes a seeded random `n>=32` realized-through-R Seg/Pose gate before requesting n600 scoring.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN boundary-grammar successor`; consumer store: `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/global_region_grammar_reference/`; fire trigger: a reference-form global topology grammar for source SHA `deafcb2f…` retains a double-decode-exact payload no larger than 47,696 B, with every video-derived byte counted.

## LIVE-HYPOTHESES

- Optimal four-class conditioning remains the strongest immediate factorization lead: D3B’s first receiver-closed candidate misses GB1 by only 360 B, and Amendment 2 establishes that the four-class model has not received optimal treatment.
- A global topology grammar may beat local row starts because OR1 paid for 352,297 local transitions and reset adaptation every 25 frames. It is viable only if it clears D3’s much tighter 49,696 B reference-form bar.
- A genuinely trained smaller renderer could relocate score-relevant capacity before quantization. Frozen-weight recoding does not test this mechanism, though prior renderer failures make its prior low.
- GB1’s roughly 1,240-second T4 margin can support richer generic prediction. This is only a compute enabler; learned or video-derived state remains counted.

## DEAD-ENDS

- Do not allocate token bytes to frame 0: the shipped receiver has no frame-0 token field.
- Do not repeat frozen frame-0 carrier coarsening, rank reduction, refitting, or repainting: the cheapest measured purchase worsens score by approximately `+0.3045`.
- Do not promote the OR1 boundary, dense, or temporal packets: each is exact but loses decisively to both GB1 and D3.
- Do not replay D3B’s already-tested field, geometry, or temporal configurations unchanged, or D3C’s source-replay pyramid: their retained payloads lose.
- Do not reuse BR1’s `cell_drop50` knee or JG5’s 330-second headroom as current GB1 facts: both belong to superseded objects.
- Do not re-race current coder or packaging defaults: JT23 found zero collectible coder bytes, and ZIP/RX1 framing is already at its measured floor.