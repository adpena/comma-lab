# RXC1 — restartable exact HPAC/RC64 state on AFR1

**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]`  
**Verdict scope:** `INSTANCE` — AFR1 archive `cbb8d928…d405bf25`, its 600-pair token field,
the shipped HPAC/corrector, and the native RC64 encoder pinned below.  
**Current execution state:** Stages 0 and 1 are complete. The preregistered Stage-2 n=32 screen is
running from retained state. No correlation verdict is claimed before `SCREEN.json` exists.

## Result first

| Gate | Denominator | Result | Receipt |
|---|---:|---:|---|
| full AFR1 control | 117,964,800 tokens; 600 pairs; 113,411 stream bytes | **PASS**, 0 differing bytes; 716.436 s | `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/BASELINE.json` |
| pair-boundary state | frames 0, 200, 300, 400, 600 | full HPAC/corrector state plus RC64 prefix/interval retained | same baseline receipt |
| null replay | 5 checkpoint starts; 567,055 compared stream bytes | **PASS**, 0 differing bytes | `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/NULL_REPLAY.json` |
| exact edit screen | 32 seeded pairs in 8 index strata; 2 strides | **RUNNING**; no aggregate verdict yet | `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/SCREEN.json` when complete |

The full control regenerated the exact shipped 113,411-byte token stream, SHA-256
`5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3`, and the exact
180,002-byte archive, SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.

Null replays were byte-identical at every configured start:

| stride | start frame | replayed pairs | measured wall seconds | differing bytes |
|---:|---:|---:|---:|---:|
| 200 | 0 | 600 | 716.436 | 0 |
| 200 | 200 | 400 | 475.067 | 0 |
| 200 | 400 | 200 | 237.690 | 0 |
| 300 | 0 | 600 | 716.436 | 0 |
| 300 | 300 | 300 | 355.997 | 0 |

The zero-start rows reuse the one already executed full control; they are not duplicate encodes.
The measured suffix costs scale approximately with replayed pairs, while all comparisons cover the
entire 113,411-byte emitted stream.

## What was built

`experiments/ddm_jg2_tail_reencode.py` remains the only physical coder mirror. RXC1 extends it with:

- immutable pair-boundary bundles containing the previous decoded plane, the complete per-frame
  ledger, every structurally discovered mutable corrector/mixer/family array, and the native RC64
  snapshot containing emitted prefix bytes and interval state;
- a fail-closed divergent-state detector, so a mutable corrector attribute cannot move away from a
  cold instance without appearing in the checkpoint;
- immutable stream, ledger, state, archive, and run-spec persistence; and
- a lazy retained replacement-plane overlay, so a 117,964,800-byte disposable candidate field is
  never materialized.

`experiments/ddm_rxc1_restartable_exact_coder.py` exposes the outer-loop boundary:

```python
api.exact_delta(edit_path, pair, stride, run_dir)
```

The API resolves the nearest checkpoint at or before the edited pair, rejects edits before the
restart boundary, restores the complete JG2/RC64 state, performs the real suffix encode, packs the
physical archive, and returns the exact archive-byte delta. Every run retains its edit payload,
stream, archive, bit ledger, terminal state, source-bound run spec, and receipt. An interrupted run
resumes from its newest immutable frame-300 checkpoint; a completed receipt is reused only after
all artifact hashes and the AFR1/source bindings validate.

## Pins and custody

| Object | Bytes | SHA-256 |
|---|---:|---|
| AFR1 `archive.zip` | 180,002 | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| AFR1 HPAC section | 13,515 | `602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98` |
| AFR1 token stream | 113,411 | `5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3` |
| token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| native route-B wrapper | 14,138 | `c2d9759a77e793d643ca1d4a557934cdb66f39473b244f382dd9f0b8faaf89e5` |
| RC64 base C source | — | `5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6` |
| executed RXC1 source | 37,869 | `115bb907520b0996bca6e3c00be34ed341266185ac3802a05cba5efe628f76e9` |
| executed JG2 source | 58,160 | `e762bead28ab981980aa64161e9104bf1ef5e61c450888edf7777a550c3ac70d` |

All materialized payloads are retained under
`/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`. The preflight records free-space,
input/runtime/source hashes, the seed and strata, both strides, and the correlation thresholds.
The final manifest will enumerate every retained file by relative path, bytes, and SHA-256. Nothing
in this arm writes `upstream/`, calls a scorer, spends money, or dispatches Modal.

## RECALL EVIDENCE

The bounded full-corpus recall searched `.omx/research/` memos and arm-final messages,
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, design/spec documents, `.omx/state/main_hot_state.md`,
task/dispatch ledgers, experiments/runtime source, and the canonical-equations registry. Content
queries included `restartable`, `checkpoint RC64`, `range state`, `incremental exact`, `halo-0`,
`direction-dependent`, `average marginal`, `reorder context schedule`, `HPAC RC64`, `SCMDL`,
`JG2`, `FS2`, `FS3`, `RR9`, `RX1`, and `CP135`.

Beyond the charter seeds, recall found two implementation precedents and one architectural guard:

1. `ddm_cp135_rate_compose_20260810.md` and `ddm_rx1_rate_representation_attack_20260814.md`
   already carried a checkpointable RC64 interval/prefix surface and direct-vs-resumed identity,
   but their probability-frame checkpoints did not constitute the current AFR1 HPAC/corrector
   state. This changed the plan from inventing a new coder to reusing the pinned route-B native
   extension and adding the missing structural corrector snapshot to JG2.
2. `ddm_rr9_reorder_refit_20260824.md` measured a fixed within-group reorder at 0 bytes and proved
   that cross-group schedule changes alter the trained causal mask rather than merely permuting a
   fixed code. This made full mixer/family/context capture mandatory and forbade a schedule-neutral
   checkpoint proxy.
3. The registered equations `token_rate_model_direction_dependence_v1` and
   `greedy_set_average_vs_marginal_price_v1`, anchored by FS2/FS3, say that modeled token prices are
   direction-dependent and that a selected set's average price can miss its marginal price by
   2.24x. This changed Stage 2 into retained physical archive re-encodes for every candidate; no
   entropy, `-log2 p`, static bank, or differentiable surrogate is accepted as the delta.

The search did not find, in those scopes, an existing AFR1 checkpoint bundle that captured both all
mutable free-corrector/mixer family state and the RC64 interval/prefix, nor an n>=32 AFR1
incremental-vs-full exact screen. That is a bounded absence, not a global nonexistence claim.

## Boundaries

Measured here: input/source identity; one full n600 physical encode; five full-stream null
comparisons across two checkpoint strides; checkpoint state size and replay wall time; retained
one-cell edit payloads and exact archives for completed Stage-2 rows.

Not measured here: `d_seg`, `d_pose`, any scorer component, exact contest score, CUDA behavior,
candidate realization, a learned SCMDL proposal, or an outer-loop search. This gate changes no
field/model/schedule and cannot move the frontier. Gate 2 remains MAIN-scheduled even if this exact
coder gate passes.

## Own-vehicle frontier

Pointer UNMOVED: AFR1 remains `S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]`.
RXC1 is `[macOS-CPU advisory / scorer-free EXACT byte measurement]` and makes no score claim.

