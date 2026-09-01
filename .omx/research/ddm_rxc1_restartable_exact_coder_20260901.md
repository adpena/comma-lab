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

## Gen-2 Stage-2 harvest blocker — 2026-09-01

### Result first

**BLOCKED(storage-reserve), verdict scope `INSTANCE`.** The frozen Stage-2 screen advanced from
5 to **26 sealed rows out of 32** before the shared APDataStore tier lost enough free space that
finishing would violate the required 1 GiB post-run reserve. The process was interrupted during
row 26 / pair 470's stride-200 suffix replay, after its full exact result and terminal state had
been retained. No payload was deleted, moved, or discarded.

`SCREEN.json` and `MANIFEST.json` do **not** exist. Therefore there is no completed n=32 gate
denominator, no final branch adjudication, and no `GATE-1-PARTIAL` claim. Gate 2 did not fire.
The machine-readable blocker is
`/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/BLOCKER.json`, 10,064 bytes, SHA-256
`581a076846dfdba0164ff5b6ab4c4818258eaa61b2591ab010e27e97885d839b`.

The binding preregistration says:

> If SCREEN.json's incremental leg is an exact suffix re-encode (deltas byte-identical to full):
> the correlation row is recorded as VACUOUS-BY-CONSTRUCTION and carries NO gate authority.
> Gate-1 is then adjudicated on the COST criterion alone.

The 26 sealed rows have that **Branch-1 shape**, but this is partial evidence only because the
named `SCREEN.json` denominator is absent. Their exact-vs-exact correlation is consequently
vacuous and is not used as gate evidence.

### Partial denominator and cost — not a gate verdict

The sealed partial denominator is 26 seeded pairs from the frozen n=32, seed-20,260,901 sample;
one retained token edit per pair; two restart strides; 52 full-stream byte comparisons. All
52/52 incremental streams are byte-identical to their row's full exact stream, archive-delta
error is 0 bytes, and sign agreement is 26/26 at each stride. Full exact archive deltas span
the integer set `{1, 2, 3, 4, 5}` bytes.

| stride | sealed n | identical | median s/proposal | mean s/proposal | median frames | physical restarts | physical-restart median s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 26 | 26/26 | **478.57277050009** | 540.7383064968654 | 400 | 15 | 476.97995079215616 |
| 300 | 26 | 26/26 | 714.2794453538954 | 580.2944720274268 | 600 | 10 | 366.5510804587975 |

The partial best aggregate row is stride 200 at 478.573 s/proposal. The table includes rows
whose nearest checkpoint is frame 0 and therefore reuses the row's full exact result, exactly as
the preregistered screen implementation defines. The physical-restart column separately exposes
the nonzero restart denominator. These costs already have the expected minutes-per-proposal
shape, but the missing six rows forbid the chartered final cost adjudication.

### Cheap Branch-2 state-reconvergence measurement — partial scope

At terminal frame 600, **0/26** sealed full-edit rows exactly reconverged to the baseline adaptive
state across the 147 registered non-ledger arrays. Per row, 57–71 arrays still differed, median
69; the final `previous` plane was equal in 26/26. This measured scope is only the sealed n=26
AFR1 instance at frame 600. It supports the narrower statement that the current adaptive state
did not reconverge by the terminal checkpoint on those rows; it does not establish global or
family nonexistence and does not build the preregistered splice-on-reconvergence form.

### Storage blocker and exact resume boundary

At blocker capture, APDataStore had 1,263,927,296 bytes free. Four sealed late-edit rows each
allocated 49,408 KiB on disk. Row 26 had allocated 24,320 KiB; completing it, the five unstarted
rows, and an 8,192 KiB manifest allowance projects another 280,320 KiB. Preserving the required
1,073,741,824-byte reserve therefore requires at least 1,360,789,504 bytes free. The fire trigger
is conservatively **APDataStore free bytes >= 1,400,000,000 with no concurrent decline**.

The interrupted row is restartable without recomputing its completed full leg:

- row-26 full `RESULT.json`: 4,349 bytes, SHA-256
  `c5f0fed2bca082e7890d25f99feddcf372f0c1d26eea34c07f06da7e5f224cf7`;
- row-26 terminal frame-600 state: 10,607,745 bytes, SHA-256
  `19afd7cd5269ee503292004358618f5404bf3dbae3a0161379473925822b344e`;
- incomplete stride-200 `RUN_SPEC.json`: 878 bytes, SHA-256
  `d23511b3ea37689bf58d1bb51a837d685e899b7497e05c0689b514e5189b4a81`;
- no `*.partial` file remained after the fail-closed interrupt.

Once the fire trigger is met, rerun:

```bash
.venv/bin/python experiments/ddm_rxc1_restartable_exact_coder.py --stage screen
```

The landed API revalidates and reuses all 26 sealed `ROW.json` receipts plus row 26's full
`RESULT.json`, then restarts the incomplete stride-200 leg from retained frame 400. After all
32 rows seal, run `--stage manifest`; only then may the preregistered Branch-1 cost adjudication
and Stage-3 API/cost finalization be appended.

### RECALL EVIDENCE update

The gen-2 bounded recall searched `.omx/research/`, arm-final messages,
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `.omx/state/main_hot_state.md`, the task ledgers,
the experiments/runtime surfaces, and the canonical-equations registry. Queries included
`rxc1`, `restartable exact coder`, `state reconverg`, `splice-on-reconverg`, `suffix re-encod`,
`incremental coder`, and `SCMDL`.

Beyond gen-1's seeds, recall found the binding preregistration's splice-on-reconvergence optimal
form and the live task-1374 / sfp1 generator-ready SCMDL chain. It did not find, in those bounded
scopes, an existing AFR1 post-edit state-reconvergence receipt. That changed the harvest plan by
adding the cheap terminal-state comparison above; it did not authorize a splice implementation or
a coder rebuild.

### Boundaries and frontier

Measured here: 26 sealed one-cell edit rows, 52 exact byte comparisons, partial per-stride CPU
wall costs, terminal adaptive-state equality, retained hashes, and the fail-closed storage
waterfall. Not measured here: the six missing rows; `SCREEN.json`; `MANIFEST.json`; a completed
Gate-1 adjudication; `d_seg`; `d_pose`; any scorer component; any exact contest score; CUDA;
candidate realization; or an outer-loop search.

Pointer UNMOVED: AFR1 remains `S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]`,
archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
