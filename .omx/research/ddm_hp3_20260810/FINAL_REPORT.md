# DDM HP3 final report — HPAC section and ZIP frame

**Axis:** `[macOS-CPU advisory, scorer-free]`  
**Score authority:** `score_claim=false`; no scorer slot was claimed and no `upstream/evaluate.py` row
was fired.  
**Exact base:** PR130 CPR1, `191052 B`, SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.

## Outcome

One candidate survived the real byte race. `requant_frame_embed_step2_hpm300` is `191044 B`, SHA-256
`004436ea59780708e446392b33ab8d8ab5ce287622f5dd919a75208abee638ae`: **8 B smaller than the exact
base**. It changes 2,371 of the 4,800 deployed int8 frame-embedding values by rounding odd values to the
nearest even value with half ties toward zero, retains one monolithic Range stream, and counts one
24-byte frame-300 seek checkpoint in the archive.

The real receiver completed on CPU in `1015.029 s` (`1093.09 s` including the harness prechecks), under
the 1,800-second inflate budget. It reconstructed all `117,964,800` tokens with canonical SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`, and rendered the full
`3,662,409,600 B` raw video with SHA-256
`a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`. That raw hash is the established
FX1/DT1 base raw hash, so the realized output is byte-identical to the base.

With unchanged scorer inputs, the eight-byte rate reduction gives a **derived, non-authority** delta
`-0.00000532687162497737` and `S = 0.1721359706202714696`. The exact pointer remains unmoved until the
queued evaluator row fires.

## HPAC decomposition

The shipped HPAC is `20,179 B` raw; its exact leave-one-out contribution inside the joint model XZ is
`15,092 B`. The raw grammar reconciles exactly:

| field | bytes |
|---|---:|
| `IHS1` magic | 4 |
| 517 per-output-channel bit-depth nibbles | 259 |
| 108,518 packed deployed-weight bits | 13,565 |
| fixed fields, total | 6,351 |
| └ frame embedding, int8 `[600,8]` | 4,800 |
| └ biases/exponents | 1,551 |
| **total** | **20,179** |

The base joint models XZ is `73,968 B`. The winning re-quantization reduces it to `73,420 B` (`-548 B`),
while its monolithic Range bytes grow from `116,980 B` to `117,496 B` (`+516 B`) and its counted seek
header costs `24 B`, yielding the measured net `-8 B`.

## Real archive race

Every listed candidate has retained HPAC bytes, model bytes, token bytes, member `p`, deterministic
archive, repeat archive, SHA-256, and parse-back proof.

| candidate | archive B | delta vs base | result |
|---|---:|---:|---|
| canonical control | 191,052 | 0 | exact archive SHA reproduced |
| control, independent Range / 24 frames | 191,218 | +166 | framing/reset loss |
| control, independent Range / 120 frames | 191,098 | +46 | framing/reset loss |
| control, monolithic + frame-300 seek | 191,076 | +24 | exact checkpoint price |
| exact frame-embedding delta, /24 | 191,406 | +354 | rejected |
| exact frame-embedding delta, /120 | 191,286 | +234 | rejected |
| exact frame-embedding delta, seek | 191,264 | +212 | rejected |
| frame-embedding step2, /24 | 191,190 | +138 | rejected |
| frame-embedding step2, /120 | 191,054 | +2 | rejected |
| **frame-embedding step2, seek** | **191,044** | **-8** | **receiver-closed survivor** |
| deployed `abs(w)<=1` prune, /24 | 210,538 | +19,486 | rejected |
| deployed `abs(w)<=1` prune, /120 | 210,422 | +19,370 | rejected |

The pruning instance saves `780 B` in joint models XZ but increases token bytes by about `20 KB`; even
removing the entire 24-frame checkpoint tax cannot make that instance competitive. This is an
INSTANCE verdict, not a family verdict on jointly retrained sparsity.

## Container closure

The actual ZIP overhead is **100 B**, not 104 B. The extra four bytes in the prior marginal accounting
are the model-length prefix inside member `p`, not ZIP framing. The one-member ZIP is already at the
structural floor: 30-byte local header + one-byte name + 46-byte central header + one-byte name +
22-byte EOCD. There are no extras, comments, or alignment bytes. Stored ZIP reproduces the exact base
archive; deflate-q9 parses back exactly but is `191112 B`, **60 B worse**.

## RECALL EVIDENCE

Sources searched before adjudication:

- Governing surfaces: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md`.
- Full research corpus queries included `hpac`, `IHS1`, `15,092`, `b07fff73`, `frame_embed`,
  `self-compress`, `memoryless bound`, `gauge`, `monolithic grammar`, and `temporal delta` across
  `.omx/research/`, receipts, indexes, DAG FEEDs, design docs, and task/bridge stores.
- Canonical equations were enumerated with `tools/list_canonical_equations.py --json`; relevant prior
  surfaces were L20 monolithic grammar, L21 byte maps, L22 storage permutations, L24 raw LZMA, L25
  temporal delta, and L37 streaming.
- Beyond the charter seeds, RR3 supplied the exact HPAC raw/checkpoint/code hashes; DT1 supplied retained
  exact n600 causal symbols/logits; FX1/DT1/SR1 supplied the canonical decoded-token and raw-output hashes.

What changed: DT1 custody made a scorer-free full-n600 representation race possible without touching
the live token arm; the L25 precedent admitted the exact frame-delta control but did not prejudge it;
the monolithic grammar clarified that ZIP is 100 B and the four-byte prefix is payload; and the prior raw
hash gave a direct byte-identity receiver verdict. No prior HP3 archive candidate was found in the
searched scope.

## Verification and boundaries

- `7` focused tests pass; Ruff and format checks pass; `git diff --check` passes.
- `tac.payload_retention_gate`: zero findings.
- Independent receipt audit: `488` unique path/byte/SHA records verified, zero mismatches.
- Full SSD receipt: `/Volumes/VertigoDataTier/pact/ddm_hp3_20260810/FINAL_RECEIPT.json`, SHA-256
  `1c5b90fcbbac3622db9da886b6cc128de218977aa505c10e6826c2f1aa159a7d`.
- Repo receipt: `.omx/research/ddm_hp3_20260810/FINAL_RECEIPT.json`, byte-identical to the SSD receipt.
- The managed macOS safe-run receipts report `peak_rss=0 MiB`; process-group RSS observation is therefore
  non-informative in this sandbox. Wall-time enforcement and on-disk stage receipts did operate.
- No scorer forward, exact evaluator, contest CPU/CUDA run, Modal dispatch, upstream edit, token-arm edit,
  semantic-arm edit, artifact deletion, or payload discard occurred.

## Follow-on disposition

- **QUEUED-WITH-FIRE-ORDER:** exact n600 evaluation of the immutable survivor. Owner: MAIN scorer
  scheduler / successor claiming the released `ddm_ai1` slot. Consumer:
  `.omx/state/main_hot_state.md` and `.omx/research/ddm_hp3_20260810/EXACT_EVAL_RECEIPT.json`. Fire only
  after the sole scorer slot is free and the lane claim is recorded; use `EVAL_QUEUE.md` without changing
  the archive bytes.
- **QUEUED-WITH-FIRE-ORDER:** compose the step2 HPAC representation with the banked split-model/ANS rate
  row only after that receiver exists. Owner: future rate-harvest composer. Consumer: the PR130 rate
  composition store in `.omx/state/main_hot_state.md`. Fire only from retained HPM/DT1 payloads, and
  require a new real archive because the `-8 B` result is not additive with a changed token coder.

**OWN-VEHICLE FRONTIER:** PR130 CPR1 remains `S = 0.172141297491896447 @ 191052 B`
`[contest-CUDA, DALI GT, n600]`. HP3 banks a receiver-closed `191044 B` archive
`[macOS-CPU advisory, scorer-free]`; pointer unmoved pending exact evaluation.

