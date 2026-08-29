# ddm_lb1 — banked lossless joint collection on the gb1 body

`date_utc: 2026-08-29` · `arm: ddm_lb1_banked_lossless_joint_collect` ·
`axis: [macOS-CPU advisory / scorer-free exact byte measurement]` · `score_claim: false` ·
`frontier_moved: false` · `verdict_scope: INSTANCE — gb1/jt21 token-corrector body`

**Verdict: `JOINT_POOL_CLEARS_BAR__NATIVE_IDENTITY_PROVEN__DUAL_AXIS_READY`.** The one physical
n600 re-encode produced a 180,083 B candidate: **−132 B vs gb1**, **−109 B beyond jt21**, and
zero token changes. It clears the ~30 B bar by 102 B. Full native receiver identity passed and
both axis seals validate. No authority fire occurred in this arm; MAIN owns the queued fire order.

## 1. Source re-derivation

The source numbers below come from the retained artifacts, not the charter headlines. The durable
preflight is
`/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/measurement_v1/PREFLIGHT.json`
(sha `df3b56d0fd736026a99b4bd6bddbbe1fdfa452d08b51ddc79d15a43ee579ea81`).

| credit | source receipt sha256 | source claim | re-derived value | lineage valid? | marginal in physical joint pool |
|---|---|---:|---:|---|---:|
| mi1 `patch192_only` | `7f297981bdd508a187071d4704b7dce2e1acf9ebae1bc59d2bdc23b5a603b4d4` | +211.13 B | **+211.13455430508475 B held-out**, +234.53749599940784 B in-sample; 192 cells; max |offset| 24.96285 | **YES, with a measured transfer haircut.** Decode-derived `patch192=(y//32)*16+(x//32)`, zero stored bytes; physical re-encode retained 109 B beyond jt21. | **−109 B** beyond jt21 |
| jt21 `cls_groupbin8` collection | `a117480b2f79efff194ccf3c54520039579ea3f5efbc3f35cf881b9945d5ffe9` | −23 B vs gb1 | candidate **180,192 B**, sha `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3`; token stream 113,601 B; `tokens_changed=0`; source control byte-identical | **YES.** Retained exact archive and stream; source runtime pin `(sha,bytes)` is consistent. | **−23 B** already present in the joint base |

The patch expression was independently reconstructed from the two component coordinate charts:
the tile48 row/column and subtile4 row/column recover the patch row/column exactly. Their flat
enumerations are not numerically multiplicative, so `tile48*4+subtile4` is not the receiver rule.
The receiver rule above is the exact 12×16 raster index and is causal because `(x,y)` are known
before the symbol is decoded.

The selected mi1 seed is not a singleton accident: held-out gains are 211.134554 B (seed
20260824), 202.511923 B (seed 777), and 204.794998 B (seed 31337). That supports the mechanism,
but does not upgrade any of those ledger values into archive bytes.

## 2. Full unfired-lossless inventory on this body

This is a bounded full-corpus recovery across `.omx/research`, the canonical index/DAG and task
status, followed to the retained receipts named by each surviving headline.

| item | receipt-backed status | disposition in lb1 |
|---|---|---|
| jt21 `cls_groupbin8`, −23 B exact | Retained 180,192 B archive; banked 7 B short of the ~30 B bar. | **CONSUMED** as the physical joint base. |
| mi1 `patch192`, +211.1346 B held-out | Zero stored bytes and receiver-expressible; never physically collected. | **OWED / ONLY NEW POOL MEMBER.** |
| mi1 `tile48`, +112.9413 B held-out | Coarser position chart, contained by the selected patch192 chart; never byte-closed. | **SUPERSEDED** by patch192 in this one-member extension. |
| mi1 `subtile4`, +56.5064 B held-out | Fine periodic chart; its group-plan mechanism has already been collected by gb1 and jt21. | **CONSUMED/SUBSUMED**, not added as an independent credit. |
| mi1 `groupbin8`, +64.1961 B held-out | Exact mechanism shipped as gb1 `groupbin8_surprise`; jt21 added `cls_groupbin8`. | **CONSUMED.** |
| mi1 `group190`, +51.2466 B held-out | Dominated in mi1 by groupbin8; no independent nomination or physical receipt. | **DEAD / DOMINATED.** |
| mi1 `row384`, +71.1084 B held-out | High-cell, overfit-prone row diagnostic; not nominated, not byte-closed, dominated by patch192. | **DEAD / NOT A SURVIVING CREDIT.** |
| jt22 mixer-context B, −1 B | Instance race explicitly closed it and did not bank it as additive. | **DEAD / NOT BANKED.** |
| jt23 coder and ZIP/RX1 axes, 0 B | 25-config section race plus structural framing floor. | **CLOSED.** No coder rerace. |
| fcd1 176,436 B archive | 5,268 tokens changed and distortion unmeasured. | **EXCLUDED: not lossless.** Its consumer store was not touched. |
| paid mi1 conditioning form | Missed break-even by 47.4×. | **CLOSED.** No stored table was introduced. |

No additional same-body, unfired, receipt-backed lossless survivor was found in that bounded scope.
That statement is not a claim of global nonexistence.

## 3. The one physical joint re-encode

Runtime staging receipt:
`/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/measurement_v1/RUNTIME_PREPARE.json`.
Both staged runtimes carry the exact jt21 archive (`ec0dd68f…`, 180,192 B) and passed the
`(sha,bytes)` pin-consistency gate. The base reproduces the 21-family Python configuration; the
candidate appends one and only one member, `patch192_only`, for **22 total families**. All adaptive
state, the range stream, per-frame ledger, checkpoints, build products, receipts, and candidate
archive are retained under the arm store.

| row | archive bytes | sha256 | Δ vs gb1 180,215 B | Δ vs jt21 180,192 B | physical status |
|---|---:|---|---:|---:|---|
| gb1 pointer | 180,215 | `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4` | 0 | +23 | retained authority anchor |
| jt21 control/base | 180,192 | `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3` | −23 | 0 | full n600 stream reproduced exactly: 113,601 B, sha `4c9dc10c…76ba7` |
| **jt21 + patch192 joint pool** | **180,083** | `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9` | **−132** | **−109** | full n600; token stream 113,492 B; `tokens_changed=0`; `delta_trustworthy=true` |

Naive headline upper bound = `23 + 211.134554 = 234.134554 B`; physical marginal = 132 B.
The ledger-to-wire shortfall is **102.134554 B**, or **43.62% overlap**, leaving **56.38%** of the
naive pool. Equivalently, patch192 transfers 109/211.134554 = **51.63%** of its held-out ledger
gain beyond the jt21 base. This overlap diagnostic mixes one held-out model ledger with exact
bytes, so the **132 B archive marginal is authority for the lossless decision**; the fraction is
not promoted to a coder law.

The registered prediction (`>=140 B`, `>=60%` retention) missed narrowly: observed 132 B and
56.38%. Its falsifier (`<30 B`) did not occur. The useful update is that absolute patch position
shares more structure with the existing adaptive family pool than the prior expected, yet still
clears the fire bar by 4.4×.

## 4. Decode identity, portability, and seal fork

The candidate cleared the bar, so the native fork was taken. The git-custodied gb1 generation-20
C corrector was extended with the exact jt21 `cls_groupbin8` and lb1 `patch192_only` rules. The
result is 22 ordered families; it compiles warning-free with `-ffp-contract=off -fno-fast-math`,
passes `assert_config_matches()`, and leaves the candidate runtime pin-consistent. Receipt:
`/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/measurement_v1/NATIVE_PORT.json`.

Full retained native decode passed at
`/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/identity_native/`: 600 pairs,
3,662,409,600 raw bytes, sha
`7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`, exactly equal to the
gb1 Python/native macOS anchor. The receiver named `NativeFreeCorrector`; decoded tokens are the
canonical `cc10a7b0…3eefb`; decode/render took 1,028.604 s, below the 1,800 s contract. Receipt:
`measurement_v1/NATIVE_IDENTITY.json` (sha `e3aca9d8…1abb7`). The 3.66 GB raw and stage
checkpoints remain on APDataStore.

Both candidate seals passed disk re-derivation and dispatcher dry-run against the same runtime
digest `4bc36d80…87e85` (39 files, 689,359 B):

| axis | seal | seal content digest | dry-run manifest | disposition |
|---|---|---|---|---|
| contest-CUDA | `measurement_v1/SEAL_lb1_contest_cuda.json` | `6d28b3a5…8934e` | `fire_cuda/FIRE_MANIFEST.json` sha `e6f79e7d…185e` | `SEAL_VALID`, fire ordinal 1 |
| contest-CPU | `measurement_v1/SEAL_lb1_contest_cpu.json` | `3f69c5e8…ee4be` | `fire_cpu/FIRE_MANIFEST.json` sha `be317ece…139cd` | `SEAL_VALID`, fire ordinal 2 |

The exact governed commands, distinct lane IDs, common pair-group, owner, consumer stores, and
triggers are sealed in `measurement_v1/DUAL_AXIS_FIRE_ORDER.json` sha
`8a5da68411e01c162a109aee0fb580c70a08ac55caea8aca05e6b57c887dac6f`. CUDA fires first because
it is the live pointer axis; CPU follows only after single-flight clears. No scorer was loaded,
no lane was claimed, and no Modal call was launched by lb1.

## 5. Component arithmetic and authority boundary

The pointer components are `d_seg=0.00020139`, `d_pose=6.37e-6`, and
`N=37,545,489`. The source authority row recomputes exactly:

| body | bytes | rate `25B/N` | seg `100*d_seg` | pose `sqrt(10*d_pose)` | S |
|---|---:|---:|---:|---:|---:|
| gb1 `[contest-CUDA T4 n600]` | 180,215 | 0.1199977712 | 0.020139 | 0.0079812280 | **0.14811799921260607** |
| lb1 same-distortion projection | 180,083 | 0.1199098779 | 0.020139 | 0.0079812280 | **0.14803010583079396** |

Only the gb1 row is an authority score. Any lb1 number here is a same-distortion projection from
an exact lossless byte delta until MAIN fires the sealed exact T4 row. Local work is
`[macOS-CPU advisory / scorer-free exact byte measurement]`, `score_claim=false`.

## 6. Verdict and handoff

**Exit verdict: `QUEUED-WITH-A-FIRE-ORDER`.** Exact marginal = −132 B versus gb1; threshold
surplus = 102 B; projected same-distortion `ΔS = −8.789338181212661e-05`. This is a real retained
archive and a real full-body re-encode, not the mi1 ledger projection. Pointer movement remains
false until MAIN's exact T4 fire passes. Local native identity and both seal validations are closed.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER (ordinal 1, contest-CUDA)** — owner: MAIN sole scorer-lane router;
  consumer store: `/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/fire_cuda/`;
  fire trigger: gb1 pointer archive still `ba1f3830…`, CUDA seal revalidates, no scorer/duplicate lane
  is active, and MAIN records claim `ddm_lb1_joint22_patch192_cuda_n600_20260829`.
- **QUEUED-WITH-A-FIRE-ORDER (ordinal 2, contest-CPU)** — owner: MAIN sole scorer-lane router;
  consumer store: `/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/fire_cpu/`;
  fire trigger: ordinal 1 is harvested, single-flight clears, CPU seal revalidates, and MAIN records
  claim `ddm_lb1_joint22_patch192_cpu_n600_20260829` under pair-group
  `ddm_lb1_joint22_patch192_dual_axis`.

## LIVE-HYPOTHESES

- The 51.63% ledger-to-wire transfer suggests another decode-derived absolute-position chart could
  pay if it partitions residual conditional structure orthogonally to patch192; this is plausible
  because 102.13 B of the held-out patch signal was shared, while 109 B was genuinely new.
- The exact contest-CUDA row should preserve gb1's distortion legs and realize the full rate delta:
  both candidates decode to the identical token field locally and share the renderer, but CUDA's
  device-specific raw sha is untested until MAIN fires the sealed row.

## DEAD-ENDS

- Re-racing coders or ZIP framing is closed on this exact body: jt23 found 0 B and the ZIP is at its
  structural 100 B floor.
- The paid patch table is closed: its storage cost misses the modeled benefit by 47.4×; only the
  zero-stored-byte receiver expression is admissible.
- Treating held-out ledger gains and jt21 bytes as additive is closed: the real re-encode retained
  132/234.134554 B and measured 43.62% overlap.
- The flat formula `tile48*4+subtile4` is closed: those contexts enumerate rows/columns differently.
  The exact receiver expression is `(y//32)*16+(x//32)`; the preflight verifies coordinate
  reconstruction rather than equating incompatible flat indices.
- Running the public receiver with no `python` on PATH is closed: it falls into an inaccessible uv
  cache before decode. The locked project interpreter already carries Brotli 1.2.0 and completed
  the exact native run; do not retry the bare-environment invocation.
- Using the unmodified gen-20 C table for jt21/lb1 is closed by its drift gate. The exact gen-22 port
  is now compiled, config-matched, and full-payload identical; successors should consume it.

**OWN-VEHICLE FRONTIER:** gb1 remains 0.14811799921260607 `[contest-CUDA T4 n600]`, archive
`ba1f3830…` / 180,215 B; lb1 is a 180,083 B scorer-free exact candidate with same-distortion
projection 0.14803010583079396, and has not moved the canonical pointer.
