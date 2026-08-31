# ddm_vr2 Vertigo reclaim round 2 — destination-headroom refusal

**Disposition:** `BLOCKED_DEST_HEADROOM__ZERO_BYTES_MOVED__POPULATION_CLASSIFIED`
**Axis:** `[byte/custody apparatus, scorer-free]`
**Frontier effect:** none. No scorer, Modal call, training, archive mutation, evaluator, or pointer write ran.

## Result

The certify-and-move batch refused before copying any byte. APDataStore had only **5.726807 GiB**
available, already **4.273193 GiB below** the charter's binding 10 GiB post-move floor. The pk4
fire-order subtree would have projected APDataStore to **-51.015 GiB**; wwc1 would have projected
it to **3.772 GiB**. Both source trees remain present and both destination paths remain absent.

**Measured reclaim: 0 B.** The >=50 GiB target was not met. Under the currently authorized
Vertigo-to-AP route, the measured movable ceiling is **0 B until destination headroom is restored**.
The 46 headroom-only rows total **288.673870 GiB** by their recorded byte bases, so this is a
destination-capacity blocker, not evidence that Vertigo lacks a sufficiently large candidate
population.

| Tier | Before used | Before available | After used | After available | vr2 delta |
|---|---:|---:|---:|---:|---:|
| `/Volumes/VertigoDataTier` | 1,950,403,064 KiB | **2,613,864 KiB (2.493 GiB)** | 1,950,403,064 KiB | **2,613,864 KiB (2.493 GiB)** | **0 KiB** |
| `/Volumes/APDataStore` | 1,947,445,632 KiB | **6,004,992 KiB (5.727 GiB)** | 1,947,445,632 KiB | **6,004,992 KiB (5.727 GiB)** | **0 KiB** |

The before and after samples are identical because both refusals occurred before destination
creation. `df -h` reported both 1.8 TiB volumes at 100% capacity (Vertigo 2.5 GiB available,
APDataStore 5.7 GiB available).

## Fire-order items

| Item | Census | Preflight | Result |
|---|---:|---:|---|
| `ddm_pk4_20260813/retained/jacobian_bank` | 6,603 data files; 60,907,736,706 logical B; 59,498,436 allocated KiB; 0 symlinks; newest descendant 2026-08-14T02:45:48Z | AP projection **-51.015 GiB** | `BLOCKED-with-reason`; no hash, copy, destination mkdir, retirement, or symlink |
| `ddm_wwc1_winwin_cone_sweep` | 1,663 data files; 2,094,598,664 logical B; 2,049,432 allocated KiB; 0 symlinks; newest descendant 2026-08-31T16:33:37Z | AP projection **3.772 GiB** | `BLOCKED-with-reason`; four committed citation surfaces recorded for the future symlink-preserving move |

The pk4 interpretation is narrower than the top-level store: vr1 authorized the **56.742 GiB
allocated `retained/jacobian_bank` subtree**, not the entire 164.89 GiB pk4 tree. The exact bank
path had no literal hit in the searched state/research/executable-source scope, pk4's lane is
terminal, and `lsof +D` reported no open file descriptors. The residual top-level pk4 scientific
payload was classified `protected-SKIP` because no whole-tree closure certificate exists.

wwc1's committed memo, charter, arm final, and hot-state references were recorded in the phase
certificate. A future successful move must install the original-path symlink; editing landed memos
is neither needed nor allowed by this charter.

## Largest-first classification

The inventory ledger contains **142 unique item rows**: the pk4 fire-order subtree first, followed
by **all 141 direct Vertigo children >=1 GiB** in non-increasing allocated-byte order.

| Action | Items | Recorded bytes | Reason |
|---|---:|---:|---|
| `live-custody-SKIP` | 3 | 622,004,289,536 B | broad live `evidence` / `experiments` namespaces, plus AFR1 effective-frontier raw and `fire_main` custody |
| `protected-SKIP` | 93 | 780,257,107,968 B | explicit custody, executable literal reference, or bounded depth-4 seal/packet-manifest marker; no SR3 carve-out/protection lift |
| `BLOCKED-with-reason` | 46 | 309,961,207,426 B | APDataStore below the 10 GiB floor; pk4 and wwc1 additionally received exact certifier censuses |

The complete per-item skip table is
`.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.jsonl` (142 rows, SHA-256
`0d1f95255fdbf7dd32db8c9e35e12dbd6e29a9a8a3e6eaef55fe5115e4263b44`). Each row records
`path`, `bytes` plus its byte basis, the null hash state, typed action, destination, certificate
path where one exists, verification status, and reason. Every `sha_or_tree_hash` is intentionally
`null`: the batch was blocked or skipped before hashing. Filling those fields with a path/size
digest would be a fake tree-hash claim.

The two exact certifier refusals are in `.omx/research/ddm_vr2_move_phase_ledger.jsonl`
(SHA-256 `377d0e0e6e7eb0bdfe7ff8cb6557bac6a0066f2729f6daeda1e663c0d0606a61`). The phase ledger is
not substituted for the one-row-per-item inventory ledger.

## Protection and tool boundary

- `tools/vertigo_certify_move.py` matched the charter pin exactly:
  `6ba92499ad3ba9ce9206fba35da521b5f6828bc117a72801e6447e1db4cc520d`.
- `experiments/ddm_sr3_ap_certify_compress_reclaim.py` matched the carve-out cure pin exactly:
  `0494f73e2978f115c34bf6d620f053d22fc2683a9284daac0ded248b1b84ea65`.
- The SR3 rule was applied conservatively to every >=1 GiB direct tree: executable literal
  references and seal/packet-manifest markers cause a named `protected-SKIP` unless a complete
  keep-uncompressed carve-out and protection lift exists.
- The frozen packet manifest digest prefix `df001d743b42` appears in the pq1/pq2 custody receipts;
  no packet-custody path was selected. AFR1 archive SHA
  `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`, its retained raw,
  and `ddm_afr1_tile48_receiver_identity/fire_main` were explicitly skipped.
- Global `ps` inspection was denied by the managed sandbox. Exact `lsof +D` checks for the only
  two attempted sources returned no open descriptors. No other source was opened for a move.
- vr1's two named hardening debts remain accurately scoped: symlink-manifest equality and a
  durable partial-`rmtree` failure row. Both attempted sources had zero symlinks, and no retirement
  began, so neither gap was exercised. The charter's `Delta: none mechanical` forbade rebuilding
  the pinned tools in this execution arm.

## Landing status

The required serializer was invoked with post-edit SHA-256 declarations for all three vr2
artifacts. It failed at `git add` with `unable to create temporary file: Operation not permitted`
and returned **rc=19** after its automatic bundle fallback correctly refused every authorized SSD
tier below the binding 40 GiB reserve. Receipt:
`.omx/state/commit_serializer_fallback_refusals/20260831T175943.544611Z-76758/receipts.jsonl`.

No direct `git add` or `git commit` fallback was attempted: the charter permits commits only through
`tools/subagent_commit_serializer.py`, and local disk is not an authorized bulk/fallback tier. The
shared staged index remained empty. The memo and both ledgers are therefore complete but uncommitted.

## RECALL EVIDENCE

**Sources searched:** `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, the operating manual, live
`main_hot_state.md`, vr1's memo/census/phase ledger, sr2 and sr3 memos/tools, pk4's terminal verdict
and lane rows, ux1/wwc1 memos and final receipts, canonical task status, active-lane claims,
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, executable source roots, and
`tools/list_canonical_equations.py --json`.

**Queries included:** `ddm_vr1|#1165|pk4|jacobian_bank|certify move|carve-out|protected|#1336|wwc1|#1360`,
the exact Vertigo paths, `df001d743b42`, `cbb8d928`, `fire_main`, absolute executable-source path
literals, and depth-bounded seal/packet-manifest filenames.

**Beyond the charter seeds:** recall found pk4's later terminal CPU-authority verdict, which makes
the exact bank quiescence test meaningful; confirmed that vr1's fire order names only the Jacobian
bank; found wwc1's conditional scorer-native follow-on path and its committed citation set; and
found SR3's current every-tree reference-scan cure. These changed the plan from a whole-pk4 move
and citation editing to a narrow bank preflight, a future original-path symlink, and conservative
protected skips. The canonical equation registry supplied no storage equation that supersedes the
machine-checked custody contract.

## Boundaries

- No source byte was deleted, renamed, or moved. No destination payload or partial copy was created.
- No SHA/tree hash was claimed for an item blocked before hashing.
- No local-disk fallback was used; the charter did not authorize one.
- `upstream/` and the pre-existing staged index were untouched.
- The exact frontier did not move. This storage arm produced no score measurement.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN storage custodian; consumer store: `/Volumes/APDataStore/pact/vertigo_coldstore/pact/ddm_pk4_20260813/retained/jacobian_bank`; fire trigger: APDataStore has at least **66.742130 GiB available** (pk4 allocated bytes plus the 10 GiB floor), pk4 remains terminal, exact `lsof +D` is empty, and the pinned certifier hashes still match.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN storage custodian; consumer store: `/Volumes/APDataStore/pact/vertigo_coldstore/pact/ddm_wwc1_winwin_cone_sweep`; fire trigger: the pk4 move is terminal and APDataStore has at least **11.954491 GiB available**; record the committed references and retire only through a verified original-path symlink.
- `QUEUED-CONDITIONAL-SWEEP` — owner: next Vertigo reclaim arm; consumer store: `/Volumes/APDataStore/pact/vertigo_coldstore/`; fire trigger: APDataStore can hold the selected batch with 10 GiB remaining and Vertigo is still below 50 GiB free; reuse the 142-row ledger, re-run live references/seals, and attempt headroom-only rows largest first.
- `HARDENING-OWED` — owner: storage apparatus maintainer; consumer store: `tools/vertigo_certify_move.py`; fire trigger: before any certifiable source with nonzero symlinks or before the next source retirement; add complete symlink-target manifest equality and a durable partial-retirement failure row without weakening the current cert contract.
- `BLOCKED-LANDING` — owner: MAIN Git custodian; consumer store: the current Pact worktree and serializer receipt above; fire trigger: Git object writes become permitted, or an authorized external tier has at least 40 GiB reserve for the serializer bundle fallback; rerun the serializer with fresh post-edit hashes and the same three vr2 files, never a direct commit bypass.

## LIVE-HYPOTHESES

- A certified APDataStore reclaim can reopen this batch without changing the Vertigo move design. This is plausible because the only fired gate was destination capacity, while both exact source censuses and reference checks completed.
- The 46 headroom-only rows contain enough allocated bulk to exceed the 50 GiB target once AP capacity exists. This is plausible from their 288.673870 GiB aggregate, but each still requires its own hash certificate and refreshed protection scan.
- Some of the 93 conservative protected rows may become movable with precise SR3 keep-uncompressed carve-outs. This is plausible because the bounded executable-literal scan intentionally treats historical and live readers alike; it is not yet tested per tree.

## DEAD-ENDS

- Copying either pk4 or wwc1 with APDataStore below the 10 GiB floor is closed by the charter and the certifier's measured refusal.
- Moving the whole pk4 top-level tree is closed by the actual vr1 fire order, which names only `retained/jacobian_bank`.
- Hand-written `mv`, local-disk spill, deletion, or citation rewrites are closed by the custody contract.
- Treating a path/size digest or an old payload manifest as a current full-tree hash is closed; blocked rows remain explicitly unhashed.
- Moving a referenced/sealed tree without an SR3 carve-out or protection lift is closed; conservative skips are the honest result.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; vr2 ran no scorer and did not move the pointer.`
