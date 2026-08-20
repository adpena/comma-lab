# ddm_rr7 — the T4 wall-clock row: the native token decoder is byte-perfect and SLOWER

Date: 2026-08-20 · Owner: ddm_rr7 · Axis: **`[contest-CUDA]`, Tesla T4, n600, rc=0**
Call `fc-01M0FCY9478HBGPMWXNPYBDDX4` · archive `f3bce5d259a0…` (180,625 B, **UNCHANGED**)

---

## HEADLINE

**The identity proof HOLDS on T4 and the speedup INVERTS. Do not fold the port.**

`ddm_rr6` proved the native token decoder byte-identical locally and measured **1.865x** on an
M5 Max, projecting a T4 inflate of **797.7 s**. The shipping axis measured **1,612.579 s** —
the projection was optimistic by **2.02x**, and the ratio did not merely fall short of the
1.804x PASS bar, it **changed sign**: **0.867x**, i.e. the port is **15.3% SLOWER** than the
Python decoder it replaces.

| gate | result |
|---|---|
| `canonical_score` | **0.14839100138338618** — delta **exactly 0.0** vs jg5 |
| `0.raw` sha256 (PRIMARY) | **`6bf8acf8d441…`, 3,662,409,600 B — IDENTICAL** |
| `decoded_token_sha256` | `cc10a7b0…` — IDENTICAL (plus logit / cdf / `bit_position` 910837) |
| `avg_segnet` / `avg_posenet` | `0.00020139` / `6.37e-06` — IDENTICAL |
| `token_decoder` | **`native-hpac-split`** — the default flip FIRED on a bare invocation |
| `decode_path` / `decode_threads` | **`scalar` / `4`** — the FORCE_SCALAR pin is live; **not a fallback** |
| **inflate** | **1,612.579 s** vs jg5 **1,419.904 s** — **+192.675 s WORSE** |

This is not a python fallback and not a broken build. The port did exactly what it was
designed to do, on the hardware that matters, and lost.

---

## 1. The measurement

Same archive, same evaluator, same 600 pairs. Only the runtime tree differs.

| | jg5 (python decode) | **rr7 (native-hpac-split)** | delta |
|---|---:|---:|---:|
| token stage `token_decode_or_checkpoint_load` | 1,341.540 s | **1,546.617 s** | **+205.077** |
| whole inflate | 1,419.904 s | **1,612.579 s** | **+192.675** |
| evaluate | 51.428 s | 44.853 s | −6.575 |
| charged (inflate+evaluate) | 1,471.332 s | **1,657.432 s** | **+186.100** |
| headroom in the 1,800 s job wall | 328.668 s | **142.568 s** | **−186.100** |
| token ratio r | 1.000x | **0.8674x** | — |

Sub-stages are otherwise unremarkable and confirm nothing else moved: render 44.675 s
(jg5 54.536), selector 3.922 s (5.145), archive setup 1.823 s (0.358).

### 1.1 Byte identity, proven on the shipping axis

`0.raw` is byte-identical to the jg5 T4 row — `6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883`
at 3,662,409,600 B — and all four rr6 token anchors reproduce. The score is therefore
identical **by construction**, and the row is a wall-clock measurement, exactly as rr6
specified. `rr6`'s central claim is CONFIRMED on T4, not merely locally.

**The `0.raw` sha was the load-bearing gate, not the score float.** The charter framed the
falsifier as "the score must equal 0.14839100138338618 exactly." That is the right instinct
but the wrong instrument: byte identity guarantees the BYTES, it does not guarantee that two
independent CUDA scoring passes over identical bytes agree to the last ULP. Had the score
moved in its final digits with `0.raw` matching, that would have been scorer non-determinism,
not a decoder defect. Both agreed here, so the distinction cost nothing this time — it is
recorded because the sharper gate should be the one a future row is judged against.

### 1.2 Why it lost, and what that retires

`ddm_rr6` §2.1 measured the win as **thread-borne**, not from lowering numpy to C: at one
thread the C is 1.007x, i.e. no win at all. The whole 1.79–1.865x came from a 4-thread pool on
**M5 Max performance cores**. The split path moves the sparse model OFF the GPU onto the host
CPU, so the shipping result was always going to be a question about the T4 box's vCPUs, and
rr6 §2.4 said plainly that the sign of the change was not determined by argument.

It is now determined by measurement, and it is negative. **On T4 the CUDA Python token stage
(1,341.540 s) BEATS the 4-thread native scalar CPU decode (1,546.617 s) by 205 s.** rr6's
"For" reasoning — that a GPU losing to this laptop's CPU by 2.28x proved the stage was
latency-bound and the round trips were the enemy — was sound about the round trips and wrong
about the conclusion, because it silently compared a T4 host vCPU to an M5 Max core. Deleting
114,000 host↔device round trips did not pay for running the decode on much weaker silicon.

**Retired by this row:** "lower the token stage to native CPU C" as a wall-clock lever on the
contest T4. Not retired: the identity machinery, the fail-closed build, the default flip, the
receipt fields — all of which worked and are reusable.

---

## 2. The budget verdict, in all three frames

The three published frames are ONE measurement seen three ways. All three worsen.

| frame | jg5 | **rr7** |
|---|---|---|
| **A.** canonical `tac.contest_budget` — charged (inflate+evaluate) vs `[822, 1302]` | REFUSE, −169.3 s | **REFUSE, −355.4 s** |
| **B.** inflate alone vs evaluate-corrected `[890.6, 1430.6]` | WARN, **+10.7 s** (fits) | **REFUSE, −182.0 s** |
| **C.** absolute 1,800 s CI job wall | 328.7 s left | **142.6 s left** |

Frame B is the packet's published reading and the only one under which jg5 fits at all. **rr7
breaks it.** Frames A and B are related exactly by
`[890.6, 1430.6] = [822, 1302] + (evaluate_est 120…180 − evaluate_measured 51.4)`, so quoting
one without the other hides the evaluate correction, not a disagreement.

**A live inconsistency MAIN should settle:** the repo's own canonical grader
(`tac.contest_budget.budget_verdict_for_receipt`) grades **jg5 as REFUSE**, while
`ddm_pq3` and the packet text say **WARN**. Both are defensible — the module deliberately
double-charges a measured evaluate against a residual that already netted out an estimated
one, and says so — but the packet publishes the *less* conservative of the two and does not
name the other. That is a disclosure gap, not an arithmetic error.

**Second measured fact worth keeping:** evaluate came in at **44.853 s** (jg5 51.428 s), well
under ua2's ESTIMATED 120–180 s. Two rows now agree that the estimate is high by roughly 2.5x,
which is what makes frame B's correction real rather than a courtesy.

---

## 3. What the #1111 packet should now say

**The port must NOT be folded. Ship the jg5 tree unchanged.** Concretely:

1. **`FREEZE_CHECKLIST.md` variant (c)** ("Optional — fold the wc2 corrector port, then RE-PIN
   and RE-PROVE identity") — record that the *native token-decode* half of this variant is now
   **MEASURED CLOSED** on the shipping axis. Its cost structure was never the issue; its sign
   was.
2. **`README_PUBLIC.md:28-35`** — the sentences stand as written. "Token decode alone is
   1,341.5 s of that 1,419.9 s, 94.5%" remains the shipped truth, and the clause saying the
   accelerator "is not in the tree evaluated here" is now not merely accurate but *correct
   policy*: putting it in the tree costs 193 s.
3. **The residual-band sentence** (`ddm_pq3:145-151`, `FREEZE_CHECKLIST:113-114`) — unchanged
   for the shipped bytes: WARN by 10.7 s in frame B. Add the frame-A REFUSE beside it per §2 so
   the disclosure carries both readings rather than the friendlier one.
4. **The rr2-corrector disclosure** — the honest update is a REPRICING, not a cancellation.
   The corrector port lowers more Python to C **on the same CPU that just lost by 205 s**, so
   it inherits this row's wall. It is no longer enough for it to be faster locally; it must
   beat the T4's CUDA sparse evaluator on T4 vCPUs. **UNMEASURED** — see §5.1.
5. **Runtime tree SHA-256** — the checklist's `2103073d739f…` stays, because the shipped tree
   stays. The rr7 tree's derived hash `58f2c0dcbead3ab0…` is recorded here for provenance only
   and must not be pinned into the packet.

Nothing in the score section changes. The score is identical and the archive was never touched.

---

## 4. The re-pin was not a code edit — and must never become one

The charter's step 1 asked for a `.py` edit to re-pin three runtime-tree SHA validators in
`experiments/contest_auth_eval.py` (cited at :1554/:1566/:1627; live at **:1787 / :1799 /
:1233** — the line numbers had drifted, which is why they were verified).

**No such edit was owed, and making it would have been a defect.** All three validators are
parameterised by `expected_runtime_tree_sha256` / `expected_runtime_files_sha256`; none carries
a hardcoded jg5 value, and a repo-wide search for jg5's tree hashes found zero occurrences in
`tools/`, `experiments/` or `src/`. `tools/fire_modal_auth_eval.py` pins the flag to **`auto`
on both axes by construction** (its failure F3), and the workers accept only `''`/`'auto'`/the
runtime FILES digest, refusing any other value — because a projected and a remote tree hash are
environment-coupled and structurally disagree (the r9m deadlock).

Hand-typing a digest into that file would have reintroduced exactly the class
`tac.candidate_seal` exists to extinct: *"a producer that accepted a typed digest would
reproduce the exact failure the seal exists to stop."*

**The seal IS the re-pin.** Steps 1 and 2 of the charter are one step. What was actually owed
was a fresh seal measured from the new tree, which is what landed:

* runtime FILES digest **`788e8a9e2558aa7d…`** (35 files, 624,504 B) — DERIVED by regenerating
  the tree from committed state and re-measuring, never typed;
* reproducibility PROVED: the regenerated tree is byte-identical to rr6's
  (`per_file_diffs = []`), so the fired tree is the proved tree;
* archive re-measured **unchanged** at `f3bce5d2…` / 180,625 B, asserted against
  `--verify-archive-sha`;
* seal `9bfc92d7017b3ed3…` → **`SEAL_VALID`**, `SEAL PIN CONSISTENT`;
* the digest is INVARIANT under the fire path's SANITIZE stage — verified empirically here
  (38 AppleDouble files stripped, digest unchanged), not merely asserted from the code comment.

The admit bar was rewritten for what this row actually is. jg5's bar (`net dS < -3.5e-6`, an
*improvement* threshold derived against the **br1** pointer) would have been meaningless: the
pointer has since moved to jg5 itself, and a byte-identical candidate cannot improve on itself.
The rr7 seal states the predicate as **identity** — `net dS` must be exactly 0.0 — with the
`0.raw` sha pre-registered as the primary falsifier.

---

## 5. Owed

1. **Price the corrector port ON THE SHIPPING AXIS before building it.** rr6 §6 puts the Python
   corrector at 59.2% of the split run *locally*. Transferring that split to T4 is precisely
   the cross-regime error this row just measured, so it is **not** quoted here. The shipping
   report emits no `native_stage_seconds`, so the T4 split cannot be decomposed from this row
   at all. **The cheap probe is to add the native/python sub-stage breakdown to the shipping
   report and re-fire once** — one T4 row that says how much of the 1,546.6 s is Python decides
   whether ~2,100 lines of corrector C can ever beat 1,341.5 s on this hardware.
2. **x86 remains UNVERIFIED and now moot for shipping.** `-DF26_FORCE_SCALAR=1` means the AVX2
   kernels were never in the binary. This row ran `scalar` on the contest's own x86 T4 host, so
   the scalar path is now x86-EXERCISED; the hand-written AVX2 kernels remain unexecuted
   anywhere.
3. **`classify_decode_path` does not know `"scalar"`** — it returned `other` for a rung its own
   docstring names ("AVX-512 → AVX2 → scalar-C → NEON → Python"). Cosmetic for this verdict
   (REFUSE in every frame regardless), but it silently drops a real rung.
4. **The jg5 harvested `contest_auth_eval.json` is not JSON** — it is a Python `bytes`-repr
   (`b'{...`), so `json.load` raises on the current frontier's own inner receipt. The pointer
   consumes the scalar mirror, so nothing load-bearing depends on it; the rr7 row harvested
   clean, so this looks like a one-off rather than the harvester's current behaviour.
5. **Three stale Modal apps** were reported "live" for **272.8 h / 168.9 h / 143.5 h** with zero
   running containers. See §6.

---

## 6. Apparatus defects found on the way to the row — four fixes, all committed

The fire could not be executed as chartered. Every blocker below was real, was hit at $0, and
is now self-protected by a test.

| # | defect | commit |
|---|---|---|
| 1 | **The port was unshippable through the canonical fire path.** `runtime/f26_split_token_decoder.py` tripped the secret-name validator on the substring `token` — this domain's core noun. The cure already existed (`RUNTIME_UPLOAD_BASENAME_ALLOWLIST`, added 2026-08-04 for `ddm_r7_token_coder.py`, the identical collision) but the gate had **no test at all**, which is why the second instance arrived with no guard. Reviewed the file first: 298 lines, byte-identical to committed git-tracked source, zero credential-shaped content. | `0af2ae231e` |
| 2 | **Unit tests were poisoning the production single-flight ledger.** `test_modal_auth_eval.py` fixtures wrote fake `fc-test-modal-auth*` ids into `.omx/state/modal_call_id_ledger.jsonl` as permanently-live `dispatched` rows. Four had accumulated and **refused this paid dispatch**. The module already had an autouse fixture isolating tests FROM the guard; it did not stop them POISONING the guard's data for everyone else. Fixed at `register_dispatched_call_id` (the ledger module), which the `_fail_closed` wrapper resolves at call time. Proven: 55 tests pass, ledger row count unchanged at 876. | `8095963f79` |
| 3 | **The single-flight cloud cross-check had been silently skipping.** `shutil.which("modal")` returns None because `modal` lives at `.venv/bin/modal` and `.venv/bin` is not on PATH. It announced this with one stderr line inside Modal's own build output. Both legs of the guard were compromised at once — the cloud leg inert, the local leg lying. | `3160e5f990` |
| 4 | **…and restoring it alone would have caused FALSE REFUSALS.** With the CLI found, the leg reported three live apps aged 272.8 h / 168.9 h / 143.5 h while `modal container list` showed exactly ONE container — mine. `modal app list`'s `tasks` count goes stale for days on detached apps. An app now counts live only if it OWNS A RUNNING CONTAINER; the tasks predicate is kept as a necessary condition so the change can only REMOVE false positives, and an unavailable container query falls back to the old over-reporting behaviour. | `053e4e5944` |

Fix 3 without fix 4 would have traded a silent skip for a guard that blocks correct work.
Neither is an improvement alone; the pair is.

Separately, **five `test_candidate_seal.py` fire-path tests had been RED** since the tool became
paired-by-default on 2026-08-18 — argparse exited 2 and the seal *refusals they exist to prove*
were never exercised. Confirmed pre-existing by running them at `23e1a601b0` in a clean
worktree. Restored (41 pass, was 36/5). — `05f1190df6`

---

## 7. Side tasks

**(a) A retained receipt whose filename lies.** Verified independently and the finding is
STRONGER than reported: `wc2c_scalar_twin_n600_t4.json` is not merely sharing a float with
`wc2c_thread_independence_t1.json` — the two files are **byte-identical whole-file**
(both sha256 `8742e769f069b136…`). The `_t4` name carries the **t1** run. The true 4-thread
scalar row is `wc2c_thread_independence_scalar_t4.json` at 324.779 s. Read at face value the
name says scalar is 1.79x SLOWER than NEON, which would make `-DF26_FORCE_SCALAR=1` look
catastrophic; the truth at matched threads is scalar 324.779 s vs NEON 326.160 s, 0.4% FASTER.
`ddm_wc2c`'s memo number is correct; only the filename is wrong. **Custody bytes were NOT
renamed** — renaming retained payload breaks every citation by path and is itself the
phantom-name genus. Wrote a sidecar that travels with the file plus an append-only correction:
`wc2c_scalar_twin_n600_t4.MISLABELLED_NOTE.json`, `RETENTION_CORRECTIONS.jsonl`.

**(b) Vertigo at 100%.** Mirrored `ddm_dx1`'s retained set to
`/Volumes/APDataStore/pact/ddm_dx1/retained_mirror` — **15 files, 154,992 B, every sha256
re-verified after copy**, manifest `DX1_RETAINED_MIRROR_MANIFEST.json`. Nothing else on Vertigo
was read or written, and this is a COPY: it does not reclaim a byte. **The certify-and-MOVE
reclaim (sr2 machinery) is owed separately** and is not done here.

---

## 8. Custody

`/Volumes/APDataStore/pact/ddm_rr7/retained/RR7_RETENTION_MANIFEST.json` — 14 files,
263,409 B, every sha256 measured from the bytes: the T4 row, all eight harvested artifacts, the
fire manifest, the spawn record, the seal, and the reproducibility receipt. The fired runtime
tree is **not** retained as bytes: it regenerates from committed state via
`experiments/ddm_wc2c_stage_native_split_runtime.py`, which refuses on drift, and that
reproducibility was proved this session. `/Volumes/VertigoDataTier` is 100% full (890 MiB), so
retention went to APDataStore per the storage waterfall. Nothing was measured and discarded.

**Pointer: UNMOVED.** `0.14839100138338618` `[contest-CUDA T4]`, archive `f3bce5d2…`,
180,625 B. This row was never able to move it — byte identity guaranteed that — and it did not.
What it bought is the wall-clock verdict the submission was blocked on: **the accelerator does
not ship.**
