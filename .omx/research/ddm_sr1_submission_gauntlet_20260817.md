# ddm_sr1 — submission review gauntlet: 5 passes run, 14 findings, verdict RED-pending-3-blockers

**Date:** 2026-08-17
**Axis:** `[macOS-CPU advisory / scorer-free exact byte + custody verification]`.
`score_claim: false`, `promotable: false`. **No Modal, no dispatch, no scorer run, no eval.
Spend: $0.00.** No submission, no publication, no GitHub write. `upstream/` untouched.

---

## ANSWER

**RED. Counter 0/5 clean.** Three blockers must clear before the PR opens; none is
arm-clearable and one is operator-only. The packet's *technical* custody is excellent —
archive bytes, score arithmetic, and the evaluated receiver all re-derive exactly and
independently. What fails is the **public-facing claim surface**: the packet points a judge at
a private repository, offers a script that does not exist at the revision it pins, and ships 63
local absolute paths behind a hygiene check that only ever scanned one file.

**The decisive finding is B1: `adpena/comma-lab` is PRIVATE.** Both the shipped `README.md` and
the PR body assert it as "Public source repository" with a pinned commit URL. A judge clicking
either gets 404. The README itself carried this as an open item; the answer is now measured.

**The instrument that should have caught B3 is vacuous.** The compliance receipt reports
`public_hygiene: {hits: [], scan_paths: ["README.md"]}` — one file out of 34. `hits: []` is
skip-equals-green, not cleanliness. My own grep over the whole tree found 63 leaked paths. This
is the vacuity==PASS genus firing on our own submission gate.

---

## 1. STORES CONSULTED

* `.omx/research/charters/ddm_sr1_submission_review_gauntlet_20260817.md` — the charter.
* `.omx/research/ddm_pq2_packet_polish_20260817.md` — the packet under review.
* `.omx/research/ddm_sr1_submission_mechanics_runbook_20260817.md` — MAIN's pass-4 object.
* `.omx/research/ddm_pq1_submission_packet_prep_20260815/{PR_BODY_DRAFT,BORROWED_SUBSTRATE_ACCOUNTING}.md`
* `experiments/results/ddm_rr4_cuda_exact_contest_cuda_20260817_r1/returned_artifacts/` — the
  recovered exact-authority payload (`contest_auth_eval.json`, `report.txt`, `provenance`).
* `/Volumes/APDataStore/pact/ddm_pq2/{submission_staging,compliance_rr4_gen2.json,competitive_statement.txt}`
* `.omx/state/canonical_frontier_pointer.json` — leaderboard snapshot for the competitive claim.
* Source read directly: `scripts/pre_submission_compliance_check.py`,
  `src/tac/phase1_packet_compiler.py`, `experiments/ddm_pq2_compress_e2e.py`,
  `experiments/ddm_rr2_{encoder_byteclose,receiver_close}.py`, `upstream/evaluate.py`.

## 2. Pass ledger

| Pass | Scope | Findings | Clean? |
|---|---|---|---|
| 1 COMPLIANCE | custody chain, rule-118, payload cleanliness, dep closure, hygiene | B3, F1, F2, F3, F4 | NO |
| 2 PR BODY + HONESTY | every number traced, NO-FAKE #7, claim language, Innovation Gate | B1, B2, F5, F6, F7, F8 | NO |
| 3 COMPRESSION SCRIPT | judge read-through, CLI exercise, sanitization, sha-assertion logic | F9, F10 | NO |
| 4 MECHANICS DRY-RUN | gh flags, default branch, asset-URL flow, refusal conditions | F11, F12, F13 | NO |
| 5 FINAL SWEEP | hostile-maintainer read, runtime assumptions, cross-doc consistency | F14, **F15**, **F16** | NO |

**Counter: 0 consecutive clean passes.** Every pass produced at least one finding, so no pass
advanced the counter. Per the charter, findings reset; the counter is honestly 0.

**Passes 3 and 5 were re-run at full depth** after the clock freeze lifted (operator EXHAUST-ALL,
2026-08-17). The deeper run strengthened pass 3's verdict (the judge-facing CLI was exercised,
not just read) and produced two new pass-5 findings, F15 and F16 — which is the argument for
running them at depth rather than under deadline compression.

## 3. Findings

Severity: **B** = blocker (must clear before publish) · **M** = medium · **L** = low/advisory.
"Swap-safe" = survives an fx1 hot-swap unchanged; "swap-bound" = must re-run on a new candidate.

### B1 — `adpena/comma-lab` is PRIVATE, and two shipped documents call it public *(swap-safe, operator-only)*

Measured: `gh repo view adpena/comma-lab --json isPrivate` → `{"isPrivate":true,"visibility":"PRIVATE"}`.
Asserted in `README.md:33-35` and `PR_BODY_DRAFT.md:158-159` as "Public source repository:
https://github.com/adpena/comma-lab" plus a pinned commit URL. Anonymous fetch 404s.

Two branches, both operator:
1. **Make `comma-lab` public.** This publishes the entire `.omx` tree, arm charters, memory
   files, and the `/Volumes` defaults in F9. Post-contest that is permitted by the open-source
   grant, but it is a deliberate act with a large surface, not a checkbox.
2. **Drop the claim.** Remove the source-repository and pinned-revision lines from README and
   PR body. The submission still stands: the archive, receiver, and report are self-contained.

Branch 2 also dissolves B2 and F9, which exist only because a source pin is claimed.

### B2 — the offered compression entry point does not exist at the pinned revision *(swap-safe)*

```
git cat-file -e e7ca85754bb9e6a4b319e5a8fa206366c90bd6f4:experiments/ddm_pq2_compress_e2e.py
fatal: path 'experiments/ddm_pq2_compress_e2e.py' exists on disk, but not in 'e7ca8575...'
```

It landed later, in `a411f612aa`. The PR body pins `e7ca8575` as the source revision *and*
names `ddm_pq2_compress_e2e.py` as the merge-offered entry point. A judge following the pin
cannot find the script.

**Recommended cure — two labelled pins, not a re-pin.** `e7ca8575` is the commit the T4 row
actually ran from (`provenance.pact_commit` in the receipt confirms it). Re-pinning to one
newer commit would misstate what was evaluated. Ship both:
`evaluation source pin = e7ca8575` and `compression-script pin = a411f612aa`.

### B3 — 63 local absolute paths ship inside the hashed runtime tree, under a vacuous hygiene green *(swap-bound in count, swap-safe in kind)*

| File | `/Volumes/` occurrences | In hashed 32-file tree? | Read by any runtime code? |
|---|---|---|---|
| `GENERATION_RECEIPT.json` | 16 | yes | **no** |
| `RECEIVER_PARSEBACK.json` | 47 | yes | **no** |

All of form `/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/…`,
exposing external-drive names and internal arm codenames. `grep -rn "GENERATION_RECEIPT\|RECEIVER_PARSEBACK"`
over `runtime/`, `cpr1/`, `inflate.py`, `inflate.sh` returns **nothing** — they are inert
payload, shipped only because the evaluated tree hash was computed over them. pq2 was right not
to edit them.

**Why the gate did not catch it:** `compliance_rr4_gen2.json → public_hygiene` is
`{"hits": [], "scan_paths": ["/Volumes/.../submission_staging/README.md"]}`. One file. And the
`submission_runtime_has_no_network_install_or_local_paths` red fired on
`forbidden_side_effect_hits = [inflate.sh:27 uv pip install]` **only** — the local-paths half
never fired because the scan never reached the JSONs. MAIN's dep-bootstrap adjudication
therefore covers the red *as it actually fired*, but does **not** cover this leak, which is
currently unadjudicated.

Cheapest hash-safe cure: one disclosure paragraph in `README.md` (excluded from the hashed 32,
so free). Removing the paths would change the pinned tree hash and break replay.

### F1 — `README.md:69` points at a file that is not in the packet *(M, swap-safe)*

"Read `BORROWED_SUBSTRATE_ACCOUNTING.md` before treating any of the learned content as ours."
That file exists at `.omx/research/ddm_pq1_submission_packet_prep_20260815/` but was **not
staged**. A judge sees a dangling pointer on the one document that governs the originality
claim — the worst possible file to have missing.

Two cures, either fine: copy it into the submission directory at PR-assembly time (it is not in
the hashed 32, so this is hash-safe), or reword to point at the PR body's inline table, which
already carries the full accounting. **I did not edit** — README is likely to be rewritten
wholesale under B1, and a hot swap regenerates the packet.

### F2 — the two stale receipts remain a hostile-read hazard *(L, swap-safe)*

`GENERATION_RECEIPT.json` / `RECEIVER_PARSEBACK.json` declare a 182,759-byte archive with
sha `80d9c8c6…` — **not** the shipped 181,161-byte `35ac2b9b…`. The README discloses this
clearly and correctly (lines 40-58), and the reasoning (editing them breaks the evaluated tree
hash) is sound. Recorded as residual risk, not a defect: a maintainer who reads the JSON before
the README sees a contradiction. The README's placement of the correction is the right call.

### F3 — every staged file is mode 0700 *(M, swap-safe)*

All 34 files report mode `0700`; the canonical packet contract expects `0644` (and `0755` for
`inflate.sh`). This is an ExFAT artifact of the staging volume. **Corrected mid-review:** I
first read this as a tree-hash break, then re-derived — the compliance check hashes a
*manifest* (`relative_path`/`bytes`/`sha256`), not the filesystem, so mode is **not** in the
pinned hash. Custody is intact. The residual issue is packaging: git records the executable bit,
so a bare copy marks every `.py` and `.json` executable in the PR. Fix at copy time:
`chmod 644` the tree, `chmod 755 inflate.sh`.

### F4 — AppleDouble residue confirmed *(finding, counter reset; already armed)*

Exactly one: `._README.md` (4,096 B) in the staged root, created 14:55 when README.md was last
edited. No `.DS_Store`, no `__MACOSX`. pq2 documented this bug class and it recurred, which is
the point: **any edit to a staged file on this ExFAT volume re-creates it.** MAIN's runbook
rsync-exclude form handles it; the standing hazard is that a bare `cp -r` at any future step
ships it. Verified count after exclusion: 0.

### F5 — the "asserts byte-identical" claim overstates what the code does *(M, swap-safe)*

`PR_BODY_DRAFT.md:81-82`: "It also writes a second archive from the same member and **asserts**
the two are byte-identical." The code computes `determinism_repeat_byte_identical` and
`token_stream_matches` and **reports** them in the result JSON; only `sha_ok and bytes_ok` gate
the non-zero exit (`ddm_pq2_compress_e2e.py:265-269`). A determinism-repeat mismatch would print
and still exit 0.

Practically harmless — the archive is derived from the token stream, so a token mismatch that
still yields the correct archive sha is not reachable. But the PR body claims an assertion the
code does not make. Cure either way: add the assert, or change "asserts" to "records". Given the
honesty bar, adding the assert is better and is three lines.

### F6 — the competitive claim is true but unverifiable by the reader *(M, swap-safe)*

`PR_BODY_DRAFT.md:111` — "below the best score on the leaderboard at the time of writing." The
claim is **TRUE**: the official ranked table in `canonical_frontier_pointer.json` has PR135
`semantic-pose-HPAC_CPR1_polished` at rank 1, **0.162**; ours is 0.15853325034789678. But the
number `0.162` appears in **no** shipped document (verified across README, report.txt, PR body,
borrowed-substrate accounting). Under the Innovation Gate a competitive statement should be
*unquestionable*, which means checkable without trusting us. Name it: "below PR #135's
leaderboard-ranked 0.162."

### F7 — the 2-decimal display collapses the win *(L, advisory — currently handled well)*

`upstream/evaluate.py` prints `Final score: … = {score:.2f}` → the recovered authority
`report.txt` literally reads **0.16**. PR135 is displayed as 0.162. A maintainer comparing
printed values sees 0.16 vs 0.162. Our packet already handles this honestly: staged `report.txt`
prints both "Recomputed score: 0.15853325034789678" and "Reported (2 dp display): 0.16", and the
PR title uses full precision. **No change required** — recorded so nobody later "simplifies" the
report to the 2 dp figure.

### F8 — borrowed-substrate accounting: COMPLETE and honest *(pass, no defect)*

Audited against NO-FAKE #7. The table is mechanism-level, categories are closed, and it is
deliberately unflattering: 7 of 8 sections classified `PR130/135-byte-identical`, the HPAC object
`PR130-lineage`, the encoder-side RC64 backend disclosed as an unmodified PR135 borrow, and the
shipped receiver backend explicitly flagged `PR135-lineage-modified` — "stated explicitly so the
difference is not mistaken for either full originality or a clean copy." The claim is narrowed to
the one thing that is ours: the zero-byte decode-time corrector. The "what we do not claim"
section refuses to present unchanged `d_seg`/`d_pose` as an achievement. **This is the strongest
document in the packet.** Its only problem is F1 — it is not shipped.

Claim-language audit also passes: "author-claimed and not yet evaluated by the maintainers, **as
is ours until this PR is run**" — correctly author-asserted, no eval-bot language, matching the
PR137/138 field norm. GPU-eval declaration present and honest (`README`, PR body §"does your
submission require gpu").

### F9 — the offered stage scripts carry 7 hardcoded local-path defaults *(M, swap-safe)*

The entry point `experiments/ddm_pq2_compress_e2e.py` is **clean** (grep for
`/Users/|/Volumes/|APDataStore|VertigoDataTier|tailscale|modal|vast` → zero hits), and the PR
body's "The script contains no local filesystem layout" is true *of it*. But the PR body also
names its stage scripts, and those carry:

* `ddm_rr2_encoder_byteclose.py` — 5 hits, incl.
  `_env_path("TAC_PQ2_HM1_DIR", "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained")` and a
  `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/…` path that also reveals a PR135 intake tree.
* `ddm_rr2_receiver_close.py` — 2 hits, incl. `STORE = Path("/Volumes/APDataStore/pact/ddm_rr2_encoder_build")`,
  which is **not** env-overridable.
* `ddm_rr4_free_corrector_v2.py` — 0 hits, clean.

pq2's `_env_path` refactor made the roots overridable; the **defaults** are still literal local
paths. Functionally fine. This only becomes a disclosure event under B1 branch 1 (making
`comma-lab` public), which is exactly why B1 and F9 should be decided together.

### F10 — sha-assertion logic: CORRECT and fail-closed *(pass, no defect)*

Re-derived rather than trusted. `ok = expected is None or measured == expected` handles the one
unpinned input explicitly rather than silently; `if problems: raise SystemExit` on input
verification; `if not (sha_ok and bytes_ok): raise SystemExit("ARCHIVE VERIFICATION FAILED")`.
Constants pinned at module scope (`EXPECTED_ARCHIVE_SHA256`, `_BYTES`, `_TOKEN_SHA256`,
`_DECODED_FIELD_SHA256`) with the comment "they are assertions, never defaults to fall back on."
The only gap is F5.

### F11 — runbook refusal condition cites the stale receipt *(M, swap-safe)*

`ddm_sr1_submission_mechanics_runbook_20260817.md:87`: "staged-tree sha mismatch vs
`GENERATION_RECEIPT.json` at copy time". That receipt declares 182,759 B / `80d9c8c6…`; the
staged archive is 181,161 B / `35ac2b9b…`. **Executed literally this refusal fires 100% of the
time on a correct packet** — a guard that always trips is a guard that gets disabled. Should cite
`RESULT_build.json` or the recovered `contest_auth_eval.json`.

Line 90's condition ("report.txt score fields not byte-identical to the recovered
contest_auth_eval.json") also needs narrowing: the recovered `report.txt` is raw upstream output
(664 B, config + 6 result lines) while the staged one is a 2,585 B superset with identity and
reproduction blocks. They are not byte-comparable. The comparable set is the score fields, and
those **do** agree — verified below.

### F12 — `gh pr create` omits `--base` *(L)*

It defaults to the target repo's default branch, which I confirmed is `master`, so the command
works. Pin it explicitly (`--base master`) per never-invent-flags discipline.

### F13 — the judge needs one line about placing `archive.zip` *(L)*

Runbook step 1 correctly removes `archive.zip` from the committed tree (it is hosted as a release
asset). But `inflate.py:52-53` resolves `archive_path = here / "archive.zip"` and `_verify_input`
raises without it — the same hard-fail pq2 flagged as a merge blocker in PR138. Upstream
`evaluate.py:63` reads `args.submission_dir / 'archive.zip'`, so the judge must place it there
anyway; this is normal. Add one line to the README reproduction block: download the release asset
into this directory first.

### F14 — internal codename in a judge-visible error string *(L)*

`inflate.sh:24`: `echo "CP135 requires Brotli==1.2.0, but uv is unavailable"`. `F26` appears 9
more times in `inflate.sh` as env-var and file names. Harmless and arguably honest about lineage,
but "CP135" in a user-facing error will read as an unexplained internal name. Cosmetic.

### F15 — the receiver requires a C compiler at inflate time, undeclared and unguarded *(M, swap-safe)*

`inflate.sh:32` compiles the range-coder backend **unconditionally** on every run:

```bash
"${CC:-cc}" -O3 -std=c11 -shared -fPIC "$HERE/runtime/entropy/rc64_backend.c" -o "$BUILD_DIR/rc64_backend.so"
```

A working C toolchain is therefore a hard runtime dependency. It is **declared nowhere** — the
README's "Dependency closure" section names only Brotli (`grep -i "compiler\|gcc\|build-essential"`
over README → no match).

The asymmetry is the real defect. The Brotli dependency is carefully fail-closed: guard, explicit
`exit 69`, and the message "CP135 requires Brotli==1.2.0, but uv is unavailable". The compiler
dependency has **no guard at all** — on a slim image the bare command dies under `set -e` with a
raw shell error and no diagnosis. Two dependencies, two standards.

Empirically it is satisfied: the T4 run inflated in 476.611040218 s, so the evaluation image has
`cc`. This is a robustness and disclosure finding, not a correctness one.

Cure is split by candidate. For rr4 **as-is**, guarding `inflate.sh` would change the runtime
tree hash and break the custody binding to the measured T4 row — so the only hash-safe cure is to
declare the dependency in `README.md` (excluded from the hashed 32). For **any new candidate**
built under EXHAUST-ALL, add the `command -v cc` guard with a clear message so the two
dependencies are held to the same fail-closed standard.

### F16 — dead macOS/homebrew branch in a judge-visible script *(L, swap-safe)*

`inflate.sh:41-52` carries a `case "$(uname -s)" in Darwin)` branch that shells out to
`brew --prefix libomp`. It is unreachable on the contest runner: it requires
`F26_TOKEN_DECODER == "native-hpac"`, and line 36 defaults that to `python` with nothing setting
it otherwise. Verified by reading the only two references to the variable.

Harmless, but a maintainer reading the receiver top-to-bottom encounters homebrew and may
reasonably ask whether the submission assumes macOS. One comment marking the branch as an
optional local acceleration path — or deleting it in the next candidate — removes the question.

## 4. What VERIFIED clean (re-derived independently, not confirmed from receipts)

| Check | Method | Result |
|---|---|---|
| Archive identity | re-hashed the staged bytes | `35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956`, 181,161 B ✓ |
| Member | `zipfile` introspection | `p`, 181,061 B, **stored** (compress_type 0), CRC32 885609521, sha `1a6b40cc…` ✓ |
| Score | recomputed from components | `100·0.00029611 + √(10·0.00000688) + 25·181161/37545489` = **0.15853325034789678**, exact float equality to the receipt's `canonical_score` ✓ |
| Denominator | recovered raw report | "Original uncompressed size: 37,545,489 bytes" ✓ |
| **Receiver ↔ evaluated receiver** | sha of staged `inflate.sh` | `e1b3df4d9178a1572cf04bc2dd9d2ddcc8f6157deac8ec1c40e89b2114522d62` = `provenance.inflate_script_sha256` **exactly** — the critical custody link ✓ |
| Portable runtime tree | compliance manifest | `4358aaf34fcbfc1cdc4a8865b9aead709199465c9909321abf279ebcd0fe3721` reproduces, matches README + report ✓ |
| Runtime file count | canonical walker | exactly **32**, matching the auth-eval manifest ✓ |
| **Rule 118** | AST scan of all shipped `.py` + literal-density on `.c` | **CLEAN** — every literal >512 chars is a docstring (verified via `ast.get_docstring` identity); no numeric sequence ≥64 elements; `.c` digit density 3.5–4.9% = ordinary code, no embedded tables ✓ |
| **Hygiene of the COUNTED bytes** | byte-scan of the `p` member (181,061 B) | **PRISTINE** — zero occurrences of `/Users/`, `/Volumes/`, `/home/`, any URL, `tailscale`, `adpena`, `.py`; and **zero printable ASCII runs ≥16 chars in the entire payload**. The scored bytes are pure entropy-coded binary with no embedded text of any kind ✓ |
| Judge-facing CLI | executed `--help` and `--emit-inputs-template` | both run clean; the emitted schema contains **placeholder paths only** (`<absolute path to the directory>`), no local layout; each input carries `role` + `expected_sha256`, and `rc64_source` self-discloses "(inherited substrate; see the accounting table)" ✓ |
| Scorer imports | compliance receipt | `scorer_import_hits: []`, `disallowed_runtime_imports: []` ✓ |
| Dependency closure | compliance + pq2 smoke | single self-installed dep `Brotli==1.2.0`, pinned wheel only, fail-closed exit 69 ✓ |
| Eval binding | recovered receipt | `n_samples 600`, `device cuda`, `Tesla T4`, `gpu_t4_match true`, `exact_cuda_eval_complete true`, upstream snapshot + `evaluate.py` sha both match the packet ✓ |
| Competitive claim | frontier pointer | leaderboard rank 1 = PR135 @ 0.162 > our 0.15853 → claim TRUE ✓ |
| Cross-document numbers | 12 values × 4 documents | consistent throughout; no contradiction found ✓ |
| gh mechanics | `--help` on this host | `release create` `-R/-t/-n` + positional files; `pr create` `-R/-H/-t/-F/-B` — all real ✓ |
| Default branch | `gh repo view` | **`master`** — runbook's `upstream/master` correct ✓ |
| Fork visibility | `gh repo view` | `adpena/comma_video_compression_challenge` **PUBLIC** — asset-URL flow works ✓ |
| Compliance | pq2 receipt re-read | 81/85; the 4 reds are exactly MAIN's four adjudications ✓ |

## 5. Hot-swap partition (freeze ~03:00Z, fx1 probability HIGH)

**Survive a swap unchanged (fix now — they are not candidate work):**
B1, B2, F1, F2, F3, F4, F5, F6, F9, F11, F12, F13, F14, F15, F16.

F15 is the one worth acting on *during* candidate construction rather than after: any new
receiver built under EXHAUST-ALL should carry the `command -v cc` guard from birth, so the
compiler dependency is fail-closed like Brotli instead of dying with a raw shell error. Retrofitting
it later changes the runtime tree hash and forces a refire.

**Must re-run on a swap (the swap-bound minimum):**
1. Archive identity — re-hash, re-read member/CRC.
2. Score re-derivation from the new receipt's components.
3. `inflate.sh` sha vs the new `provenance.inflate_script_sha256`.
4. Portable runtime tree hash + file count.
5. Rule-118 AST scan **only if** the runtime tree changes (a mixer candidate will change it).
6. B3's leak count (re-grep; the two stale receipts may or may not carry forward).
7. Every pinned number in README / report.txt / PR body.

Passes 2 (honesty structure), 3 (script logic), 4 (gh mechanics) survive a swap except for their
pinned numbers. **Estimated swap re-run: ~15 minutes.**

## 6. VERDICT — RED, with the blocker set already consumed by MAIN

**RED at the close of this gauntlet. Counter 0/5.** This verdict BANKS as the standing
pre-submission artifact under the operator's EXHAUST-ALL order; candidate-bound passes re-run at
the eventual freeze.

### Blocker disposition (MAIN adjudicated 2026-08-17, all three consumed)

| # | Blocker | Disposition | State |
|---|---|---|---|
| B1 | `comma-lab` private; two documents call it public | **SURFACED TO OPERATOR**; visibility line flagged inside `PR_BODY_DRAFT.md` so the packet cannot ship with the claim unresolved | **OPEN — operator** |
| B2 | offered entry point absent at the pinned revision | **FIXED**, `fe71c06b9a` — two labelled pins landed exactly as recommended; verified by this arm | **CLOSED** |
| B3 | 63 leaked local paths + vacuous hygiene green | **FIX-FORWARD**: the leak ships in NO candidate — fx1 emits receipts with relative paths (clean by construction); if rr4 ships, its receipts are regenerated and refired rather than disclosed | **CLOSED-BY-CONSTRUCTION** |

Runbook defects F11 and F12 also **FIXED** in `fe71c06b9a` (refusal authority → `RESULT_build.json` /
the recovered `contest_auth_eval.json`; `--base master` pinned). Verified by this arm.

**B1 is the sole remaining hard gate**, and it is operator-only. Note the coupling: if B1 resolves
as "drop the source-repo claim," F9 (local-path defaults in the stage scripts) dissolves with it,
because those paths only become a disclosure event if `comma-lab` is published.

### Residual risk list for the operator's final confirm

1. **B1 — the public-source decision.** Publishing `comma-lab` exposes the whole `.omx` tree,
   arm charters, memory, and the `/Volumes` defaults of F9. Dropping the claim costs nothing
   technically: the archive, receiver, and report are self-contained.
2. **F1** — `README.md:69` still points at an unstaged `BORROWED_SUBSTRATE_ACCOUNTING.md`. The
   packet's strongest honesty document is currently a dangling link.
3. **F5** — the PR body says the script "asserts" determinism; the code only records it. Add the
   assert or change the verb.
4. **F6** — the competitive claim is true but names no number. Cite PR #135 @ 0.162 so a reader
   can check it without trusting us.
5. **F15** — the C-compiler dependency is undeclared and, unlike Brotli, unguarded.
6. **F3** — mode 0700 on every staged file; `chmod` at copy or the PR marks all `.py`/`.json`
   executable.
7. **F2 / F14 / F16** — cosmetic hostile-read items: stale receipts contradict the archive until
   the README is read, `CP135` appears in a user-facing error, and a dead homebrew branch invites
   a "does this need macOS?" question.

### The standing apparatus debt this gauntlet found

`public_hygiene` in the submission gate scans **one file**. A gate that reports `hits: []` after
scanning `README.md` alone is not measuring cleanliness — it is measuring nothing and printing
green. It missed 63 leaked paths on our own submission. Named as a two-landing debt (fix + a check
that refuses a hygiene result whose `scan_paths` does not cover the tree). **A gate that scans one
file is a gate that lies**, and the honest report of it is that our own instrument passed a packet
this arm's `grep` failed.

## 7. What I did NOT do

No submission, no PR, no release, no hosting, no publication, no GitHub write of any kind. No
Modal or paid dispatch. No scorer run, no eval, no n600. No edits to `upstream/`. No edits to the
staged packet (every finding is flagged with a recommended cure; a hot swap regenerates the tree
and B1 likely rewrites the README wholesale, so editing now would collide with MAIN). No writes
to VertigoDataTier. Payloads retained — the staged tree is unmodified.

## 8. NEXT

| # | Item | State | Owner |
|---|---|---|---|
| 1 | B1: decide `comma-lab` public vs drop the claim | **BLOCKED on operator** — sole remaining hard gate | operator |
| 2 | B2 dual pins + F11/F12 runbook fixes | **CLOSED** `fe71c06b9a`, verified by sr1 | MAIN |
| 3 | B3 leak: emit receipts with relative paths in every candidate | **CLOSED-BY-CONSTRUCTION** (fx1 instructed; rr4 regenerate-and-refire if it ships) | MAIN / fx1 |
| 4 | F1 (stage or reword accounting pointer), F5 (assert or reword), F6 (name 0.162) | **OWED** | MAIN |
| 5 | F3 `chmod 644` tree + `755 inflate.sh` at copy time | **OWED** | MAIN |
| 6 | F15: `command -v cc` guard in the next candidate's receiver (build it in, do not retrofit) | **OWED** | candidate builder |
| 7 | Two-landing fix: `public_hygiene` must walk the tree + a check refusing a hygiene result whose `scan_paths` does not cover it | **NAMED DEBT** | runtime owner |
| 8 | Swap-bound re-run at the eventual freeze | **ARMED**, ~15 min, 7 items listed in §5 | sr1 |

---

## Artifacts (ALWAYS KEEP THE PAYLOAD)

* This memo — the pass ledger, all 14 findings with measurements, the swap partition.
* Staged tree unmodified at `/Volumes/APDataStore/pact/ddm_pq2/submission_staging/`
  (archive `35ac2b9beb…`, 181,161 B — re-hashed by this arm, byte-identical).
* All verification commands are inline above and re-runnable; no scratch artifacts were created.

**Own-vehicle frontier: S 0.15853325034789678 @ 181,161 B `[contest-CUDA T4, n600]`.
This unit did not move it — a review gauntlet measures the packet, it does not lower the score.**
