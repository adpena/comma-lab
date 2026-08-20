# ddm_rv17 — wave-end adversarial review, ROUND 1: findings round, counter 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [receipt + source review, scorer-free]` ·
`score_claim: false` · cost $0 · counter authority for the #1157 wave-end 3-pass cycle
(the packet's own `ADVERSARIAL_REVIEW_SCAFFOLD.md` 0/5 counter is a SEPARATE authority and is
untouched by this memo).

## THE ANSWER, FIRST

**Round 1 is a FINDINGS round. Counter 0/3.** The prior-law prediction (every review round this
week found ≥1 real defect) is CONFIRMED, not falsified: 7 findings from my own re-derivation and
13 from a supervised code-review arm, none manufactured.

**The score is sound.** The sixteenth-move row re-derives EXACTLY. What is defective is the
*record*: the memo that anchors the pointer move makes a false negative-existence claim about its
own receipt, and in doing so throws away the two things the campaign's critical path most needs —
a token-level decode-identity proof and the measured inflate wall.

**The single highest-value correction (RV17-F1):** the rc2 receipt DOES carry `raw_sha256`,
a full `stage_seconds` split, `decoded_token_sha256`, and `inflate_elapsed_seconds`
= **458.752594349 s**, plus the harness's own printed verdict
`Wall budget: PASS [contest-CUDA] … 498.476 s charged <= 822 s cold-cache ceiling`. The memo says
none of that exists. The packet is holding **24 `GATED-ON-RC2` placeholders** waiting for exactly
these numbers.

---

## PER-ITEM VERDICT ROWS

| # | Item | Method | MEASURED result | Verdict |
|---|---|---|---|---|
| 1 | rc2 sixteenth-move row | recompute S from components; delta arithmetic; pointer diff | S = **0.14827847122030852** exactly; −169 B; ΔS −1.125302e-4 | **2 FINDINGS** (F1 HIGH, F2 LOW) — score itself CLEAN |
| 2 | CPU-leg wall adjudication | cross-axis token-sha comparison | `decoded_token_sha256` **MATCHES** CUDA exactly | **CLEAN — and stronger than claimed** |
| 3 | sw1 portable paths + #208 guard | supervised arm + my own re-verification | 13 findings; 25 tests not 35; whole-blob scan | **FINDINGS** (see arm rows) |
| 4 | secrets guard (step 1i) | supervised arm, both control directions | see §Item 4 | **see §Item 4** |
| 5 | sw2 sweep verdict | re-derive triage from redacted receipt | 38,869 breakdown EXACT; counts unreconciled | **2 FINDINGS** (F3, F4) + 1 LOW (F5) |
| 6 | pq9 packet polish | manifest verify; custody-pairing grep | 33/33 OK; no false pairing | **2 FINDINGS** (F6 MED, F7 LOW) |

---

## ITEM 1 — the sixteenth-move row

**CLEAN (the score).** Recomputed from components, never the `final_score` display field
(Catalog #877, which reads `0.15`):

```
100 × 0.00020139            = 0.020139
sqrt(10 × 6.37e-06)         = 0.007981227975693965
25 × 180456 / 37_545_489    = 0.12015824324461455
                              ─────────────────────
S                           = 0.14827847122030852   ← equals the receipt field exactly
```

Archive sha `df7fd266…9e2080`, 180,456 B, `gpu_t4_match: True`, n600, `evidence_grade:
contest-CUDA`. Delta arithmetic: 180,456 − 180,625 = **−169 B**; ΔS = **−1.1253016307766e-4**;
rate-term delta = −1.1253016307765e-4; residual **−1.5e-17** (float noise). The "entire delta is
the rate term" claim is MEASURED-true by an independent route: feeding the *identical* components
with 180,625 B reproduces jg5's published `0.14839100138338618` exactly. `2.90×` checks
(1491.560144927 / 513.7945766859999 = 2.9031). Pointer file consistent: `effective_frontier` =
`our_local_frontier_contest_cuda` = 0.14827847122030852, same sha, lane
`lane_ddm_rc2_composed_cuda_20260820`.

### RV17-F1 — HIGH — a false negative-existence claim discards the critical-path evidence

`.omx/research/ddm_rc2_t4_row_sixteenth_move_20260820.md` §"DECODE IDENTITY" + §"THE DECODE WALL";
propagated verbatim into `.omx/state/main_hot_state.md` POINTER_LINE ("per-stage split not
emitted").

The memo states: *"This receipt schema does not emit a raw sha or per-stage timing split"* and
*"the composed tree's T4 inflate-only wall is bounded above by 513.8 s minus evaluate but NOT
separately measured."* Both are FALSE of the receipt as a whole. MEASURED from
`t4_row_r2/MODAL_REMOTE_RESULT.json`:

| Claimed absent | Actually present |
|---|---|
| raw sha | `raw_sha256 = 6bf8acf8d4412e43…`, `raw_bytes = 3,662,409,600` |
| per-stage split | `stage_seconds = {archive_setup 0.565, frame0_selector_and_io 3.609, neural_render_and_resize 41.950, render_checkpoint_copy 0.0, token_decode_or_checkpoint_load 397.877, total 454.596}` |
| inflate wall "not separately measured" | `inflate_elapsed_seconds = 458.752594349`, `evaluate_elapsed_seconds = 39.723591300` |
| — | `Wall budget: PASS [contest-CUDA] … 498.476 s charged <= 822 s cold-cache ceiling` |
| identity only "at COMPONENT level" | `decoded_token_sha256`, `corrected_cdf_input_sha256`, `corrected_quantized_logit_sha256`, `decoder_bit_position` all emitted |

The claim is true only of the receipt's *top-level* keys; the data sits one level down, inside
`artifacts.contest_auth_eval.{json,stdout.log}`. Three real costs: (a) the memo settles for a
weaker identity proof than it owns — token-level identity was available and, as Item 2 shows,
MATCHES; (b) it substitutes a derived bound (513.8 − 39.7 = 474.07 s) for a measured 458.75 s,
15.3 s looser; (c) the SHIP WALL — the campaign's stated critical path — was **already
adjudicated PASS by the harness** against the tight 822 s ceiling, while the memo reasons instead
from wc2's 498–978 s *projection* against the loose 1,800 s wall.

This is the named `#1 false-claim class` (negative-existence without exhaustive search). Genus:
`the_denominator_and_the_falsifier_can_both_be_vacuous`.

**CURE** (owner: MAIN / pq9): append a correction to the rc2 memo and the hot-state POINTER_LINE
recording the four measured values above; fill the packet's 24 `GATED-ON-RC2` fields from them at
swap; cite the harness's own wall verdict rather than re-deriving a looser one.

### RV17-F2 — LOW — "2 ULP" is 1 ULP

Same memo, §THE ANSWER FIRST; also hot state. MEASURED: projection `0.14827847122030854` minus
measurement `0.14827847122030852` = `2.7755575615628914e-17`, and
`math.ulp(0.14827847122030852)` = `2.7755575615628914e-17` — exactly **1 ULP**, not 2. Error in
the conservative direction, but a precision memo should carry the precise figure.
**CURE:** one-word edit.

---

## ITEM 2 — the CPU-leg wall-infeasible adjudication — **CLEAN, and stronger than claimed**

The adjudication ("pure wall failure, NOT desync") is MEASURED-correct, and the evidence is better
than the hot state's wording ("decoded-token sha emitted") suggests. It does not merely exist — it
**matches**:

| field | CUDA leg | CPU leg | match |
|---|---|---|---|
| `decoded_token_sha256` | `cc10a7b09353c0af…` | `cc10a7b09353c0af…` | **YES** |
| `corrected_cdf_input_sha256` | `370a5e2a85ccbb1e…` | same | **YES** |
| `corrected_quantized_logit_sha256` | `8269fe1aad031620…` | same | **YES** |
| `decoder_bit_position` | 910837 | 910837 | **YES** |
| `raw_sha256` | `6bf8acf8d4412e43…` | `2fc5dd3d7547f2e4…` | no — expected |
| `raw_bytes` | 3,662,409,600 | 3,662,409,600 | YES |

The rr2 (#1096) failure mode is *precisely* CPU-prob encode vs CUDA-prob decode desync, so a
cross-axis token-sha match is the exact discriminator, and it excludes desync. The `raw_sha`
divergence is expected and is not evidence of anything: the neural render differs in float between
CPU and CUDA.

The timeout reading holds: `subprocess.TimeoutExpired … timed out after 1799.99997045 seconds`
against the harness's 1,800 s contest budget, with `token_decode_or_checkpoint_load = 2,427.166 s`
— i.e. token decode ALONE exceeds the whole budget by 1.35×. One structural note for the record
(not a finding): the report shows `total_including_raw_sha256 = 2,850.78 s` and
`modal_elapsed = 2,859.06 s`, so the orphaned child outlived the parent's timeout kill and ran to
completion. That is *why* a full report exists for a run the harness recorded as failed — worth
stating explicitly wherever this receipt is cited, so no future reader treats the 2,850 s report
as contradicting the 1,800 s timeout.

---

## ITEM 3 — sw1 portable-path resolver + Catalog #208 guard (commit `f0bb5d90c9`)

Reviewed by a supervised arm; I re-verified the load-bearing rows myself. **13 findings.**
Highest first; MED/HIGH rows I personally reproduced are marked ✓rv17.

| id | sev | file:line | finding | evidence |
|---|---|---|---|---|
| SW1-F1 ✓rv17 | MED | `payload_retention.py:132` | `portable_path_form` raises bare `RuntimeError` (not `PayloadRetentionError`) on `~<unknown-user>/…`; reached from `retain_payload:245` AFTER bytes are on disk — payload survives, custody row dies | `portable_path_form('~notauser/x')` → `RuntimeError: Could not determine home directory.` |
| SW1-F3 ✓rv17 | MED→HIGH at strict-flip | `preflight.py:19031` | scans the WHOLE staged blob, so touching one line reports every pre-existing absolute path in the file — the name promises new/modified, the code delivers file-level | source iterates `text.splitlines()`; 8,448 tracked files / 248,733 matching lines are in scope |
| SW1-F4 | MED→HIGH at strict-flip | `preflight.py:19087` | no vendored/intake or seal-pinned-custody exclusion; public-PR intake clones get flagged, and the only in-file remedy (inline waiver) is itself forbidden by CLAUDE.md → deadlock | staged intake clone path reported |
| SW1-F5 ✓rv17 | MED | `preflight.py:19095` | VACUITY==PASS outside a commit: nothing staged → denominator 0 → prints `OK`; denominator only shown when `verbose=True` | 0 staged now; returns `[]` with `OK` |
| SW1-F6 ✓rv17 | MED | commit message | claims "35 tests green"; the 3 test files the commit touched collect **25** | `pytest --collect-only -q` → `25 tests collected` |
| SW1-F2 | MED | `payload_retention.py:72` vs `checkpoint_retention.py:41`, `operator_storage_waterfall.py:22` | `PACT_TIER1/2` honored by ONE module; setting it relocates payload retention but not checkpoint retention or the storage waterfall — split-brain, latent while unset | tier tuples diverge under injected env |
| SW1-F7 | LOW-MED | `preflight.py:19015` | waiver over-match: token inside a string literal waives; one waiver covers every path on the line; substance bar is only `len ≥ 8` | 3 cases reproduced |
| SW1-F8 | LOW-MED | `preflight.py:18996` | detector misses `/home/<user>` (every Modal container and 2 fleet hosts are Linux), `/private/var/folders/`, Windows paths | 4 negatives |
| SW1-F9 | LOW | `preflight.py:19049` | non-ASCII filenames silently dropped AND the denominator is not incremented (`quotePath` quoting breaks `git show :file`) | `café.py` violation vanished |
| SW1-F10 | LOW | `payload_retention.py:72` | relative `PACT_TIER1` breaks the round-trip: `F(R(x)) != x` | token not recovered |
| SW1-F11 | LOW | `payload_retention.py:84` | `SSD_TIERS` frozen at import — staleness foot-gun | docstring already warns |
| SW1-F12 | LOW | `payload_retention.py:153` | `portable_retention_record` does not thread `environ=` like every sibling — untestable under injected env | signature diff |
| SW1-F13 | LOW | guard messaging | both live warn hits are durable SSD receipt paths that CLAUDE.md *requires*; message should name `$PACT_TIER1/…` as the compatible form | 2 live warns |

**DISPROVED (recorded so the next round does not re-hunt):** prefix collision is ABSENT
(`/Volumes/APDataStore/pactfoo/x` is untouched; the code uses component-wise `Path.relative_to`,
never `str.startswith`) ✓rv17; `R(F(p)) == p` holds for every absolute form tested including
embedded literal tokens; symlinked tier roots are symmetric; empty-string env falls back
correctly; nested/identical tiers resolve deterministically (longest-root wins); **Catalog #287
placeholder rejection IS enforced** (`<rationale>`, `reason`, `TODO` all refused, case-insensitive);
waiver is correctly same-line and case-sensitive; staged-blob-over-worktree precedence is real;
staged deletions excluded and binary blobs skipped without inflating the denominator; the patch
`e8c9b955…` matches the commit at file-set and ±line granularity; positive control genuine
(0 violations across its own 15 blobs).

**Strict-flip is BLOCKED** on SW1-F3 + F4 + F5 jointly. The gate cannot graduate as written.

---

## ITEM 4 — the blocking staged-secrets guard (commit `db153b5073`)

**The guard is real, correctly placed, and both control directions genuinely pass.** The suspected
deprecation time-bomb is **DISPROVED** — a removed subcommand fails loud and closed. But the guard
is **fail-OPEN on the most likely real-world failure**, which contradicts its own "BLOCKING"
framing. **10 findings.** Rows I reproduced at source myself are ✓rv17.

| id | sev | file:line | finding | evidence |
|---|---|---|---|---|
| SEC-F1 ✓rv17 | **HIGH** | `preflight_hook.py:1634-1641` | missing `gitleaks` binary → `return 0`. On any machine without the binary the guard passes every commit. Nothing anywhere asserts gitleaks is installed | `except FileNotFoundError: … return 0`; arm measured `FAITHFUL_NOBIN_RC = 0` with 1 staged file unscanned |
| SEC-F2 ✓rv17 | MED | `preflight_hook.py:1642-1648` + tool-error branch | timeout → `return 0`, and the printed text literally says **"not a pass"** while returning a pass; tool error rc=2 same shape | arm measured rc=2 → `MODE=rc2 RC=0` |
| SEC-F5 ✓rv17 | MED | `preflight_hook.py:1620` | `gitleaks protect` is **already absent** from the installed gitleaks 8.30.1 published command list (`completion, dir, git, help, stdin, version`) — the hook rides a hidden deprecated alias, while the repo's own `auto_push_main.py:277` already uses the modern `gitleaks git` | `gitleaks --help`; arm measured `protect --staged` and `git --staged` both rc=1 / 1 leak — drop-in equivalent |
| SEC-F7 ✓rv17 | MED | `preflight_hook.py:1601-1607` | `TAC_SECRETS_WAIVE=1` is the **first** statement in the function — before the staged-count read — so it is a total bypass, and it demands **no rationale** (Catalog #287) and writes to no ledger | source read; arm measured `MODE=waive RC=0` |
| SEC-F3 | MED | `preflight_hook.py:1580` | **zero tests** for a BLOCKING gate, against 6-pillar pillar-2 (≥15 tests); every sibling step has a test file | `grep -rn run_staged_secrets_scan` → only the def and its callsite |
| SEC-F4 | MED | `preflight_hook.py:1633-1637` | findings output names no file, no rule, no line — the last-20-line tail carries only a count. In the 236-file `git add` this guard exists for, the developer cannot tell which file to fix. The repo already solves this at `auto_push_main.py:284-289` (`-f json` → `RuleID @ File`) | arm measured the tail as `leaks found: 2` with no path |
| SEC-F8 | MED | `preflight_hook.py:1602-1609` | if `configs/gitleaks_pact.toml` is deleted/renamed the hook warns, falls back to gitleaks defaults, and on a clean result prints **GREEN** `0 findings` — vacuous for all five campaign shapes | arm measured bare `hf_` token: default ruleset rc=0/no leaks vs campaign ruleset rc=1/1 leak |
| SEC-F6 | LOW | `preflight_hook.py:1630-1642` | behaviour correct (blocks), message wrong: a removed subcommand exits rc=1, which collides with "leaks found", so the headline asserts a secret that does not exist — the pressure that drives people to the waiver | arm shim: `unknown command "protect"` printed under a `SECRETS IN STAGED DIFF` headline, RC=1 |
| SEC-F9 | LOW | `preflight_hook.py:1591-1600` | `-1` sentinel reaches operator text: `(-1 staged file(s) UNSCANNED …)` | measured |
| SEC-F10 | LOW | `gitleaks_pact.toml:8` | `\b(ak\|as\|tok)-[A-Za-z0-9]{20,}\b` is broad, with no allowlist and no entropy floor — low-entropy placeholders in docs will hard-block, feeding SEC-F7 | INFERRED from source |

**The sharpest form of SEC-F1/F2:** the function's own docstring names the failure it commits —
*"skip-as-green is the #875/#1050 vacuity class"* — and then implements exactly that class on three
branches (missing binary, timeout, tool error). The detector zeroes on its own cure.

**DISPROVED (recorded so round 2 does not re-hunt):**
- **Placement is correct** — callsite `:1751` precedes `run_preflight()` at `:1756`.
- **Deprecation does NOT degrade silently** — the charter's highest-value question, answered by
  test: removed subcommand → rc=1 → blocks, and the tail names the true cause.
- **The config EXTENDS, not replaces** — `[extend] useDefault = true`; a probe fired the *default*
  `generic-api-key` rule alongside the campaign `huggingface-token`. Coverage is not narrowed.
- **No over-broad allowlist** — `grep -c allowlist` → **0**; there are no allowlist stanzas at all.
- **No dead campaign regex** — all five fire (`modal-token` across all three prefixes,
  `huggingface-token`, `anthropic-key`, `tailscale-key`, `wandb-key-context`).
- **No earlier-step escape** — a file with both a ruff error and a token is refused at step 1
  (`MAIN_RC=1`); the scan is skipped, but never in a direction that admits a secret. And
  `PREFLIGHT_SKIP_RUFF=1` still reaches step 1i and blocks.
- **No `--no-verify` bypass** in repo tooling; the serializer respects the hook.
- **The gitignore trap is settled, and the memo's honest first-control failure is VERIFIED at
  source:** gitleaks does not apply `.gitignore` — `git` does, at add time. `.playwright-mcp/` is
  ignored at `.gitignore:262`, so a bare `git add` refuses, nothing stages, and gitleaks scans an
  empty index → false pass. The harness fix (non-gitignored path) is verified to work, and a
  force-added ignored file **is** scanned.

**Control-design caveat worth carrying forward:** `AKIA` + 16 `X` does **not** fire the AWS rule —
low entropy is filtered. Any future positive control must use a high-entropy synthetic string or it
will silently under-report.

---

## ITEM 5 — sw2 comprehensive sweep verdict

**CLEAN where it matters most.** The 38,869 `generic-api-key` false-positive adjudication is
MEASURED-sound and the memo's three named counts are EXACT:

| field name | memo | measured |
|---|---|---|
| `key` | 20,675 | **20,675** |
| `contract_key` | 15,640 | **15,640** |
| `evidence_key` | 1,747 | **1,747** |
| remainder (memo: "· `*_sha256`") | — | 807 = `archive_bound_candidate_contract_key` 238 · unparsed 57 · `threshold_token` 52 · `token_frame_sha256` 46 · `candidate_tokens_sha256` 38 · tail |

Also CONFIRMED: the receipt is genuinely **redacted** (`Secret == "REDACTED"` for all 16
credential-class rows — the "no secret values reproduced anywhere" claim holds); and
`a3237dd6b7` **IS** an ancestor of `origin/main`, so the public-history framing is correct.

### RV17-F3 — MED — the "16 hits" arithmetic never reconciles, and two of our memos disagree

Headline and §THE ANSWER FIRST say *"the 16 hits are third-party/expired"*; the body then
enumerates *"The 3 JWTs"* and *"The 4 AIza keys"* — which sums to 7, not 16, with no bridging
sentence. MEASURED from the receipt:

```
gcp-api-key  12 rows @ 3 distinct (file,line) locations × 4 columns each
jwt           4 rows @ 4 distinct locations × 1 each
```

So "16" is a ROW count and "3 + 4" is a VALUE count — two aggregation levels presented as one
quantity. Worse, `main_hot_state.md` (from sw1) records **"12 GCP keys + 4 JWTs"** while sw2 says
**"3 JWTs"**. One of our own memos contradicts the other, and the redacted receipt cannot settle it
(4 JWT rows sit at 4 distinct locations; 3 distinct *values* would require two of them to be the
same token). This is the `UNITS × LEVEL × AGGREGATION are part of the claim` genus. It matters
because **#1168 was downgraded from P0 on this adjudication** and the operator's residual action is
a 4-prefix GCP-console check — an under-count would make that check incomplete.
**CURE:** state row-vs-value explicitly ("16 rows = 4 distinct GCP keys × 3 log locations + 4 JWT
occurrences"), and reconcile the JWT count against the unredacted receipt.

### RV17-F4 — MED — "full 15,340-commit public history" is 97.8%, not full

The gitleaks leg's denominator is MEASURED-real — its own log says `15340 commits scanned` /
`2.14 GB` / `leaks found: 38885` — but the *same memo* uses **15,678** commits as the blast-radius
denominator (git rev-list; today `main` = 15,681, `--all` = 15,692). The sweep therefore covered
15,340 / 15,678 = **97.8%** of commits while the memo calls it "full". The 338-commit delta is
almost certainly diff-less merge commits (gitleaks scans patches), which would make coverage
materially complete — but that reconciliation is never stated, and content introduced only via a
merge would be missed. Denominator-honesty genus.
**CURE:** one sentence reconciling 15,340 against 15,678, or a blob-level pass to close the gap.

### RV17-F5 — LOW — receipt citation points at a file that does not exist, and at the wrong one

The memo cites `gitleaks_custom_history.json` "(+ `.log`)". There is no
`gitleaks_custom_history.log`; the log is `gitleaks_custom.log`. Separately, the memo's decisive
adjudication facts — the four `AIzaSy…` prefixes, the JWT `aud=gpu-t4-…` claim, the "~55 min apart"
expiries — are **not reproducible from the cited receipt**, which is redacted; they required the
unredacted `gitleaks_history.json`. **CURE:** cite the receipt that actually carries each claim.

---

## ITEM 6 — pq9 packet polish (commit `fd8e6024c7`)

**CLEAN on the question that mattered most: there is no false custody pairing.** Grepping every
public doc for score and archive literals returns **zero** occurrences of `df7fd266` or
`0.14827847…`. The packet consistently pairs jg5 (`f3bce5d2` / 180,625 B /
`0.14839100138338618`); ck1 literals appear only inside the `generation_4_superseded` block.
`PACKET_TARGET.json` is coherent: `swap_generation = 5`, `active_candidate = jg5_joint_waterfill_455`,
generations 2/3/4 explicitly superseded. `SWAP_PROCEDURE.md` step **4A** carries the hosted-URL
re-pin requirement in full, including *"A prior candidate's working URL is historical evidence
only; it never transfers across a swap."* The 24 `GATED-ON-RC2` markers are honest, deliberate
placeholders — correctly gated, and now fillable (RV17-F1).

One standing consequence, stated plainly rather than as a finding: while those 24 placeholders
remain, the packet is REFUSED by its own refusal condition (*"Any public artifact contains
unresolved placeholders"*). Publication is correctly blocked until the swap fills them.

### RV17-F6 — MED — six git-tracked receipts in the shipping custody tree are not parseable

`submissions/robust_current/jg5_sub015_runtime/t4_receipts/harvested_artifacts/` — all 8 files
(6 of them git-tracked) begin with `b'` and contain literal `\n` escapes: they are Python
`repr(bytes)` dumps, not decoded JSON/text. None parses.

```
contest_auth_eval.json  inflated_outputs_manifest.json  modal_cuda_auth_eval_validation.json
modal_cuda_preflight.json  provenance.json  report.txt  (+2 untracked .log)   → all `b'`
```

Scope is bounded and the defect is NOT in the general harvester: the normal path
(`experiments/results/*modal*/harvested_artifacts/`) is clean — **8 checked, 0 corrupt**. The bug
is in the one-off custody-copy step that populated the shipping tree. Content is fully recoverable
(`ast.literal_eval` → `canonical_score 0.14839100138338618`), so no evidence is lost. It matters
because these are the receipts a public reviewer would open to verify our score, and because live
consumers glob exactly this filename: `src/tac/deploy/modal/anchor_lookup.py:503` and
`training_claims.py:43-44`.

Sealed custody → **routed, not edited**. **CURE** (owner: pq9 / swap): re-extract the artifacts
with proper decoding from the valid `MODAL_REMOTE_RESULT.json` sitting beside them, as part of the
rc2 swap; and fix the copy step so the next generation cannot repeat it.

### RV17-F7 — LOW — the packet's own verify command does not work as written

`MANIFEST.sha256` header says `Verify:  sha256sum -c MANIFEST.sha256`. Run from the file's own
directory it fails **33/33**. It verifies **33/33 OK** only from
`submissions/robust_current/jg5_sub015_runtime/runtime/`, a working directory the packet never
names. A maintainer following the instruction verbatim concludes the pin is broken. Separately,
the header claims the pin *"is derived from exactly this enumerated row set"*, but the canonical
derivation (`src/tac/phase1_packet_compiler.py:404`) hashes `relpath + bytes + sha256 + mode` —
so `2103073d…` is **not recomputable from `MANIFEST.sha256` alone**.
**CURE:** state the required `cd`, and name the tool that reproduces the tree hash.

---

## COUNTER

**0 / 3.** Round 1 produced **30 findings** — 7 from my own re-derivation, 13 from the supervised
sw1 arm, 10 from the supervised secrets arm. Two are HIGH: **RV17-F1** (the pointer-move memo
discards its own critical-path evidence) and **SEC-F1** (the "BLOCKING" secrets gate passes every
commit on a machine without gitleaks). No code fix landed this round: every cure is either sealed
custody (RV17-F6), a memo correction
(F1, F2, F3, F4, F5, F7), or a cure whose *choice* is a design decision the module owner should
make (SW1-F1: crash vs. return-raw vs. typed error). Per the operating manual, a fix is unreviewed
new code; routing beats reflex-patching in a review round.

**Prior-law prediction: CONFIRMED.** I predicted this round would find ≥1 real defect and set a
genuine clean pass as the falsifier. It found 30, each with a reproduction command. Round 2 should
re-verify the cures and re-run Items 1–6 from the receipts, not from this memo.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_rederivation_receipt.json` — the
cross-axis token-decoder comparison, both stage-second splits, the recomputed S, and the
rule-count census, so round 2 can diff against machine values rather than this prose.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
