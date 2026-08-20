# ddm_oc2 — comma-lab public repo: PUSH_READY receipt

`date_utc: 2026-08-20` · `owner: ddm_oc2` · authority: operator consolidation directive 2026-08-20
(our own repos only; the **contest PR is untouched** and still gates on the operator's one-line
confirm, #1111).

## 1. The premise in the charter was wrong, and it matters

The charter said to "locate the local comma-lab public repo clone/remote … check `~/Projects` for a
clone." There is no separate clone, because:

```
$ git remote -v
origin  git@github.com:adpena/comma-lab.git (fetch)
origin  git@github.com:adpena/comma-lab.git (push)

$ gh repo view adpena/comma-lab --json visibility,isPrivate
{"isPrivate": false, "visibility": "PUBLIC"}
```

**The pact research repo IS `github.com/adpena/comma-lab`, and it is PUBLIC.** `README.md` line 1 is
`# comma-lab`. There is no branch to prepare and no second repo to sync — the consolidation push
below *is* the public update. Every future commit to `main` here publishes on push.

This is worth stating plainly because it inverts the risk model an arm would otherwise assume: work
committed to this repo is not "internal until promoted." It is public on the next push.

## 2. What was pushed

`8aaa9ffcf4..31b90fffc9`, 19 commits, `main -> main`, no pre-push refusal, 8.5 s. The pre-push hook
(`tools/preflight_hook.py`) ran and passed; **no skip was used and none was needed** — the CI-blind
co-load class (#1154) did not fire.

## 3. Hygiene verdict — with a passing positive control

⚠ **My first sweep was vacuous and I nearly shipped a false all-clear.** It used
`xargs -a <file> grep …`; BSD `xargs` on macOS has no `-a`, so it errored and matched nothing,
returning a clean `0` on every check. A positive control (a file known to contain 1,073 `/Volumes/`
hits, verified present in the scanned list) came back "no match" — which is what caught it.

Re-run with `xargs -0 … < list` and a **passing** positive control, over the **891 distinct files**
this consolidation touched:

| Check | Count | Verdict |
|---|---:|---|
| Credentials, API tokens, private keys | **0** | PASS |
| Tailscale / fleet IPs / hostnames | **0** | PASS — the lone regex hit is `BRANCH_DISPOSITION.md` quoting the CIDR `100.64.0.0/10` as a check label, a self-match |
| `/Volumes/…` absolute paths | **365** | **FAIL against strict hygiene** — published |
| `/Users/adpena/…` absolute paths | **399** | **FAIL against strict hygiene** — published |
| `python` (bare) invocations in staged scripts | not re-verified by this arm | pq7 verified the packet script is `sys.executable`-only |

**Honest read:** nothing secret leaked. What did go public is the same class of local-path citation
already present in 4,226 (`/Users`) and 5,242 (`/Volumes`) tracked files — a standing condition this
push widened by roughly 9% on the `/Users` axis. CLAUDE.md's Public Disclosure Hygiene rule does say
local absolute paths stay off public surfaces. **This is an operator decision, not an arm decision:**
accept it as the cost of the research corpus being public, scrub the new files, or scrub the corpus.
I did not scrub unilaterally because scrubbing 764 research memos would mutate dated receipts, which
the HISTORICAL_PROVENANCE discipline forbids without a decision.

## 4. The packet: what is ready and what is MAIN's call

`ddm_pq7` already staged and committed the packet docs
(`.omx/research/ddm_pq7_pr_engineering_20260820/packet_staging/{COMPRESS.md,STAGING_PLAN.md}`).
`LICENSE` (1.1 K) and `THIRD_PARTY_NOTICES.md` (7.3 K) are already on `main`. The jg5 runtime custody
tree was versioned by the wc2 custody P0 in commit `2d61b51988` (44 files / 10,547 insertions).

**One decision is explicitly NOT an arm's to make**, and pq7 measured why. Staging `compress.py`
under the submission directory **will change `runtime_tree_sha256`**: `_runtime_root_file_manifest`
(`experiments/contest_auth_eval.py:203-226`) walks `root.rglob("*")` and keeps every `.py/.sh/.txt/.json/.c`,
so the manifest is *not* a closed allowlist. A re-derived tree hash would no longer equal
`2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`.

- It does **not** touch the score — `upstream/evaluate.py` sizes `archive.zip` only; the
  0.14839100138338618 row stands.
- It **does** break naive re-validation, which then refuses dispatch per Catalog #229/#146.

pq7's conditions stand: if MAIN stages it, MAIN must simultaneously (A) record in the freeze receipt
that the tree hash is pinned over the **enumerated 33 rows** rather than a fresh directory walk, and
(B) expect the census to report 40 files, not 38.

## 5. State

**PUSH_READY = YES for the research consolidation, which is already pushed.** Nothing further is
staged awaiting a public push. The contest PR remains untouched and operator-gated.

Owed to MAIN, in priority order:

1. **Decide the absolute-path question** (§3) — the only real hygiene finding.
2. **Decide `compress.py` staging** (§4) — with pq7's conditions (A) and (B).
3. Fix the six review findings in `REVIEW_FINDINGS_OWED.md`, then apply
   `sh1_held_deltas_BLOCKED_ON_FIXES.patch`.
4. Prune `dy1` and `einstein_kolmogorov` as superseded refs.
