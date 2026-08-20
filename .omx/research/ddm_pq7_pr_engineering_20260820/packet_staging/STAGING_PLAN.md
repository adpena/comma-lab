# Staging plan for the compression script — for MAIN, at the freeze boundary

`date_utc: 2026-08-20` · `owner: ddm_pq7` · **Nothing here has been staged. I wrote files; I moved none.**

## The source file, and why there is no copy of it here

**Stage from `experiments/ddm_pq2_compress_e2e.py`**, sha256
`47d0e23dc3b17862c543f5af3823cab71a4859113e60a91123bee742480e28b9`, 30,742 B, 855 lines. Rename to
`compress.py` at staging to match the contest convention.

I made a copy in this directory, verified it byte-identical, and then **deleted it rather than
commit it**. A second copy of an 855-line script is a maintenance hazard and it is precisely the
duplication smell the maintainer rejects PRs for — it would have been odd to commit one inside the
report arguing against duplication. The sha above is the payload receipt; the bytes live in the
repo at a committed path.

Sanitization was **checked, not performed**, because the file was already clean (below). No
"sanitized variant" fork exists or should exist.

| Prepared file in this directory | Purpose |
|---|---|
| `COMPRESS.md` | ships beside the script; states what it rebuilds and what it cannot |
| `STAGING_PLAN.md` | this file |

## Sanitization: checked, clean

| Check | Result |
|---|---|
| `/Volumes` paths | none |
| `/Users/...` paths in source | none |
| Fleet IPs / tailscale / operator name | none |
| Bare `python` invocations | none — child processes use `sys.executable` |
| Local-path defaults for inputs | none — `--inputs-json` or environment variables only |
| Third-party dependencies | none; stdlib plus the in-repo `tac` package |
| `--help` accuracy | accurate; stage list matches `choices` |

**One residual leak, in output rather than source.** When the expected archive is resolved from the
canonical frontier pointer, the refusal message prints that pointer's absolute path, e.g.
`…resolved from frontier_pointer:contest_cuda:/Users/<user>/…/canonical_frontier_pointer.json`.
This path is only reachable **inside the research repo** (the pointer does not exist in a packet),
so it cannot leak from the published artifact. Ranked SHOULD in `OSS_STANDARDS_GAPS.md`; fix the
repo file, not a packet fork.

## ⛔ Where it may be staged — and the hash consequence, MEASURED

**My charter told me to verify from `gen5_receipts/provenance.json → inflate_runtime_manifest` that
a `scripts/` member sits outside the sealed manifest. I verified it, and the answer is the opposite
of the assumption.**

The manifest is **not** a closed allowlist. `experiments/contest_auth_eval.py:203-226`
(`_runtime_root_file_manifest`) walks `root.rglob("*")` and keeps every file whose suffix is in
`_RUNTIME_DEPENDENCY_SUFFIXES` (`contest_auth_eval.py:79-91`), which includes **`.py`, `.sh`,
`.txt`, `.json`, `.c`**. The only exclusions are `.git`, `__pycache__` and the cache dirs.

The 33 rows are 33 because the evaluated tree on the Modal host contained exactly 33 matching
files. `README.md` and `BORROWED_SUBSTRATE_ACCOUNTING.md` are absent because `.md` is not in the
suffix set; `archive.zip` because `.zip` is not.

**Therefore: adding `compress.py` (or `compress.sh`, or `COMPRESS.md`'s sibling `.txt`) anywhere
under the submission directory WILL be picked up by any future re-derivation of the manifest, and
the re-derived `runtime_tree_sha256` will not equal `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`.**

What that does and does not mean:

- **It does not touch the score.** `upstream/evaluate.py` sizes `archive.zip` only. The archive
  bytes, `d_seg` and `d_pose` are untouched. The 0.14839100138338618 row stands.
- **It does break naive re-validation.** Anyone re-running our own `contest_auth_eval` against the
  augmented directory gets a tree-hash mismatch and, per the Catalog #229/#146 contract, a refused
  dispatch. The identity proof must then be evaluated against the **pinned 33-row subset**, not a
  fresh walk of the directory.

## The recommendation

**Stage it, in the submission directory, as `compress.py` + `COMPRESS.md`.** Reasons:

1. It is the contest convention. `upstream/README.md:62` tells submitters to copy
   `submissions/baseline_fast/{compress.sh,inflate.{sh,py}}` into their own directory, and **every
   merged neural submission ships a `compress.sh`** — `hnerv_muon`, `hnerv_lc_v2_scale095_rplus1`,
   `hnerv_fec6_fixed_huffman_k16`, `rhnerv_comma` — as does PR #130.
2. The operator directed that the compression script ship.
3. A sibling `scripts/` subdirectory would carry the same hash consequence (the walk is recursive)
   while breaking convention. There is no placement that avoids the hash effect.

**Two things MAIN must do at the same time, or not stage it:**

- **(A) Record the subset rule.** Add to `STAGING_RECEIPT.json` (or the freeze receipt) an explicit
  statement that `runtime_tree_sha256 = 2103073d…` is pinned over the **enumerated 33 rows**, and
  that re-validation must hash those rows rather than re-walking the directory. Without this, the
  next arm to re-validate will read a mismatch as corruption.
- **(B) Do not re-run `packet_census_guard` expecting the old count.** The census will see 40 files
  rather than 38. Purge → census → receipt ordering still applies.

**If MAIN prefers zero risk to the identity proof:** link the script from the PR body and the
README instead of staging it, and state plainly that it lives in the research repository. This
costs the merge-eligibility criterion the operator asked for, so I do not recommend it — but it is
the honest alternative and the choice is MAIN's, not an arm's.

## What the script cannot do, which the PR body must not obscure

`compress.py` **cannot rebuild the shipped `f3bce5d2…` archive** — verified by execution: it
refuses by name at rc=1 before creating its store directory. `COMPRESS.md` states this in its first
sentence, and `FREEZE_CHECKLIST` (f)(2) already carries it as an owed item. Shipping a compression
script that refuses the shipped candidate is a weaker claim than the merged submissions make, and
the PR body must present it as what it is: a provenance emitter plus an exact rebuild of the token
stream, not a full compressor.
