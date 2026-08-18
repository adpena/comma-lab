# Submission PR adversarial review scaffold

Status: `HOLD`, consecutive clean passes: `0/5`.

Any finding, however small, resets the counter to `0/5` after the finding is
fixed. A pass cannot be counted while the strict compliance chain is red.

| Round | State | Reviewer | Candidate archive SHA-256 | Compliance receipt SHA-256 | Findings | Counter after round |
|---:|---|---|---|---|---|---:|
| 1 | RUN 2026-08-17 | ddm_pq2 | `35ac2b9beb7e6fa8…9618956` | `ba596648e1e1f6a5…c43656` | 6 (3 closed, 3 open) | 0/5 |
| 2 | NOT_RUN | unassigned | pending | pending | not reviewed | 0/5 |
| 3 | NOT_RUN | unassigned | pending | pending | not reviewed | 0/5 |
| 4 | NOT_RUN | unassigned | pending | pending | not reviewed | 0/5 |
| 5 | NOT_RUN | unassigned | pending | pending | not reviewed | 0/5 |

## Round 1 — findings

**The counter stays at `0/5`, and would stay there even with zero findings:** the
strict chain is red at 81/85, and the scaffold forbids counting a clean pass
while it is red. Round 1 is therefore a real review, not a countable pass.

Round 1 was run by the same arm that re-targeted the packet. **That is a
conflict** and is recorded as such: rounds 2 through 5 must be run by reviewers
who did not build generation 2.

| # | Axis | Finding | State |
|---|---|---|---|
| F1 | dependency behavior | The charter's premise that this runtime carries a `constriction` declared-dep is **false**. An AST scan of all 32 runtime files returns `brotli`, `numpy`, `torch`; `constriction` appears nowhere. The real self-installed dep is `Brotli==1.2.0`, and it was smoked instead. | **CLOSED** — premise corrected, correct dep verified |
| F2 | public-text / packet hygiene | The ExFAT store creates AppleDouble `._*` files on every copy. They are non-UTF-8, they **crashed the compliance checker mid-run** leaving a stale receipt on disk, and they would have shipped inside a submission directory. | **CLOSED** — `COPYFILE_DISABLE=1`, explicit strip, non-UTF-8 scan before every run |
| F3 | stale receipts across the byte boundary | The two inherited receipts are **inside** the hashed 32-file runtime tree, so the prescribed cure of writing a supersession file beside them would have changed the pinned tree hash `7acedb07…`. | **CLOSED** — supersession moved to the store root; correction placed in `README.md`, which the evaluated manifest excludes; pinned hash verified to reproduce |
| F4 | exact score axes | `tools/fire_modal_auth_eval.py`, the one canonical deterministic firing path, is **CUDA-only**. A contest-CPU row would have to be hand-assembled, which is the hazard the standing law names an error factory. | **OPEN** — cure the tool before firing any CPU row |
| F5 | both exact score axes on the same bytes | Only one axis exists. `[contest-CPU]` is unmeasured on these bytes and the packet says so everywhere it matters. | **OPEN** — operator/MAIN economic call; fire order prepared, $1.38 headroom |
| F6 | hosted archive URL and manifest | No public URL exists; hosting is unauthorized. The packet claims none. | **OPEN** — blocked on operator authority |

## Axes covered in round 1

- **archive/member SHA, size, grammar, deterministic repeat** — verified
  independently: `35ac2b9b…`, 181,161 B, single stored member `p`, 181,061 B,
  CRC32 885609521; rebuild reproduced the sha and the determinism repeat was
  byte-identical.
- **runtime closure, dependency behavior, budget, hidden payloads** — 32-file
  tree reproduces the pinned hash; bootstrap smoked in a bare venv and exits 69
  fail-closed without `uv`; only a pinned wheel crosses the network; inflation
  used 476.6 s of 1,800 s (3.78x headroom).
- **score recomputation, labels, snapshot, runtime binding** — S re-derived from
  components to all 17 digits; upstream snapshot, `evaluate.py` sha, and runtime
  tree sha all bound in the terminal lane row.
- **borrowed-substrate accounting** — re-run at section level against the shipped
  sections; two lineage items added that the prior generation did not disclose
  (the encoder compiles PR135's `rc64_backend.c` unmodified; the shipped receiver
  coder is a **modified** descendant, sha `05839d14…` ≠ `5c75e2c7…`).
- **public source pin, hosted URL, report linkage** — pin present, visibility
  unverified; no hosted URL claimed.
- **public-text hygiene** — scanned `README_PUBLIC.md`, `REPORT_PUBLIC.txt`,
  `PR_BODY_DRAFT.md` and the staged `README.md` / `report.txt` for local paths,
  infrastructure addresses, credentials, provider records, and machine
  attribution: **clean**.
- **swap delta vs prior generation** — recorded in `GENERATION_LOG.md`; no stale
  receipt crosses the byte boundary, and the two that live in the tree are named
  explicitly in the public README.
- **PR template conformity and scope of pending numbers** — every pending or
  advisory number is labelled; the competitive claim explicitly declines to
  assert a win over PR138's unverified figure.

The fifth clean pass authorizes only a recommendation to MAIN. It does not
authorize submission, push, or hosting.

## Generation-3 swap note (2026-08-18)

The candidate hot-swapped to the sz1 composed archive
(`debb025f45bb42e3…`, 179,930 bytes, measured `[contest-CUDA]`
0.15771357797660338). Round 1's findings were reviewed against the
generation-2 (rr4) bytes and packet text; its three open findings carry
forward where they are generation-independent (hosting authorization,
Brotli-bootstrap policy, hygiene ledger rows). The counter remains `0/5`.
All five counted passes must run against the generation-3 bytes, the
refreshed packet documents (commits f4a3882345 / e54bdfa37e / 9fe84725f5),
and a green strict compliance chain — which additionally requires the
reproduction re-bind (`PENDING_REBIND` in `PACKET_TARGET.json`) to land
first. The CPU axis is closed for review purposes: measured infeasible
within the 1,800 s budget, receipt in `PACKET_TARGET.json` `cpu_axis`.
