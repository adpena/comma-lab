# GPU-routing variants for packet generation 5 — the decision, its cost, and why the two options are NOT symmetric

**Author:** ddm_pq3 · **Date:** 2026-08-20 · **Status:** PREPARED, DECISION RESERVED TO THE OPERATOR

This document exists because the charter asked for two packet variants "as a one-line
flip". **They are not a one-line flip, and delivering them as if they were would have been
the more convenient answer rather than the true one.** The measurement is below.

---

## 1. The routing rule, read at source

| Fact | Source | Exact text / value |
|---|---|---|
| Time limit | `upstream/README.md:114` | "The official evaluation has a time limit of 30 minutes." |
| Hardware fork | `upstream/README.md:114` | "If your **inflation script** requires a GPU, it will run on a T4 GPU instance (RAM: 26GB, VRAM: 16GB), if it doesn't it will run on a CPU instance (CPU: 4, RAM: 16GB)." |
| The limit is on the WHOLE job | `upstream/.github/workflows/eval.yml:30` | `timeout-minutes: 30` on the job, not on a step |
| The runner is a dispatch INPUT | `upstream/.github/workflows/eval.yml:17-24` | `runner` is a `workflow_dispatch` choice, default `ubuntu-latest`, options `ubuntu-latest` / `linux-nvidia-t4` |

The last row matters and is easy to miss: **a maintainer selects the runner when they
dispatch the workflow.** The README states the policy; the workflow does not auto-detect.

## 2. What our `inflate.sh` actually does

It runs `python inflate.py` per video, compiles `rc64_backend.c` with `cc`, and
bootstraps `Brotli==1.2.0` if absent. **It contains no CUDA requirement and no CUDA
check.** `inflate.py` performs a neural render that runs far faster on a GPU, but nothing
in the script *requires* one or fails without one.

So on a literal reading of the routing rule, this submission would be routed to the
**4-core CPU instance**, which is not where its measured row was taken.

## 3. The measured wall-clock position

From the authority receipt, on the T4:

| Quantity | Measured |
|---|---|
| Inflation | 1,419.9042126240001 s |
| Evaluation | 51.427507448999904 s |
| Inflation + evaluation | 1,471.3 s |
| Job wall | 1,800 s |
| Residual for checkout + deps + download | **328.7 s** |

Our internally derived residual window for those remaining steps on the CUDA path is
`[890.6, 1430.6] s`. This candidate fits only at the most optimistic end, by about
**10.7 s**. That is graded **WARN, not PASS**: a 10.7 s margin resting on a warm-cache
assumption is not a margin.

On the CPU path the projection is superseded by measurement: `ddm_cpu1` (2026-08-20)
MEASURED contest-CPU inflation on THESE bytes at **4,369.6 s** — over the `[1,044, 1,332] s`
residual by 3,037.6 s and **2.43x the entire 1,800 s job wall** on its own. The prior
lineage's 3,422.711146813 s (gen-3 bytes) understated this candidate's cost by 946.9 s
(+27.7%). The decoded token stream is bit-identical on both axes.

**Conclusion: CPU routing is MEASURED to time out on these exact bytes. T4 routing is tight but measured.**

## 4. THE MEASUREMENT THAT BREAKS THE SYMMETRY

`inflate.sh` is **row 8 of the 33-row runtime manifest**, pinned at sha
`e1b3df4d9178a1572cf04bc2dd9d2ddcc8f6157deac8ec1c40e89b2114522d62`. The manifest-derived
`runtime_tree_sha256` is a canonical-JSON hash over every row. Therefore **any** edit to
`inflate.sh`, however small, changes the tree hash.

Measured directly rather than argued:

| Tree | `runtime_tree_sha256` |
|---|---|
| As shipped (authority row valid) | `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b` |
| With a one-line CUDA fail-fast added | `75a1aeef5effc44edce5b5e3cbc783f9e82ca196233ee183f9abeb6cecce84f2` |

The score `0.14839100138338618` is bound to `2103073d…`. Under `75a1aeef…` **there is no
row at all.** This is the round-11 F2(a) lesson in its original form: editing
manifest-pinned bytes ships what the exact evaluation never evaluated.

And this is not a hypothetical worry for this lineage — it is the failure that already
happened once. The previous candidate scored **79.40** under one receiver tree and
**0.157** under another on byte-identical archive bytes, because arithmetic decoding under
a mismatched model returns rc=0 and emits garbage rather than raising.

## 5. The two variants, priced honestly

### Variant (b) — CURRENT / AUTO  ← this is what is staged

- `inflate.sh` unmodified; tree `2103073d…`; **authority row valid**; compliance 83/87.
- Routing is obtained by **telling the maintainer**, not by failing fast. The PR body
  §"does your submission require gpu for evaluation (inflation)?" answers **yes** and
  states the CPU-path infeasibility and the T4-path WARN explicitly.
- **Risk:** if the workflow is dispatched with `runner: ubuntu-latest`, the run is
  expected to exceed the budget. The PR body is the only thing preventing that.
- **Cost to ship: zero.**

### Variant (a) — GPU-REQUIRED-EXPLICIT  ← prepared, NOT applied

- Prepend to `inflate.sh` after `set -euo pipefail`:

  ```bash
  python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "requires CUDA" >&2; exit 69; }
  ```

  (2,203 B → 2,323 B; exit 69 matches the existing fail-closed convention in the script.)
- The script then genuinely requires a GPU, so the README's routing rule selects the T4
  by the script's own behaviour rather than by a maintainer reading our prose.
- **Cost to ship: one new T4 exact-eval row**, because the tree hash moves to
  `75a1aeef…` and every score claim, the seal, the archive-manifest pin, the compliance
  receipt and the PR body's identity table must be re-derived against it. Roughly one
  paid dispatch of about 25 minutes plus a full re-stage and re-review.
- **Second-order risk:** the added check runs `import torch` before inflation, adding a
  small fixed cost to a budget whose measured margin is 10.7 s. It should be measured,
  not assumed, on the new row.

## 6. An honesty note the operator should weigh

Variant (a) makes the script require a GPU that its computation does not actually need,
in order to obtain a faster machine. That is a routing device, not a lie — after the edit
the script really does refuse to run without CUDA — but it is worth naming plainly rather
than shipping quietly. Variant (b) obtains the same routing by stating the requirement in
the PR body, which is the surface the maintainer reads when choosing the runner.

**Recommendation is deliberately withheld. This is the operator's call at freeze.** What
this document supplies is the price: variant (b) costs nothing and relies on the
maintainer selecting the T4 runner; variant (a) costs a new T4 row and a re-stage, and
removes that reliance.
