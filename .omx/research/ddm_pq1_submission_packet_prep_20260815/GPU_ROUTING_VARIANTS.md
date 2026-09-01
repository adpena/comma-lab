# GPU-routing variants for packet generation 7 — AFR1

> **LIVE AFR1 OVERRIDE.** Archive `cbb8d928…`, 180,002 B, 38-row runtime tree
> `6cdfa27d…`. Its T4 authority run measured inflate 578.9354022370001 s and
> evaluate 42.69640948199992 s; charged total 621.631811719 s passes the
> projected 822 s cold-cache ceiling by 200.36818828100002 s. The packet remains
> GPU-routed. No receiver flip is authorized because changed runtime bytes need
> their own authority row. Routing/hosting is folded into the operator's single
> publication-confirmation gate.

## Preserved generation-6 analysis

# GPU-routing variants for packet generation 6 — the decision, its cost, and why the two options are NOT symmetric

**Author:** ddm_pq3, refreshed for generation 6 by ddm_pq11 · **Date:** 2026-08-20 · **Status:** PREPARED, DECISION RESERVED TO THE OPERATOR

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

From the authority receipt, on the T4, for **these exact bytes and this exact runtime**:

| Quantity | Measured |
|---|---|
| Inflation | 458.752594349 s |
| Evaluation | 39.72359129999995 s |
| Charged (inflation + evaluation) | 498.476185649 s |
| Job wall | 1,800 s |

The residual left for checkout, `uv sync`, apt and upload is a **projection**: `[822,
1302] s`, cold to warm `uv` cache. Only the job wall and the dependency payload sizes are
measured; no per-step CI second has been timed by us on a real runner. The charged time
fits the **cold** end — the binding corner — with **323.5 s** of margin, and the verdict
is **PASS**.

**This is the one place where generation 6 is not an increment on generation 5 but a
different verdict.** Generation 5 charged 1,471.3 s on the same axis and the same wrapper,
over BOTH ends of that window, and was graded WARN on about 10.7 s of warm-cache-only
margin. Replacing the Python free corrector with the C port cut inflation 3.10×, from
1,419.904212624 s to 458.752594349 s, at zero change to any decoded value. The old WARN
and its 10.7 s belong to the superseded object and are not carried forward.

On the CPU path the projection is superseded by measurement on **these** bytes: inflation
was killed at the **1,800 s** contest wall before `upstream/evaluate.py` started. The
receiver's own instrumentation finished afterwards at 2,850.781244341 s, token decode
alone 2,427.166373672 s (6.10× its CUDA cost), render 410.182710582 s (9.78×). The decoded
token stream is bit-identical on both axes at the same decoder bit position, so this is a
wall result and not a decode failure. No CPU score exists.

Prior lineages' CPU walls (3,422.7 s on generation-3 bytes, 4,369.6 s on the ck1/jg5
lineage) are **not** carried onto this object; it has its own measurement.

**Conclusion: CPU routing is MEASURED to time out on these exact bytes. T4 routing is tight but measured.**

## 4. THE MEASUREMENT THAT BREAKS THE SYMMETRY

`inflate.sh` is a row of the **36-row** runtime manifest, pinned at sha
`971eaa12b78e716825741ea86c28f9362eb9be077cc8cb3b873810ca979beb65`. The manifest-derived
`runtime_tree_sha256` is a canonical-JSON hash over every row. Therefore **any** edit to
`inflate.sh`, however small, changes the tree hash.

Measured directly rather than argued:

| Tree | `runtime_tree_sha256` |
|---|---|
| As shipped (authority row valid) | `fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2` |
| With a one-line CUDA fail-fast added | NOT MEASURED on this tree |

The score `0.14827847122030852` is bound to `fdd57749…`. Under any edited tree **there is
no row at all.** This is the round-11 F2(a) lesson in its original form: editing
manifest-pinned bytes ships what the exact evaluation never evaluated.

**A measured hash from the previous generation is NOT reused here.** On generation 5's
33-row tree the same one-line edit was measured to move the hash from `2103073d…` to
`75a1aeef…`. That pair describes a different tree and is recorded as history only. The
*mechanism* transfers — any edit to a manifest-pinned file moves the tree hash, which is
arithmetic over the row set, not an empirical finding. The *hash* does not. Carrying a
constant across regimes is the exact error this packet has already paid for twice, so the
flipped hash for this tree is left unmeasured until someone measures it.

And this is not a hypothetical worry for this lineage — it is the failure that already
happened once. The previous candidate scored **79.40** under one receiver tree and
**0.157** under another on byte-identical archive bytes, because arithmetic decoding under
a mismatched model returns rc=0 and emits garbage rather than raising.

## 5. The two variants, priced honestly

### Variant (b) — CURRENT / AUTO  ← this is what is staged

- `inflate.sh` unmodified; tree `fdd57749…`; **authority row valid**; compliance receipt
  owed for these bytes (generation 5's is stale on bytes, instrument and world alike).
- Routing is obtained by **telling the maintainer**, not by failing fast. The PR body
  §"does your submission require gpu for evaluation (inflation)?" answers **yes** and
  states the CPU-path infeasibility and the T4-path PASS-with-margin explicitly, with
  the projected half of the budget labelled as a projection.
- **Risk:** if the workflow is dispatched with `runner: ubuntu-latest`, the run **is
  measured to exceed the budget** — inflation on Linux x86_64 CPU was killed at the
  1800 s wall on these exact bytes. The PR body is the only thing preventing that.
- **Cost to ship: zero.**

### Variant (a) — GPU-REQUIRED-EXPLICIT  ← prepared, NOT applied

- Prepend to `inflate.sh` after `set -euo pipefail`:

  ```bash
  python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "requires CUDA" >&2; exit 69; }
  ```

  (2,203 B → 2,323 B; exit 69 matches the existing fail-closed convention in the script.)
- The script then genuinely requires a GPU, so the README's routing rule selects the T4
  by the script's own behaviour rather than by a maintainer reading our prose.
- **Cost to ship: one new T4 exact-eval row**, because the tree hash moves — to a value
  nobody has measured on this 36-row tree — and every score claim, the seal, the
  archive-manifest pin, the compliance receipt and the PR body's identity table must be
  re-derived against it. Roughly one paid dispatch plus a full re-stage and re-review.
- **Second-order risk:** the added check runs `import torch` before inflation, adding a
  small fixed cost to the decode budget. That budget now has 323.5 s of margin at the
  binding corner rather than 10.7 s, so the check is very unlikely to threaten it — but
  it should still be measured on the new row rather than assumed.

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
