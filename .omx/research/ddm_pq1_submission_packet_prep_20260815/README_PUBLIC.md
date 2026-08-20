# jg5_joint_waterfill — submission packet

This directory is a prepared submission packet held at
`submissions/jg5_joint_waterfill/`. It is **not submitted**. It is staged so
that every claim below can be checked against a receipt before anyone acts on it.

The one number this packet exists to carry: on the exact bytes in this directory,
the measured `[contest-CUDA]` 600-sample score is **0.14839100138338618**.

## Evidence boundary

Read this before reading anything else, because it is where most submissions
overstate themselves.

- **What is measured.** One exact evaluation: `upstream/evaluate.py` at
  `--device cuda` on a Tesla T4, Linux x86_64, over all 600 public samples,
  driven through this directory's own `inflate.sh` on this directory's own
  `archive.zip`. Score, both distortion legs and the rate leg all come from that
  run. The score published here is **recomputed from the reported components**,
  not read off the evaluator's rounded `final_score` field — that field prints
  `0.15`, a 2-decimal display that rounds up across the very boundary this
  submission sits on.
- **The precision of that claim.** The evaluator reports its distortion
  components at 8 decimal places, so the recomputed score carries a worst-case
  absolute error bound of `3.63296497868841e-06`. The claim is
  `0.14839100138338618 ± 3.633e-06`. The distance from that interval to 0.15 is
  about 443 times the bound, so the sub-0.15 statement is not a rounding artifact.
- **What is NOT measured.** There is **no `[contest-CPU]` row on these bytes**,
  and none is claimed. This submission is GPU-required for evaluation.
- **The open runtime risk.** Inflation took 1419.9 s of a 30-minute job wall.
  That is disclosed in full in `report.txt` under "Evaluation-time budget", and
  it is the largest open risk on this submission. It is a runtime risk, not a
  score risk. The cost is concentrated rather than diffuse: **token decode alone
  is 1341.5 s, 95.72% of inflation.** We hold a native port of that stage's
  integer half which reproduces this candidate's decode bit-for-bit on the full
  600-frame field at 1.77–1.83× on local hardware; it is **not** in the tree
  evaluated here, and folding it would move the runtime-tree hash and require a
  new exact evaluation. Disclosed as available work, not claimed as a fix.
- **What is not authority.** Any local macOS number, any advisory row, and any
  projection appearing in our own research notes is not a score and is not used
  here.

## Exact identity

| Property | Value |
|---|---|
| Archive SHA-256 | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` |
| Archive size | 180,625 bytes |
| Member | `p`, 180,525 bytes, stored, SHA-256 `54b445da3a1a4b4c7012c83b25c3e0d87daab5ce10cd54a1598cfb239ab05b4a` |
| Members in archive | 1 |
| Runtime tree SHA-256 | `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b` |
| Portable runtime content tree SHA-256 | `3ba9987771e1be967cf80942faedc7c5f6641f15039e03dd2b0909fd6613ab99` |
| Upstream snapshot SHA-256 | `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008` |
| Upstream `evaluate.py` SHA-256 | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` |
| Candidate seal SHA-256 | `96e9860aad9021e6dc9a9619036b54bd0a2205f60468e8585089db1d8044a7d0` (`SEAL_VALID`) |

The runtime tree hash is load-bearing rather than ceremonial, and the reason is
empirical. See "Runtime tree pin" in `report.txt`: the previous generation's
archive scored 79.40 under one receiver tree and 0.157 under another, on
byte-identical archive bytes. For this receiver, the archive hash alone does not
determine the score.

## What this submission is

The same inherited vehicle as the prior candidate. What changed is how the seg
edits and the pose carrier are decided — together instead of one after the other.

1. **The problem this candidate solves.** The prior composition applied all 573
   seg token edits and then re-solved the pose carrier. Measured, that direction
   is seg-descending but not pose-null: the edits bought seg and cost about 13
   times more pose than they bought, and the composed result scored far worse
   than either part suggested. Solving one axis and then repairing the other is
   the failure; the two are one problem.
2. **Joint admission by waterfill (this candidate's mechanism).** Edit admission
   is swept over a Lagrange multiplier on pose damage rather than fixed in
   advance, so each edit is kept only if it pays for the pose it costs.
   **455 of the 573 edits are admitted**; the other 118 are dropped and those
   pairs keep the prior carrier's codes.
3. **Carrier re-solve against this candidate's own renders.** The frame-0 pose
   carrier is re-solved against the edited renders this archive actually decodes
   to — not against the base renders — so the compensation is fitted to the
   state that ships. The descent uses a derived materiality stop rule rather than
   a fixed iteration budget: **600 of 600 pairs stopped on `no_improving_step`,
   with zero budget hits**, so the stopping criterion was never the binding
   constraint.

The net effect is the reverse of every prior generation in this packet: it
**spends** 3,443 bytes and buys both distortion legs, for a net
`-0.008710980`. A reader scanning byte counts alone would misread the larger
archive as a regression, so the leg split is published in `report.txt`.

The tail sections, the HPAC stream and the 13-context fixed-point integer
log-odds mixer that produces the token stream are carried over from the prior
candidate.

**Read `BORROWED_SUBSTRATE_ACCOUNTING.md`, shipped in this directory, before
treating any of the learned content as ours.** Most of it is not: the semantic
renderer state and the pose carrier state originate in PR #130 and PR #135, and
the compressed model container, the HPAC probability object's architecture, the
residual payload and the range-coder backend all come from that lineage. This
candidate ships a lossy re-representation of their trained state, which raises
the attribution question rather than settling it, and the accounting says so in
its own words. The same table is reproduced inline in the pull-request body.

## Two names a reader will meet in the receiver

Both are cosmetic and neither changes behaviour.

- `CP135` and `F26` appear in `inflate.sh` as an error string, environment
  variables and file names. They are internal codenames for the inherited
  PR130/PR135 lineage, kept because renaming them would change the evaluated
  runtime-tree hash.
- `inflate.sh` carries a `Darwin` branch that calls `brew --prefix libomp`. It is
  **unreachable on the contest runner**: it requires `F26_TOKEN_DECODER` to equal
  `native-hpac`, and the script defaults that variable to `python` with nothing
  setting it otherwise. This submission assumes Linux.

## Dependency closure

`inflate.sh` is **not fully self-contained**, and that is stated here rather than
left for a reader to discover:

- It requires **`Brotli==1.2.0`** exactly. If the interpreter does not already
  have that version, the script calls `uv pip install --only-binary :all:
  "Brotli==1.2.0"`, which **reaches the network at decode time**. If `uv` is
  absent the script exits 69 rather than proceeding — it fails closed, but it
  does fail.
- It invokes a **C compiler** (`${CC:-cc}`) at decode time to build
  `runtime/entropy/rc64_backend.c`. The compiler is assumed present on the runner.
- Otherwise: PyTorch and NumPy, both already required by the evaluator itself.

The declared-dependency approach follows the precedent set by earlier accepted
submissions in this contest, which likewise declared a pinned Brotli. It is
flagged here because "no network at decode time" is a reasonable thing for a
judge to expect, and this submission does not meet it.

## Reproduction

The end-to-end rebuild entry point has **not been re-run for these bytes, and it
cannot rebuild them.** Those are two different statements and only making the first
would invite the wrong conclusion. That entry point rebuilds the token stream and
carries the other seven sections through verbatim; this candidate's chain also
re-decides content in sections it copies — the seg token edit solve, the edit
splice, the admission waterfill and the pose-carrier re-solve. No configuration
closes that gap, so the script refuses this archive by name and names the builders
that do produce it.

What does exist for this candidate is the seal binding archive to receiver, the
staging proof that this directory is byte-identical to the evaluated tree, and the
authority receipt.

One further reproducibility limit, disclosed rather than left to be discovered: a
census of the 34 files in this candidate's tree found **24 with no source in
version control**, including the receiver modules and `inflate.sh` itself. The
shipped bytes are pinned by hash and the decode is deterministic, so what a judge
runs is fully determined — but a reader who expects to rebuild the receiver from a
public repository cannot currently do so.
