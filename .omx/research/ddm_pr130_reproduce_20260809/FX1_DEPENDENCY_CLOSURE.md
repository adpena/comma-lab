# FX1 — PR130 `constriction` dependency closure

Axis: `[byte-only scorer-free, macOS-CPU dependency smoke]`. This work did not use a scorer,
Metal/MPS, or CUDA, and it did not measure or move a contest score. Numeric receipts are in
[`FX1_DEPENDENCY_CLOSURE_RECEIPT.json`](./FX1_DEPENDENCY_CLOSURE_RECEIPT.json) unless another
source is named explicitly.

## 1. Cured bare-venv receipt

**CURED for the measured dependency-smoke scope.** Before this arm, I did not find an owned,
derived PR130 shipping runtime in five enumerated repo/custody scopes (596 files total; 0 matching
trees). That is a scoped absence, not a global nonexistence claim. I therefore built the minimal
derived tree at [`src/tac/pr130_runtime/fx1_runtime_tree`](../../../src/tac/pr130_runtime/fx1_runtime_tree).
Its five receiver modules are byte-for-byte
copies of intake commit `e34f31bc4969042c0051ac81aa3c56884419a231`; 0/5 receiver modules changed.
The intake remained clean and was not edited.

The derived entrypoint now:

- declares the five unchanged receiver modules as verbatim borrowed intake substrate and claims
  only the FX1 wrapper/manifest as original work;
- declares and pins `constriction==0.5.0` in `runtime-dependencies.json`;
- tries the exact version and required `RangeDecoder`/`Categorical` APIs first;
- installs only a binary wheel, with dependencies disabled, into an isolated target if needed;
- verifies the installed version and APIs before publishing that target atomically;
- refuses an invalid existing target instead of overwriting it; and
- exits nonzero on every closure failure. There is no numeric fallback: a different arithmetic
  decoder or probability contract could desynchronize the entire stream.

This mirrors the #666 e4 structural precedent—declared dependency plus a loud, closed failure
surface—without copying e4's codec fallback. The r5/IC2 lesson changed the deliverable from a
manifest-only declaration into an executed clean-host bootstrap.

The final smoke began with the current upstream CPython 3.11.15 venv reporting
`site.ENABLE_USER_SITE=False`, `find_spec("constriction") is None`, and a fresh absent install
target. The final `inflate.sh` installed `constriction==0.5.0`, imported the real copied
`inflate.py`, printed `PR130_DEPENDENCY_READY constriction=0.5.0 receiver=inflate.py`, and
terminated rc=0. `uv` reported 111 ms to resolve and 223 ms to install; the complete entrypoint
smoke took 17.50 s. A warm repeat terminated rc=0 in 0.51 s. The host venv still reported
`find_spec("constriction") is None` afterward, proving that the bootstrap did not mutate it.

The positive receipt used a locally retained wheel cache because this sandbox cannot resolve PyPI
DNS. The separate live-network trial terminated rc=2 after 6.61 s; the wrapper did not publish a
partial dependency target. The final target contains 10 files / 1,931,498 bytes, with content
manifest SHA-256 `bfe25761e26f32b1dca1f7114a45648fa9b25dc8f98f0fa2e98b199992fd4a4b`.
An invalid-target positive control failed 1/1 with rc=65. These are terminal process results, not
partial log reads.

This receipt proves dependency closure and import of the real receiver on one macOS arm64 Python
3.11 host. It is not a full archive inflate, n600 decode, or contest-platform result.

## Contest wheel and source-build risk

The current workflow runs either `ubuntu-latest` or `linux-nvidia-t4`, pins a 30-minute whole-job
timeout, and the upstream project pins Python 3.11. PyPI's current 0.5.0 release lists a CPython 3.11
`manylinux_2_17_x86_64` wheel at a displayed 410.4 kB with SHA-256
`eb7909d0ad4940d3b74696d98f0dc16dec7294e57f9e0797bc06d5ce7b3b1507`; it lists no source
distribution. See the [PyPI 0.5.0 release](https://pypi.org/project/constriction/0.5.0/).

The cross-platform `uv` attempt selected that exact wheel URL, then terminated rc=1 because the
Linux wheel bytes were not in the offline cache. Therefore:

- **wheel availability is supported by primary package metadata and an actual resolver selection;**
- **Linux wheel execution is not measured here.** This platform statement is a `TOY-BRACKET`, not
  a contest-host closure receipt;
- **a source build cannot occur through the cured entrypoint.** `--only-binary constriction`
  forbids it. If a compatible wheel is unavailable, the entrypoint fails closed instead of spending
  the job budget compiling Rust.

## Whole-job budget

The 1,800-second limit covers the entire GitHub Actions job, not merely `inflate.sh`. The measured
17.50-second cold entrypoint smoke consumes 0.972222% and leaves 1,782.5 seconds for every other
job step. The wheel installation itself consumed 0.223 seconds from a local cache; network fetch
latency remains unmeasured.

RR3 retained an independent RTX 5070 full-wrapper completion at 142.098 seconds with
`constriction==0.5.0`. Adding the present 17.50-second smoke produces a deliberately conservative
159.598-second cross-host projection, leaving 1,640.402 seconds (8.866556% used). That sum
double-counts receiver startup, mixes macOS with RTX 5070, and omits current checkout/LFS/base-env
setup. It is budget context, not a measured contest total. A current Linux whole-job receipt is
still required before shipping.

## Declared dependency versus vendored decoder

| Arm | Archive/rate price | Runtime price | Proof state |
|---|---:|---:|---|
| Exact declared dependency plus fail-closed self-install | 0 archive bytes; rate delta 0 | 17.50 s cold smoke here, including 0.223 s cached install; network fetch unmeasured | **Executed:** clean-host receiver import rc=0; Linux execution not measured |
| Vendor the official Linux CPython 3.11 wheel as generic OSS code | 0 archive bytes; rate delta 0 | 410.4 kB displayed code payload plus extraction/import glue; install and decode time unmeasured | **Unexecuted:** no Linux wheel bytes under this arm's custody |
| Port the queue range decoder and categorical quantizer to pure Python/stdlib | 0 archive bytes; rate delta 0 | Code size and n600 runtime unknown | **Unexecuted:** 0 compatible receipts found in the bounded search |

Both arms have zero archive delta because the dependency or a generic decoder implementation lives
in submission code, not `archive.zip`; rule 118 charges video-derived large artifacts, not generic
tools. Under the current evaluator the rate remains
`191052 / sum(file sizes under the current videos directory)`. I did not reuse 37,545,489 as the
current denominator.

**Recommendation: ship the declared, wheel-only, fail-closed bootstrap as the present closure.** It
is the only arm that executed the actual derived entrypoint from a host where the dependency was
absent. Vendoring remains structurally preferable once the exact Linux wheel is under custody and
passes the same receiver plus full-inflate receipts; declaring it preferable does not make its
unexecuted compatibility or runtime real.

The bounded vendor search found two pure decoder functions by the exact queried names in submission
trees, plus several other range-coder implementations under `src/`. None carried a receipt proving
compatibility with the PR130 `constriction.stream.queue` word stream and its per-symbol categorical
models. Treating a different range coder as a drop-in decoder is therefore closed for this landing,
with verdict scope limited to the searched `src`, `submissions`, and `.omx/research` corpus.

## Ranked residual risks and falsifiers

1. **Contest-network/bootstrap risk — high, INSTANCE scope.** The local live-network trial failed
   1/1 because sandbox DNS was unavailable. Falsifier: a clean Linux CPython 3.11 contest-like job,
   with a fresh target and no cached `constriction`, fetches the pinned wheel, imports the real
   receiver, and terminates rc=0 inside the whole-job timer.
2. **Linux native-wheel execution — high, INSTANCE scope.** PyPI and `uv` selected the correct
   wheel, but this macOS host could not load it. Falsifier: record its SHA-256, import it on
   glibc x86_64 CPython 3.11, and verify the two required APIs from the derived entrypoint.
3. **Full decode identity — high, INSTANCE scope.** The final wrapper imported the receiver but did
   not inflate the canonical 191,052-byte archive. Falsifier: a terminal full-wrapper run whose
   output hashes match the established PR130 decode receipt and whose total job time is below 1,800
   seconds.
4. **Vendored decoder parity — medium, FORMULATION scope.** No searched pure decoder is proven
   wire-compatible. Falsifier: cross-decode the canonical PR130 token stream with exact token and
   inflated-output equality, then measure n600 runtime.
5. **Dependency source substitution — low, INSTANCE scope.** The version is pinned but the current
   bootstrap does not pin one hash across all platforms. Falsifier/cure: place the exact contest
   wheel under custody or add platform-aware hash verification without re-enabling source builds.

## Could not check / why

- I could not execute the Linux x86_64 wheel: this host is macOS arm64 and the sandbox could not
  download the selected Linux wheel.
- I could not time a live PyPI fetch: sandbox DNS failed. The measured install is cache-backed and
  is labelled that way.
- I did not run full n600 inflate or exact evaluation: the charter permits dependency-entrypoint
  smoke, this arm owns no scorer slot, and no contest Linux/CUDA host is present.
- I did not compute a current absolute rate denominator: no current evaluator `videos/` tree was
  materialized. The code-defined dynamic denominator is preserved instead.
- I did not vendor a decoder: no compatible vendored implementation or Linux wheel was under this
  arm's custody, so implementing a substitute would have exceeded the real closure proved here.

## RECALL EVIDENCE

Queries covered `constriction|brotli|dependency closure|self-install|RangeDecoder|PR130|#666|r5`
across the memory registry, `.omx/research` memos and receipts, `CANONICAL_RESEARCH_INDEX*`, the
`sub015_DAG_*` FEED blocks, task/lane ledgers, source/submission trees, and the canonical equations
registry produced by `tools/list_canonical_equations.py --json`.

Beyond the charter's RR3 seeds, the search found:

- the e4/FEED-603 declaration-and-fail-closed structural precedent;
- IC2's actual clean-venv self-bootstrap receipt, which changed this arm from a manifest-only cure
  to an executed isolated-target bootstrap;
- canonical equation `pr95_family_l42_lazy_brotli_auto_install_bootstrap_v1`, which reinforced lazy
  bootstrap at the receiver boundary;
- the current upstream workflow's Python 3.11/Linux/T4 surface and whole-job timeout;
- an independent completed wrapper receipt with `constriction==0.5.0`, which selected 0.5.0 for the
  final pin rather than relying on the intake's older 0.4.2 environment; and
- several non-constriction range coders, which changed the vendor conclusion from “no code exists”
  to the narrower, evidence-correct statement “no searched implementation is proven compatible.”

No relevant memory-registry entry was found for `PR130|fx1|constriction dependency closure`; that
negative is scoped to the registry query, not the full corpus.

## Follow-on dispositions

- **QUEUED-WITH-A-FIRE-ORDER:** owner = next PR130 candidate-assembly arm; consumer store =
  `.omx/research/ddm_pr130_reproduce_20260809/FX1_LINUX_DEPENDENCY_AND_INFLATE_RECEIPT.json`;
  fire trigger = before the first public PR130-derived evaluation, on clean Linux CPython 3.11,
  first prove network bootstrap and then run the canonical full wrapper to terminal status.
- **FOLDED for this landing:** owner = FX1; consumer store = this memo; fire trigger to reopen =
  the Linux bootstrap fails on the contest surface or a constriction-wire-compatible vendored
  decoder receipt appears. Until then, do not replace the proven bootstrap with an unproved port.

Frontier status: PR130 CPR1 remains `S=0.172141297491896447` at 191,052 bytes
`[contest-CUDA, DALI GT, n600]`; this scorer-free dependency cure did not produce a new row.
