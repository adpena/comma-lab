# submission name: ck1_composed_rowprune

Prepared by the repository operator. **This is a hold-state draft.** It must not
be opened as a pull request until the download URL, source-visibility check,
strict compliance pass, and five consecutive clean review passes are complete.
Generation 4 of this packet; it supersedes the sz1-composed, rr4, hv1, and e480b
drafts.

# upload zipped `archive.zip`

Download status: pending operator-authorized public hosting. No public URL is
claimed in this draft.

Exact file identity:

- 177,182 bytes
- SHA-256 `35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3`
- single stored member `p`, 177,082 bytes, SHA-256
  `ee904fbf6b10e4fadd69ca9c820bd7db0d334694bdf23c4a93147cd242d8c462`, CRC32 1722708006

# report.txt

The complete `report.txt` shipped beside the archive, copied verbatim:

```text
=== Exact result identity ===
Evidence axis: [contest-CUDA]
Hardware: Tesla T4, Linux x86_64
Samples: 600
Archive SHA-256: 35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3
Archive size: 177182 bytes
Member: p, 177082 bytes, stored, SHA-256 ee904fbf6b10e4fadd69ca9c820bd7db0d334694bdf23c4a93147cd242d8c462
Runtime tree SHA-256: da91e06744b94f77077303b2b760cb259aa84b078d998921fb99e018d52fff6f
Portable runtime content tree SHA-256: 944c8c574f377cbe625c007b44bfc8e88ec572bf3fc7a2e9ac7aca5750217078
Upstream snapshot SHA-256: cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008
Upstream evaluate.py SHA-256: 7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b

=== Evaluation results over 600 samples ===
Average PoseNet Distortion: 0.00000777
Average SegNet Distortion: 0.00030309
Seg contribution: 0.030309
Pose contribution: 0.008814760348415605
Rate contribution: 0.11797822103209257
Recomputed score: 0.15710198138050818
Reported (2 dp display): 0.16
Report-8dp worst-case absolute score error bound: 3.336608391523776e-06
Inflation wall time: 1197.696784587 seconds
Evaluation wall time: 40.764544933000025 seconds
Total authority wrapper wall time: 1249.358265802 seconds
Inflate budget: 1800 seconds; measured headroom 1.503x

=== Relationship to the prior candidates ===
Prior packet generation 3 measured S = 0.15771357797660338 at 179930 bytes on the
same axis and the same hardware class. This archive is -2748 bytes and
delta S -0.0006115966 against it.

The intermediate row between them (177576 bytes, S 0.1571619225142182) is the same
lineage with the row-prune applied and no compensation edit; this archive is
-394 bytes and net delta S -5.994113e-05 against that row.
The measured leg split against it sums to the net:
  rate -2.6235e-04 (-394 bytes)
  seg  +1.7400e-04
  pose +2.8407e-05
  sum  -5.994113e-05
Unlike generation 3, this candidate does NOT hold decoded state constant: the
edit changes semantic-section values, so both distortion legs move and are paid
for out of the rate credit. Retained fraction of the rate credit: 0.228.

=== CPU boundary ===
Evidence axis: [env-mismatch advisory] -- NOT a score, and NOT [contest-CPU].
A full 600-sample local decode-and-score of these exact archive bytes ran on
macOS arm64 (not the contest 4-thread x86_64 runner): inflate 941.187799999956 s,
evaluate 406.79247495904565 s. The receipt stamps
score_axis=cpu_env_mismatch_advisory, evidence_grade "auth-eval env mismatch advisory",
score_claim=False, promotion_eligible=False.
It is retained as a decode-correctness proof: the rate contribution
(0.11797822103209257) is identical to the T4 row's, and the pose
residual (0.00014829) matches the encoder-side authority solve to
8 decimal places, which is what proves the composed receiver decodes the composed
container correctly.

Status of the [contest-CPU] axis on these exact bytes: NO ROW EXISTS. The prior
generation-3 lineage measured contest-CPU inflate at 3422.711146813 s
against the 1800 s budget on a contest-like 4-thread x86_64 CPU,
and this candidate inherits that token decoder
unchanged, so the axis is expected to remain infeasible -- but that expectation is
INHERITED, not measured on these bytes. No CPU score exists and none is claimed.
This submission is GPU-required for evaluation.

=== Provenance ===
Candidate seal: ck1_composed_rebased_r4, seal SHA-256 a64b3483c0d5d3b5a589c45590c503db94287b7182154f5e6c675149ddef65e3
Seal validation at fire time: SEAL_VALID
Torch: 2.5.1+cu124; CUDA 12.4; driver 580.95.05
Source commit pinned into the eval container: 9e194bc1a7fe80e501752bb493d83a63b83d57a6
Provider job identifiers are retained privately with the authority receipts and
are deliberately not reproduced on this public surface.
```

# eval host info

Linux x86_64, Tesla T4, all 600 samples, unmodified upstream scorer
(`evaluate.py` SHA-256 `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`,
upstream snapshot `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`).
Inflation used 1,197.7 s of the 1,800 s budget — 1.50x headroom. The
token-mixer decode is the dominant cost; it is deterministic integer
arithmetic.

# build cost info

No public total-training-cost claim is made. This submission adds **no training
cost at all** over its base candidate. Nothing was trained: the semantic tensors
are re-quantized arithmetically, and the pose carrier is re-solved by a
deterministic numerical solve against the frozen scorer at compile time. Both
run in minutes of CPU from the retained checkpoint. The base candidate's
training cost is a separate figure and is not reconstructed after the fact here.

# does your submission require gpu for evaluation (inflation)?

**Yes — this submission requires a GPU for evaluation.** The measured score
above used a T4 (inflation 1,197.7 s of the 1,800 s budget).

We state the basis honestly rather than borrowing a stronger one. An earlier
candidate in this same lineage, sharing this token decoder unchanged, was
measured on a contest-like 4-thread x86_64 CPU at 3,422.7 s of inflation against
the 1,800 s budget; that harness failed closed at 1,800 s. **These exact bytes
have not been measured on a contest CPU.** No contest-CPU score exists on them
and none is claimed. The decode itself is correct off-GPU — a full 600-sample
local decode-and-score of these exact bytes ran to completion — so the boundary
is wall-clock, not correctness.

# did you include the compression script? and want it to be merged?

**Yes, and it is offered for merge — with one scope reduction stated up front.**
The chain, stated exactly:

- **Stage A — provenance (documented, not re-run).** Reproducing the underlying
  checkpoint from raw video is multi-day GPU compute. The chain emits the
  lineage, the stage scripts, their arguments, and the input manifest with
  every SHA-256. It does not pretend to re-run training.
- **Stage B — build (exact and verifiable).** From the retained checkpoint the
  chain replays the shipped decode order, re-quantizes the semantic section
  under the row-prune / mixed-depth format, re-solves the frame-0 pose carrier
  against the frozen scorer, and repacks the archive. Each stage hashes its
  output and fails closed on mismatch. The build additionally asserts, before
  it will run at all, that the decoded state it is compensating against is the
  state the compensation was solved on — a compensation carried onto a changed
  lattice is a silent wrong answer, and this repository has shipped that bug
  once.
- **Stage C — decode.** Runs the shipped receiver over the archive and checks
  the parsed codes against the encoder's — `PASS` at
  `max_abs_code_deviation = 0`.

**Scope reduction, stated rather than inherited.** The previous generation could
claim that one entry point rebuilt its exact bytes end-to-end from pinned
retained inputs. **That entry point has not been re-run for this candidate.**
The compile receipt proves how these bytes were assembled and the receiver
parse-back passes over the shipped runtime, but the previous generation's
end-to-end VERIFIED label does not transfer and we do not carry it forward.

# changes from upstream

This submission changes **two sections** of an inherited archive. Unlike every
prior generation of this packet, the changes are **not lossless** — the decoded
state moves, and both distortion terms are paid for out of the byte saving.

1. **Semantic section — row-pruned, mixed-depth re-quantization.** Three FiLM
   weight tensors keep only their two highest-L2-norm rows, transmitted as a
   row bitmask plus a compact kept-rows block; a per-tensor 4-bit depth table
   then drops `frame_embed.weight` and `blocks.0.film.weight` to 3-bit codes
   while the remaining quantized tensors stay at 4. The receiver recomputes the
   tensor selection mask and refuses on mismatch. The resulting semantic stream
   is 31,469 bytes.
2. **Pose carrier — compile-time frame-0 compensation.** The re-quantization
   damages PoseNet, because PoseNet reads the frame pair while the semantic
   renderer produces only frame 1. The frame-0 carrier lattice is therefore
   re-solved at compile time — a damped Gauss-Newton step on the
   receiver-realized Jacobian followed by a multi-scale integer descent — so
   the induced pose error is cancelled in the frame SegNet does not read.
   6,713 of the 7,200 signed-int12 carrier coordinates change; the compensation
   costs 41 archive bytes and cancels 99.98% of the leakage energy in the local
   solve.

The token stream and the HPAC stream are spliced byte-identically from the
previous candidate. The previous candidate's semantic byte-plane serialization
split is **off** here: the row-prune changes the semantic body length, and
re-measured on the edited body that split is negative. Its receiver support
ships and is inert on these bytes.

# competitive or innovative?

**Competitive, on a measured row, stated against what is actually verified.**

On the exact submitted bytes the measured `[contest-CUDA]` 600-sample score is
`0.15710198138050818`, which we re-derived from the reported components
independently. That is below the best ranked score on the leaderboard at the
time of writing (PR #135, `semantic-pose-HPAC_CPR1_polished`, 0.162) and below
our own prior custodied rows.

Four honesty qualifications we would rather state than have found:

1. There is an open PR claiming `0.1591495384` (PR #138, `opal_v1`). That
   figure is **author-claimed and not yet evaluated by the maintainers**, as is
   ours until this PR is run. Our measured number is lower than that claim,
   but we are comparing a measured row against an unverified one and say so.
2. **PR #138 published the decode-time-corrector mechanism class first, and we
   did not know it when we built our first corrector.** Its online correction
   is learned from the already-decoded prefix, reproduced identically by
   encoder and decoder, adds no table or weight to the archive, and yields
   pure rate — the same class as our token probability model, by a different
   construction. PR #138 opened 2026-08-17 08:31Z; our first measured
   corrector result landed 14:41Z the same day, and we first read PR #138 at
   19:32Z, after our byte-close. We describe this as **concurrent independent
   development** and make **no priority claim**. PR #136 is adjacent and also
   earlier.
3. **The edit-then-recompensate pattern is PR #135's, not ours, and our solver
   adapts theirs.** PR #135's own competitive mechanism was a joint renderer
   edit followed by a frame-0 carrier re-solve, and the damped Gauss-Newton
   form plus bounded integer-cube solver our compensation uses are adapted from
   its published experiment book. What is ours in that mechanism is narrower:
   the compile-time binding that refuses to carry a compensation onto a changed
   lattice, the frame-0/frame-1 disjointness argument that makes the
   compensation SegNet-invariant by construction, and the rate route that folds
   the compensation into the existing Rice-coded lattice instead of a sidecar —
   which is what turns roughly 7,000 bytes into 41.
4. **This generation stops being a purely lossless program, and the
   originality claim gets narrower, not wider.** Previous generations could say
   "we re-coded their bytes and the decoded state is provably identical." This
   one re-quantizes their trained semantic tensors and re-solves their pose
   carrier. The learned vehicle underneath is still PR130/PR135 lineage and is
   still not ours; what changed is that we no longer reproduce it faithfully.
   The one learned object in the archive that *is* ours remains the HPAC
   probability object: PR130's architecture, retrained here on our own label
   field, inherited unchanged in this generation.

# additional comments

## Score and runtime boundary

The CUDA score is a 600-sample exact evaluation of the archive hash printed
above through the unmodified upstream scorer. CPU and CUDA are separate axes.
On the CPU axis we report an absence rather than a pending promise: **no
contest-CPU row exists on these bytes.** The nearest measurement is on an
earlier candidate in this lineage sharing the same token decoder, where CPU
inflation took 3,422.7 s against the 1,800 s budget. We do not transfer that
number onto these bytes as if it were measured here; we state it as the reason
the axis is expected to remain infeasible. The dominant term is the token-mixer
decode, which is the named optimization surface if CPU feasibility is ever
wanted.

## What the distortion legs cost

This is the first candidate in this packet whose improvement is not purely
rate. Against the immediately prior row in this lineage (177,576 bytes,
S 0.1571619225142182) the measured legs are rate −2.6235e-04, SegNet
+1.7400e-04, PoseNet +2.8407e-05, for a net of −5.994113e-05. Roughly 23% of
the rate credit survives the two distortion payments. The seg leg came in near
three times its pre-fire model, and the reason is a real gap in our own
instrumentation rather than a surprise about the vehicle: we have a measured
CPU-to-CUDA transfer relationship for the pose term and none for the seg term,
so the CPU-side estimate that fed the projection was an upper bound on the win.
We record that as an open gap.

## Borrowed-substrate accounting

Classes: `inherited-substrate` (theirs, used as-is) · `mechanism-adopt-with-
attribution` (their idea or source, our implementation or re-fit) ·
`ours-original` (built here, with a receipt).

**Read this first: the two largest rows changed class at this generation.** In
every previous generation the semantic renderer state and the pose carrier state
were `inherited-substrate` and *proven byte-identical to PR135 after decode*.
That byte-identity is gone. We now ship a lossy re-representation of their
semantic tensors and a re-solved version of their pose carrier. That is not an
upgrade in our favour — it means we can no longer claim faithful reproduction of
their work, and it does not make the underlying learned content ours.

| Section or mechanism | Classification | Receipt and boundary |
|---|---|---|
| Semantic renderer state | **`mechanism-adopt-with-attribution` — our format over their values, and the values are lossily changed** (was: `inherited-substrate`, byte-identical) | semantic body 36,130 B, stream 31,469 B; row-prune keeps 2 rows each of `blocks.{1,2,3}.film.weight`; `frame_embed.weight` and `blocks.0.film.weight` at 3-bit |
| Pose carrier state | **`mechanism-adopt-with-attribution` — their solver form, our binding, their lattice re-solved** (was: `inherited-substrate`, byte-identical) | 6,713 of 7,200 signed-int12 coordinates changed; +41 archive bytes; 99.98% leakage cancellation in the local solve |
| Compressed model container | `inherited-substrate`; unchanged from base, PR-level equality not independently verified | 70,453 B, `e35d12371fa79747…` |
| **HPAC probability object** | **`mechanism-adopt-with-attribution`** — PR130's architecture, **retrained here on our own label field**; inherited unchanged in this generation | HPAC stream 13,515 B, spliced byte-identically from the prior candidate |
| Residual payload + table codes | `inherited-substrate`; **provenance unresolved, no originality claimed** | carried inside the 109,897 B tail, spliced byte-identically |
| **RC64 token stream** | **`ours-original` probability model over inherited symbols** — 13-context fixed-point integer log-odds mixer, zero archive bytes | unchanged from the prior candidate; inside the 109,897 B tail |
| **Row-prune / mixed-depth semantic format** | **`mechanism-adopt-with-attribution`** — magnitude-based structured pruning and mixed-precision weight quantization are standard practice with a long public literature, and the tensors are PR135's; ours are the wire format, the measurement identifying the two surviving marginal tensors, and the fail-closed receiver integration | semantic stream 31,469 B; receiver recomputes the selection mask and refuses on mismatch |
| **Frame-0 pose compensation** | **`mechanism-adopt-with-attribution`** — the edit-then-recompensate pattern and the Gauss-Newton / integer-cube solver form are PR135's; ours are the compile-time content-fingerprint binding, the SegNet-invariance-by-construction argument, the step-matched Jacobian, and the in-lattice rate route | +41 B against a zero-compensation control; encoder-side only dependency on PR135's experiment-book source, nothing from which enters the archive |
| Carrier framing runtime patch | `ours-original`, **zero counted bytes** | `runtime/residual_archive.py`: packed-CAP1 length derived from the section's own bit counts instead of a pinned constant; generic framing algorithm, no video-derived content |
| Semantic serialization split | **DROPPED this generation** | reserved bit 0; re-measured negative on the edited body |
| RC64 backend, encoder side | `inherited-substrate` (PR135, verbatim) | compiles PR135's `rc64_backend.c` unmodified |
| RC64 backend, shipped receiver | `mechanism-adopt-with-attribution` (PR135-derived, modified) | shipped `05839d1416e68a49…`, which **differs** from the PR135 source |
| Receiver binding and archive assembly | `ours-original` | validated runtime tree `da91e06744b94f77…`; archive `35c318d541d70370…` |
| Compression chain | `ours-original` **— e2e re-verification NOT carried forward for these bytes** | compile receipt with fail-closed identity assertion; the single-entry-point rebuild claim belongs to the previous generation only |

The full accounting, including the ancestry chain, the re-classification
argument, and the open provenance items, is in
`BORROWED_SUBSTRATE_ACCOUNTING.md`, shipped beside this archive.

## Credits and prior work

Every number below was read from the pull request itself, not from our notes.

- **PR #130 — `semantic-pose-HPAC_CPR1`, Fesal Fayed (`fesalfayed`)**, leaderboard
  0.172, archive 191,052 B. The base vehicle: the semantic-token / HPAC / CPR1
  architecture this submission descends from.
- **PR #135 — `semantic-pose-HPAC_CPR1_polished`, Shreyan Mohanty (`codexblack`)**,
  leaderboard 0.162, archive 186,724 B. The archive we actually built on. Its
  semantic renderer and pose carrier are the learned content inside ours; our
  encoder compiles its `rc64_backend.c` unmodified, and our pose compensation
  adapts the solver form from its published experiment book. It is also the
  ranked score this submission is measured against.
- **PR #133 — `cpr1_cbq_matched8`, `JasonMo123`**, leaderboard 0.166. Not taken
  directly, but in our ancestry transitively: PR #135 already incorporates its
  constrained basis and re-solved int12 carrier.
- **PR #138 — `opal_v1`, Cristian (`ccastillo1043`)**, author-claimed 0.1591495384.
  Published the decode-side probability-correction mechanism class first; see
  the competitive section above. Concurrent and independent; no priority claim.
- **PR #136 — `hnerv_rc`, Jacky Li (`JPL11`)**. Adaptive range coding with
  per-tensor context reset, on a different vehicle. Adjacent prior work.
- **Upstream** — `commaai/comma_video_compression_challenge`: the scorer,
  `evaluate.py` (`7da71a84ce24286b…`), the frozen SegNet and PoseNet weights, the
  600-sample test list, and the 37,545,489-byte denominator.
- **Third-party runtime** — PyTorch, NumPy, Brotli 1.2.0, and a C compiler
  at inflate time.

## Public source and reproducibility

- Source repository: https://github.com/adpena/comma-lab
  (anonymous visibility to be re-verified at packet freeze)
- Evaluation source pin: commit `9e194bc1a7fe80e501752bb493d83a63b83d57a6` —
  the commit the T4 evaluation actually ran from (matches
  `provenance.pact_commit` in the receipt)
- Encoder source pins: to be re-pinned at packet freeze against the public
  mirror; the compile receipt for these exact bytes is retained with the
  packet's authority receipts
- Validated runtime tree SHA-256:
  `da91e06744b94f77077303b2b760cb259aa84b078d998921fb99e018d52fff6f`
- Portable runtime content tree SHA-256:
  `944c8c574f377cbe625c007b44bfc8e88ec572bf3fc7a2e9ac7aca5750217078`
- Upstream snapshot:
  `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`
- Chain scripts: `experiments/ddm_rr2_encoder_byteclose.py` (encode/build),
  `experiments/ddm_rr2_receiver_close.py` (receiver/parse-back),
  `experiments/ddm_fx2_model_axis_corrector.py` and
  `experiments/ddm_fx1_logistic_mixer_corrector.py` (token probability model),
  `experiments/ddm_sm3_semantic_representation.py` (row-prune / mixed-depth
  semantic format), `experiments/ddm_sa3_rebase_sz1.py` (compensated compile).

Before submission the operator must verify anonymous visibility of the pinned
source URLs and replace the download-status paragraph with the verified public
archive URL and its hosted manifest. Two items are open by construction and
must not be quietly closed: the end-to-end rebuild has not been re-run for these
bytes, and no contest-CPU row exists on them.
