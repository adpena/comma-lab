# Submission PR draft — level-set task-space witness (v9 line)

**Status: DRAFT — no PR until every `${...}` placeholder is a measured value and the Final Gate
passes.** This supersedes the stale Apogee-era `docs/submission_template.md` for the witness line
(that template's "GPU required: yes" and PR100-adapter framing do NOT apply here).

Internal rules for the public body (binding):
- PR is authored/attributed to the operator (adpena). NO Claude/AI attribution anywhere in the
  public body, commits, or branch (memory L62/L63).
- No private paths, fleet IPs, provider transcripts, or `.omx` internals in the body. Links only to
  the hosted release asset + (optionally) a sanitized public writeup.
- Every score in the body comes from `upstream/evaluate.py` on the EXACT hosted archive bytes.
  Full-precision numbers on BOTH axes, like PR #129 did (`0.190506` CPU local, `0.190503` x86 CI) —
  Yousfi's bot evals on x86 `ubuntu-latest`; report both our local contest-CPU row and note the
  expected CI axis. NEVER quote the rounded 2-dp `final_score`.

## Why this draft exists now (process signal from #125/#127/#128/#129, 2026-07-12)

1. **Leaderboard placement happens on EVAL, independent of merge.** Yousfi added #125/#127/#128 to
   the leaderboard while refusing all three merges for code overlap with already-merged submissions.
2. **Merge requires refactor-to-reuse.** #129 is #127 resubmitted importing `rhnerv_comma` modules
   instead of copying them. OUR vehicle has no overlap problem — the witness shares no code with any
   merged submission — so we can honestly answer "want it merged: yes" and expect it to stick.
3. **Attribution is rewarded.** #129 ships a `THIRD_PARTY_NOTICES.md` naming PRs #95/#98/#101/#110/
   #112/#125 and explicitly flags concurrent-independent work. We do the same (list below).
4. **Hosting pattern that works:** GitHub release asset on the author's fork, `curl -L`-able, with
   SHA-256 + byte count + ZIP member layout stated inline.

## Final Gate (ALL must pass before `gh pr create`)

- [ ] Byte-close on the FINAL checkpoint: `tools/levelset_byte_close_and_eval.py` full n600, archive
      bytes frozen, SHA-256 recorded.
- [ ] `scripts/pre_submission_compliance_check.py --contest-final --strict` with
      `--expected-archive-sha256` + `--expected-archive-size-bytes` + auth-eval JSON.
- [ ] Exact `[contest-CPU]` row (Linux x86_64 — Modal CPU or GHA) on the exact bytes.
- [ ] Exact `[contest-CUDA]` row (T4-class — Modal) on the exact bytes. Both axes, never inferred.
- [ ] Inflate wall-clock < 30 min on contest-class hardware (re-confirm on the final archive; #214
      established the path).
- [ ] Archive hosted as a release asset on the operator's fork; `curl -L` round-trip verified,
      downloaded SHA matches.
- [ ] 5-turn consecutive clean-pass adversarial review of the score + packet (CLAUDE.md "Submission
      PR gate" — stricter than the 3-pass greenup).
- [ ] Beats public best `0.187946` [contest-CPU] on OUR measured row — otherwise the
      "competitive" claim is softened to "innovative" only (see field below; do not overclaim).
- [ ] VERIFY-at-close list resolved (marked `VERIFY` below): final dep list of `inflate.py`; whether
      any PR98-style decode-side bias is active (currently believed NOT); final section inventory.

## The PR body (fill placeholders, then paste verbatim)

````markdown
# submission name:
levelset_taskspace_witness

# upload zipped `archive.zip`
Hosted as a release asset (`curl -L` works):
${RELEASE_ASSET_URL}

SHA-256 `${ARCHIVE_SHA256}`, ${ARCHIVE_BYTES} bytes, single ZIP member `0.bin` (ZIP_STORED).

# report.txt
```
${PASTE FULL report.txt FROM upstream/evaluate.py ON THE EXACT HOSTED BYTES — CPU AXIS}
```
Full precision: `${SCORE_CPU_LOCAL}` on local Linux-x86_64 CPU
(seg `${DSEG}`, pose `${DPOSE}`, rate `${RATE}`); T4 CUDA: `${SCORE_CUDA}`.
(`evaluate.py` prints 2 dp; the numbers above are recomputed from the printed components.)

# does your submission require gpu for evaluation (inflation)?
no — inflate is CPU-only (device pinned to CPU, deterministic fp32; same `archive.zip` →
bit-identical output across runs/hosts). Deps: `numpy`, `brotli`${VERIFY_DEPS: torch? none else}.
Inflation completes in ${INFLATE_MINUTES} min on a contest-class CPU runner.

# did you include the compression script? and want it to be merged?
yes — `compress.sh` rebuilds `archive.zip` byte-for-byte (seeded + deterministic: single recorded
seed, same inputs → same bytes). The training/optimization pipeline is described below. The
submission shares no code with previously merged submissions (new decoder family, own archive
grammar and coders), so merge should not raise the code-overlap concern from #125/#127/#128.

# is this submission competitive or innovative? explain why
${PICK ON MEASURED ROW — do not overclaim:}
Competitive: CPU `${SCORE_CPU_LOCAL}` vs current leaderboard best `0.187946` (#128).
Innovative (holds regardless of rank): this is NOT an HNeRV/NeRV descendant. The decoder is a
task-space level-set witness — a small coordinate-INR trained ONLY against the frozen scorers
(no full-RGB reconstruction objective), spending its bytes on the SegNet argmax partition
(the codim-1 decision boundary) and the PoseNet-relevant photometric structure instead of on
pixel fidelity. Everything deterministic and generic lives in `inflate.py` (free per the rules);
`archive.zip` carries only the learned/video-derived payload: the INR weights, the per-pair ego
twist ξ (quantized + entropy-coded, ~${DXI_BYTES} B), and small per-class seeds. Training runs
through the exact inflate chain (bicubic→874×1164, uint8 straight-through, exact packing grid),
so the optimizer sees exactly what ships. The boundary objective is a level-set/margin surrogate
(smooth argmax-flip fraction with annealed temperature) — developed independently in this work
from the level-set formulation; we note the convergent use of a `sigmoid(-margin/τ)` surrogate in
PRs #125/#127 as concurrent independent evidence the boundary-native objective is the right one.
Pose is not reconstructed from pixels: the render is conditioned on the stored ξ (joint-descent
trained), a design in the spirit of #55/#56's stored-target conditioning.

# additional comments
Lineage & attribution (full list in `THIRD_PARTY_NOTICES.md`):
- Contest scaffold: `upstream/evaluate.py`, frozen SegNet/PoseNet, `inflate.sh` convention (commaai).
- Conceptual lessons (no code or weights reused): the PR #95 family's staged-curriculum + EMA
  discipline (#95/#100/#101); #55/#56 stored-target/FiLM conditioning concept; concurrent
  boundary-surrogate work in #125/#127 (independent, acknowledged above).
- Train-time priors only (never shipped, never counted): openpilot lane/camera geometry and
  comma10k class conventions (commaai's own public stack).
- No code, weights, latents, or archive sections from any merged or open submission are included.
Reproduction: `compress.sh` (deterministic, seeded); evaluation is the canonical
`archive.zip -> inflate.sh -> upstream/evaluate.py` path on CPU.
${OPTIONAL_WRITEUP_URL}
````

## Internal borrowed-substrate accounting (NO-FAKE #7 — stays internal, backs the public claims)

| Component | Ours / borrowed | Notes |
|---|---|---|
| Decoder (coord-INR level-set witness) | OURS | `lever_b_generator` lineage; no NeRV/HNeRV code |
| Archive grammar (length-prefixed monolithic sections) | OURS | own layout; brotli entropy backend (public lib) |
| ξ pose coder (`xi_pose_coder`, derive-H) | OURS | #257; sidecar-SHAPED, joint-descent VALUES |
| Curriculum (event-gated witness-native) | OURS | derived from level-set energy (#302/#430); PR95 staging = ancestral lesson only |
| Boundary surrogate (level-set margin) | OURS (concurrent: #125/#127) | acknowledge convergence in body |
| Stored-pose conditioning CONCEPT | Quantizr #55/#56 concept | our implementation + joint descent |
| Exact-chain training (train/pack-gap closure) | OURS (convergent with #125/#127/#129) | our eval_roundtrip discipline predates; their exact-grid framing validates |
| openpilot/comma10k priors | commaai public | train-time only, rule-118 free side |
| Scorers/eval harness | contest upstream | unmodified (pinned snapshot) |

**Verdict this table supports:** the innovation claim is UNQUESTIONABLE per the Innovation Gate —
ours-original mechanism list is the vehicle itself; borrowed items are concepts + the contest
scaffold, all attributed. This is the opposite pole from the banked 0.18804 splice
(PR128-on-PR110 = borrowed substrate, NEVER submitted as ours).
