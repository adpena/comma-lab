# PR130 full stack: OFF-THE-SHELF vs PORTED — the per-component ledger

**Date** 2026-08-09 · **Operator frame** *"We know it works. It produced the score. We just need
to port it properly where it needs to be ported."* + *"important that we distinguish between
everything we can use off the shelf, which we have full authority to do, and that which must be
ported"* + *"we wanna start with the PR one thirty as our full stack And then do AB tests and
measurements and everything from there."*

**Dependency policy (operator 2026-08-09):** *"We can use any and all dependencies necessary."*
Consistent with CLAUDE.md L4 as amended — the dep cap is DELETED; the binding constraints are
(a) it installs/imports in the contest runtime inside the 30-min decode budget, (b) deterministic
decode, (c) rule-118 (no video-derived data smuggled as code or as a "dependency").

**BASE = PR130 CPR1 S = 0.172141297491896447** `[contest-CUDA, DALI GT, n600]`, archive 191,052 B
sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.

---

## The three-way classification

| class | meaning | authority |
|---|---|---|
| **OTS** | their file runs unmodified on our host; we invoke it directly | full authority, use as-is |
| **PORT** | device/runtime-specific work required; adapter lives in `src/tac/pr130_lift/`, intake stays READ-ONLY | build + verify |
| **OURS** | our own substrate/optimization, A/B-able against the OTS path | must beat OTS on a measured row |

`src/tac/pr130_lift/__init__.py` already carries this: `lifted/` = borrowed source with per-file
`borrowed_substrate_accounting` + `source_sha256`; top-level = our adaptation; and it explicitly
separates `LIFTED_AT_HEAD` (our copies) from `SOURCE_REPO_HEAD` (files we execute directly).
**That distinction is already load-bearing** — hb2's HPAC fix advanced the repo while the pin
read the old value, and the two-constant split is what caught it.

## Per-stage ledger — MEASURED status

| stage | file | class | status on Metal |
|---|---|---|---|
| `prepare` (GT cache) | `build_gt_cache_official.py` | **OTS** | ✅ ran |
| `semantic` (6k QAT tail) | `train_semantic_quantized.py` | **OTS** | ✅ ran, `verdict: PASS`, `packed_parameter_bytes: 40252` = shipped section exactly. `--device mps`, no edit |
| semantic inference | same, `evaluate_all` | **OTS** | ✅ DALI-GT 0.0002857038709852431 = 0.998650× published Ada; 19 s/n600 |
| `carrier` (pose, 4k int6) | `train_pose_carrier_full.py` | **PORT** | 🔧 **PORT VERIFIED, TRAINING RUN OWED.** 2 device sites: `torch.cuda.empty_cache()` :264 · `load_file(..., device='mps')`. Adapter `pr130_lift/pose/mps_port.py`. Sparse `nn.Embedding`+COO+`RowLocalSparseAdam` verified on MPS @ pinned torch 2.10.0 — zero cpu-fallback, `grad_is_sparse`/`grad_is_coalesced` True, row-local clocks `[1,2,1,2]` preserved, untouched rows bit-identical, CPU/MPS parity within predeclared fp32 tol. Receipt `ddm_pq1_probe_20260809/probe_torch2100_pinned.json`. **Scope: 2 steps × 4 rows.** The 4,000-step n600 carrier training has NOT been run here |
| `hpac-init` | `extract_integer_hpac_archive.py` | **OTS** | ✅ ran |
| `hpac` (60ep self-compress) | `train_hpac_self_compress.py` | **OTS** | ✅ ran, 59.18 s/epoch |
| `pack-hpac` | `pack_hpac_self_compress.py` | **OTS** | ✅ (hb2 fixed the round-trip upstream) |
| `encode-tokens` | `codec_hpac_integer.py` | **OTS + dep** | ✅ **BYTE-IDENTICAL on Metal.** 600 frames, 1,057 s, rc=0. `tokens.bin` = **116,980 B**, sha256 `948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb` — exact match to the intake's own `verify_file` expectation. Needed `constriction==0.5.0`; **no code change**. ⚠ **Consumed the BANKED `hpac_selfcompress_l1_fastbits_e60.pt`** (`canonical_hpac_checkpoint()` default, `train.sh`; `HPAC_CHECKPOINT` unset) — so this validates the CODEC PORT, not our hpac training run |
| archive assembly | `build_submission_archive.py` · `rebuild_submission_hpac.py` · `compress.sh` | **OTS** | ✅ **BYTE-IDENTICAL** 191,052 B, sha matches |
| verification | `scripts/verify.sh` | **OTS** | ✅ 24 passed, "CPR1 repository verification passed" |

**Only ONE stage in the whole chain needed a real port: the pose carrier.** Everything else is
off-the-shelf with `--device mps`, plus one pip install.

## PORT CLOSURE — scoped honestly (corrected by recursive review round 1, 2026-08-09)

**Every stage RUNS on this host. That is not the same as every stage being VERIFIED, and the ✅
column above conflated two verification regimes with wildly different strength.**

### The verification-strength ladder — what each ✅ actually proves

| regime | stages | bar met | what it proves | what it does NOT prove |
|---|---|---|---|---|
| **A — byte-exact** | `encode-tokens` · `pack-hpac` · archive assembly · `reproduce.sh` | sha256 identity vs the banked reference | The Metal forward reproduces the CUDA forward **exactly**. Arithmetic coding desyncs on any single differing probability, so byte-identity over 116,980 B / 600 frames means every one of ~118M per-pixel decisions matched | Nothing about our *training*: the inputs are the BANKED artifacts (`reproduce.sh` reads `artifacts/base/` + `artifacts/hpac/`; `encode-tokens` reads the banked `hpac_…e60.pt`) |
| **B — metric-approximate** | `semantic` inference | ratio 0.998650× published Ada on DALI GT | The semantic renderer reproduces to within 0.14% | Not bit-exact; a *ratio*, not an identity |
| **C — ran, output unverified** | `semantic` 6k tail train · `hpac` 60ep train · `prepare` | rc=0, `verdict: PASS`, plausible metrics | The training path executes on Metal | The produced checkpoints were never compared to the banked ones — and **cannot** be: CUDA→MPS training is not bit-reproducible |
| **D — port-verified, run owed** | `carrier` (pose) | 2 steps × 4 rows, CPU/MPS parity within predeclared fp32 tol | The sparse mechanism (nn.Embedding backward, COO coalesce, row-local clocks) survives on MPS at the pinned torch | The 4,000-step n600 training has not been run here |

**So: the DETERMINISTIC TAIL of the chain is closed byte-exactly. The STOCHASTIC HEAD (three
training legs) is executable but unvalidated-by-construction.** The genuine port was narrow — the
pose leg's two device sites. Everything else is off-the-shelf with `--device mps` plus one pip
install. That remains true and is the useful headline; it is the *strength* of the ✅s that was
overstated, not their existence.

### Also corrected at source

- The prior text read *"The chain that produced S = 0.172141297491896447 is reproducible here
  end-to-end."* **Withdrawn.** What is reproducible end-to-end is the assembly/encode tail from
  banked trained artifacts. The chain from *video* to *archive* has not been run here.
- The carrier row previously carried `mean d_pose 2.4437744286842644e-05` presented as a measured
  n600 result. **CORRECTED TWICE.** Round 1 withdrew it as "no locatable receipt." **ddm_rr4
  FALSIFIED that withdrawal**: the receipt exists at
  `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/reports/METAL_SMOKE_carrier.json`
  (sha256 `0c85e4a31928361e4f3977cd6365569937ea10a78d31e9b7bb5fa740e4d5ec6f`, 26,152 B), where the
  exact value is the **step-6 history mean over all 600 pairs** (`steps=4000`, `stop_after_step=6`,
  pair IDs 0–599; step-3 companion `2.3431171939591877e-05`), with a full-state checkpoint
  `…step000006.full_state.pt` (sha256 `abd09ca0…`, 362,405 B).
  **MECHANISM CORRECTED AGAIN (third pass on this one figure).** My first correction blamed a
  non-exhaustive scope — *"never looked in `ddm_pr130_train_20260809/`."* **That was also wrong.**
  The round-1 background search (task `bh2s037tu`) completed at exit 0 and returned **10 hits**,
  including `…/ddm_pr130_train_20260809/reports/METAL_SMOKE_carrier.json` — the exact file — and
  `.omx/research/ddm_pr130_reproduce_20260809/LOCAL_TRAINING_AUDIT.md`, **a doc in my own custody
  directory for this very work.**
  The search was correct and exhaustive over the right scope. **I read its output while it was still
  running, saw an empty buffer, and converted "not yet printed" into "does not exist."** The harness
  had told me the command was moved to the background; I consumed the partial file anyway and never
  re-checked completion.
  This is the **#50 vacuity genus, not m53's scope genus**: an unfinished search returning nothing is
  indistinguishable from a finished search returning nothing *unless you check the completion state* —
  the silent-instrument failure mode, where absence-of-output is read as output-of-absence.
  **Structural cure (owed, sister landing):** never consume a background job's output without first
  asserting terminal status; a partial read must be typed as INCOMPLETE, never as a negative result.
  **The withdrawal's SUBSTANCE stands**: it is a **6-step `[macOS-MPS advisory]`** training-history
  row produced by the **dense adapter**, not the owed 4,000-step result, not byte-closed, not exact,
  not score authority — and its JSON omits torch version, optimizer kind, git SHA, and argv, so it
  cannot answer those questions retroactively.

### Round-1 findings that did NOT hold up (recorded, so they are not re-hunted)

- **A1 — "byte-identity proves nothing about Metal; `--device mps` may be inert."** REFUTED. Device
  is load-bearing at 8 sites in `codec_hpac_integer.py` (`:58` model, `:66/:105` context, `:70/:108`
  index, `:119` symbols, `:174` masks, `:206` tokens); the only `.cpu()` calls (`:29`, `:81`, `:122`)
  are necessary numpy marshalling for `constriction`, which is a CPU library by construction.
- **A3 — "`--width 96` was a script default, so the FLOP derivation is unprovenanced."** REFUTED for
  the architecture: strict `load_state_dict` succeeded against the real checkpoint, which pins
  width 96 / blocks 4 / frame_dim 8. **But it surfaced a live hazard:** the checkpoint's embedded
  `config` dict describes an ANCESTOR run — `steps: 3000`, `lr: 0.001`, `amp: True`, save path
  `…w96_b2_12x3000` — while the file is `…w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt` and the trainer
  contains no autocast/GradScaler at all. **Architecture fields survive; schedule/precision fields
  are stale.** Anything citing `ckpt["config"]` as provenance for the shipped weights reads the
  wrong run (#893 stale-fit genus, live in the intake).
- **A4 — "the 31.8% eval share may be 2× if distillation is on."** REFUTED. The semantic invocation
  passes no `--master-cache`, so `master_targets is None` and the second (`evaluate_rgb`) call site
  is skipped. 24 in-loop evals + 1 pre-loop `step: 0` baseline = 25. Share stands.

### Method note on my own A3b check

The grep I ran for `amp|autocast|GradScaler` returned three hits that were all `clamp`/`clamp_min` —
a substring false positive, the same class as #829. The correct reading required looking at the
matches, not the count. Recorded because the check that hunts a bug class should not exhibit it.

## The port-verification bar for `encode-tokens` — MET

Arithmetic coding is bit-exact by construction (`hpac_integer.py` keeps every accumulation inside
fp32's exact-integer range via an explicit constructor guard; `probability_table` quantizes logits
to int16 at 1/8 resolution). So a correct port has a **binary** test, not a tolerance:

```
tokens.bin  sha256 948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb
            size   116980
```
(the intake's own `verify_file` expectation in `scripts/train.sh`). Byte-identical ⇒ the Metal
path reproduces the encoder exactly. Any diff ⇒ a real device divergence in the integer path,
which would be a genuine finding, not a nuisance.

## OURS — the A/B column (not yet raced against OTS)

| ours | vs which OTS path | status |
|---|---|---|
| `pr130_lift/mlx_semantic_renderer.py` | `train_semantic_quantized.py` on MPS | UNRACED. OTS already runs and reproduces; MLX must WIN a measured row to earn the slot |
| `pr130_lift/pose/mlx_pose_carrier.py` | ported `train_pose_carrier_full.py` | UNRACED |
| `pr130_lift/pose/repack_race.py` | their repack | UNRACED |
| our fused Metal conv suite (#478) / megakernel (#356) | the PyTorch-MPS renderer step | UNRACED — and the throughput receipt says the renderer is 75.40% of a step at 5.5% of the device ceiling, with autocast and torch.compile BOTH measurably unavailable on MPS |

**Discipline this ledger enforces:** OURS does not displace OTS by assertion. PR130 is the full
stack; each of our pieces has to win an A/B against the OTS baseline on a measured row.

## What this changes about "we already ported stuff"

`src/tac/pr130_lift/` is **not** redundant work — but a chunk of it is now in the OURS column
rather than the PORT column, because today's measurements showed the OTS path runs on Metal
directly. The genuinely-required port is narrow (the pose leg's 2 device sites). The rest is
optimization to be **raced**, per the operator's A/B frame.

## Honesty bars

- Every ✅ above is a run I executed today on this host; every 🔄 is live; nothing is inferred.
- Metal figures are `[macOS-Metal advisory]`, `score_claim=false`. The BASE 0.172141 is
  `[contest-CUDA, DALI GT]` and is not reproducible locally — no CUDA here.
- The intake at `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` is READ-ONLY;
  every adapter lives in our tree.
