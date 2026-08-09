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
| `carrier` (pose, 4k int6) | `train_pose_carrier_full.py` | **PORT** | 🔧 2 device sites: `torch.cuda.empty_cache()` :264 · `load_file(..., device='mps')`. Adapter `pr130_lift/pose/mps_port.py`. Sparse `nn.Embedding`+COO+`RowLocalSparseAdam` VERIFIED on MPS @ pinned torch 2.10.0, zero cpu-fallback, row clocks preserved. ✅ 600 pairs, mean d_pose 2.4437744286842644e-05 |
| `hpac-init` | `extract_integer_hpac_archive.py` | **OTS** | ✅ ran |
| `hpac` (60ep self-compress) | `train_hpac_self_compress.py` | **OTS** | ✅ ran, 59.18 s/epoch |
| `pack-hpac` | `pack_hpac_self_compress.py` | **OTS** | ✅ (hb2 fixed the round-trip upstream) |
| `encode-tokens` | `codec_hpac_integer.py` | **OTS + dep** | ✅ **BYTE-IDENTICAL on Metal.** 600 frames, 1,057 s, rc=0. `tokens.bin` = **116,980 B**, sha256 `948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb` — exact match to the intake's own `verify_file` expectation. Needed `constriction==0.5.0`; **no code change** |
| archive assembly | `build_submission_archive.py` · `rebuild_submission_hpac.py` · `compress.sh` | **OTS** | ✅ **BYTE-IDENTICAL** 191,052 B, sha matches |
| verification | `scripts/verify.sh` | **OTS** | ✅ 24 passed, "CPR1 repository verification passed" |

**Only ONE stage in the whole chain needed a real port: the pose carrier.** Everything else is
off-the-shelf with `--device mps`, plus one pip install.

## PORT CLOSURE — every stage of the PR130 chain now runs on this host

With `encode-tokens` byte-identical, **all seven stages plus archive assembly and verification are
green on Metal.** The chain that produced S = 0.172141297491896447 is reproducible here end-to-end,
and the only genuine port in it was the pose leg's two device sites.

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
