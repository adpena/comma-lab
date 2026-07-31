# ddm_ua2 — Upstream defenses + the 30-minute budget as a SURFACE

**Actor:** ddm_ua2 · **Date:** 2026-07-31 · **Mode:** READ-ONLY on `upstream/` (pinned, immutable; nothing
inside it was created, edited, moved, or deleted). $0, no scorer slot, no training, no heavy job.
**Axis:** `[macOS-CPU advisory]` for every local probe · `score_claim=false` · pointer **0.1910828242
[contest-CPU] UNMOVED**. This memo is MEANS, not a score-mover.

**Deliverable form (operator-binding):** no verdicts, no ranked lists. Every result is a **placed point on a
named surface** — surface, coordinates, the level set where the language flips, what moves that level set,
where we stand, which way it falls.

**Operator correction absorbed (verbatim, 07-31):** *"don't naively discount something though just because
our initial decoder takes too long there is always room for optimization often significantly so."* The
budget is **not** a kill criterion. A slow first decoder is an **engineering state**, not a property of the
approach. The finding is always the **required-speedup ratio against a demonstrated ladder**, never
"too slow."

---

## 0. DENOMINATOR (stated up front)

| Scope | Count |
|---|---:|
| Files in `upstream/` excluding `.venv/`, `.git/`, `__pycache__/` | **110** (MEASURED, `find`) |
| — of which are prior competitor submissions (`submissions/`) | 87 |
| — harness proper (everything else) | 23 |
| Harness files I read **in full or programmatically** | **21 / 23** |
| Not read | `LICENSE` (1.0 KB, non-technical), `ffmpeg-new` (24.5 MB stripped binary — not readable as text; sized only) |

The charter said "115 excluding .venv/.git"; my measured count is **110**. The difference is
`__pycache__/` exclusion (charter's 115 likely counted `.pyc` files). Stating the discrepancy rather than
adopting either number silently.

**Read in full:** `evaluate.py` (113 lines) · `evaluate.sh` (77) · `.github/workflows/eval.yml` (145) ·
`pr-welcome.yml` (47) · `update-leaderboard.yml` (45) · `pyproject.toml` (85) · `.gitattributes` (66) ·
`download_and_remux.sh` (79) · `.devcontainer/devcontainer.json` · `.python-version` · `uv.lock`
(119 packages, parsed programmatically) · `frame_utils.py:210-250` (direct) · `.gitignore` (232 lines,
pattern-class sampled — **not** line-by-line; flagged as partial).
**Consumed from ddm_us1, credited not re-derived:** `modules.py`, `frame_utils.py` (full), `README.md`,
`public_test_*.txt`, dependency-as-installed probes.

---

## 1. RECALL FIRST — what ddm_us1 (#811/#812) already settled

`ddm_us1_upstream_reread_20260731.md` ran the PREDICT-THEN-DIFF protocol over the scorer surface on
2026-07-31. **I did not re-derive any of it.** Settled there and treated as given here:

- Rate numerator = `archive.zip` stat only (`evaluate.py:63`); score formula `evaluate.py:92`; **no time
  term anywhere** in the score.
- **Rate denominator is a dynamic `rglob('*')` sum** (`evaluate.py:64`) — the #812 finding, landed as a
  fail-closed guard. Not re-opened here.
- Frozen-scorer factorization, `x[:,-1]` seg, first-6-of-12 pose, `seq_len=2` → 600 pairs, rule-118,
  MPS-default branch, `camera_fl=910`, yuv6 polyphase, scorer weight sizes (posenet 55,835,560 B >
  the 37,545,489 B video), inflate 3-arg contract, `.raw` format.
- `evaluate.sh` has **no in-script timeout** (us1 C1) — us1 correctly localized the 30-min limit to the
  external harness but did not decompose it. **That decomposition is this memo's primary object.**

**Also already recorded elsewhere, and I am crediting rather than claiming:** `zip(dl_gt, dl_comp)`
truncates to the shorter iterator — recorded at `ADVISORY_RESTART_HANDOFF_v752_v753_v8_20260710.md:125`
and already *weaponized* as the `--batch-size n` free-eval trick
(`sub015_DAG_…_20260611.md:6707`). My exhaustive grep for this specifically prevented a false novelty
claim; see §6.

---

## 2. SURFACE A — the 30-minute wall: WHERE it is attached

**Coordinates:** `upstream/.github/workflows/eval.yml:30` → `timeout-minutes: 30`, indented under
`jobs.test:` (`:28`), a sibling of `runs-on:` (`:29`) and `steps:` (`:35`).

**MEASURED (primary artifact):** the attribute is on the **job**, not on a step, and not on the decode.
All twelve steps (`:36`–`:98`) are inside that one wall.

**DERIVED (GitHub Actions semantics + in-file corroboration):** job-level `timeout-minutes` bounds the
whole job. The workflow corroborates its own reading at `:132-133` — a timeout surfaces as
`needs.test.result == 'cancelled'`, and the comment body is literally `'Job timed out or was cancelled'`.
The authors named the **job** as the thing that times out.

### The ordered step chain inside the wall (all receipts `eval.yml`)

| # | Step | Line | What it does |
|---:|---|---|---|
| 1 | `actions/checkout@v4` | :36-39 | `refs/pull/{N}/merge` or `master`; default depth-1; **no `lfs:` input ⇒ LFS not fetched here** |
| 2 | `check_name` | :41-48 | `git fetch origin master --depth=1` + duplicate-name refusal |
| 3 | `check_gpu` | :50-54 | `nvidia-smi` — **T4 axis only** (`if:` guard) |
| 4 | `download` | :56-60 | `curl -L -o …/archive.zip "${{ inputs.submission_url }}"` |
| 5 | `install_lfs` | :62-67 | `apt-get update` + `apt-get install -y git-lfs` + `git lfs install` |
| 6 | `pull_lfs` | :69-71 | `git lfs pull` |
| 7 | `install_uv` | :73-77 | `astral-sh/setup-uv@v4`, `enable-cache: true` |
| 8 | `install_deps` | :79-81 | `uv sync --group "$UV_GROUP"` |
| 9 | `install_ffmpeg` | :83-85 | `apt-get update` + `apt-get install -y ffmpeg` |
| 10 | `evaluate` | :87-89 | `uv run --group "$UV_GROUP" bash evaluate.sh …` → unzip + **inflate** + evaluate.py |
| 11 | `Upload artifacts` | :91-98 | `if: always()`; archive.zip + report.txt |

The `comment` job (`:100-144`) is a **separate job** on `ubuntu-latest` — outside the 30-minute wall.

### The doctrine drift this places

| Location | Text | Reading |
|---|---|---|
| `upstream/README.md:114` | "The **official evaluation** has a time limit of 30 minutes" | **CORRECT** — "evaluation", not "decode" |
| `CLAUDE.md:294` | "30-min **full-eval** budget" | **CORRECT** |
| `CLAUDE.md:302` | inflate.py "may run an arbitrarily sophisticated deterministic program … finishing within 30 min" | **CONFLATED** |
| `CLAUDE.md:929` | "the only constraint is the **30-min decode budget**" | **CONFLATED** — and this is the sentence licensing both the unbounded-free-interpreter doctrine and the dependency policy |

**The drift is systemic, not two lines.** A scan of `.omx/research/*.md` surfaced ~20 distinct memos
carrying "30-min inflate budget" / "30-minute decode budget" / "inflate.sh runs ≤ 30 min on T4" framings.
MAIN's finding is **verified and broader than stated**.

**Which way it falls:** toward the conservative side of every decode-cost decision we have already made —
i.e. our historical planning assumed *more* headroom than exists. No landed claim is invalidated (§4 shows
the shipped rungs still clear), but the margin is thinner than doctrine says.

---

## 3. SURFACE B — the budget decomposition, per axis

**The two axes are computed separately and neither is inferred from the other** (apples-to-apples
discipline). `ubuntu-latest` = 4 CPU / 16 GB RAM (`README.md:114`, us1 R3). `linux-nvidia-t4` = 26 GB RAM /
16 GB VRAM (`README.md:114`); **its vCPU count is not stated anywhere I read — UNKNOWN.**

### Payload terms I could measure exactly

| Term | Value | Label |
|---|---:|---|
| Depth-1 checkout pack (`.git/objects/pack`, snapshot is a genuine shallow clone: `.git/shallow` present, `git rev-list --count HEAD` = 1) | **31.6 MB** | **MEASURED** (lower bound for today's repo — snapshot pinned at HEAD `11ad728f`, 2026-04-13) |
| `git lfs pull` payload — exactly **5** objects (`git lfs ls-files`) | **132,856,531 B = 126.70 MiB** | **MEASURED** (posenet 55,835,560 + segnet 38,502,892 + 0.mkv 37,545,489 + 2 PNGs 972,590); independently confirmed by `du .git/lfs` = 126.7 MB |
| `uv sync --group cpu` sized closure | **~78 MB** + `torch 2.10.0+cpu` wheel | **MEASURED** from `uv.lock` `size` fields; torch-cpu wheel has **no size in the lock** (pytorch index) |
| `uv sync --group cu128` — torch 2.9.0+cu128's own NVIDIA dependency closure, x86_64 wheels | **3,190,398,780 B = 3.19 GB** | **MEASURED** from `uv.lock` `size` fields — **identical under pinned-HEAD and worktree `uv.lock` (§10)** |
| … plus `torch==2.9.0+cu128`, `torchvision`, `nvidia-dali-cuda120`, `nvidia-nvcomp-cu12`, `nvidia-nvimgcodec-cu12` | **unsized in the lock** | **UNMEASURED** |

The cu128 3.19 GB breaks down as cudnn 674.0 + cublas 566.8 + nccl 307.4 + cusparse 274.9 + cusparselt
273.9 + cusolver 255.1 + cufft 184.2 + triton 179.5 + nvshmem 132.7 + nvrtc 84.0 + curand 60.7 + nvjitlink
37.4 + cupti 9.8 + runtime 0.9 + cufile 1.1 + nvtx 0.1 MB.

> **This ~40× install-payload asymmetry between the two axes (≈0.08 GB sized vs ≈3.19 GB sized) is the
> single largest previously-unrecorded term in the budget.** Its mitigation is `enable-cache: true`
> (`eval.yml:77`) — but a cache **miss** (new lock hash, 7-day idle eviction, or the 10 GB per-repo cache
> ceiling) puts the full multi-GB download inside the 30-minute wall.

### The time table — CPU axis (`ubuntu-latest`, 4 vCPU)

| # | Term | Receipt | Typical | Worst | Label | **FIXED / OURS** |
|---:|---|---|---:|---:|---|---|
| 1 | checkout (31.6 MB) | `:36` | 10 s | 15 s | ESTIMATED from measured payload | harness-FIXED |
| 2 | `git fetch --depth=1` | `:44` | 4 s | 5 s | ESTIMATED | harness-FIXED |
| 3 | `curl` archive.zip | `:60` | 2 s | 3 s | ESTIMATED | **OURS** (bounded: the rate term forces the archive small — ~0.18 MB at the pointer) |
| 4 | apt update + git-lfs | `:64-66` | 32 s | 45 s | ESTIMATED | harness-FIXED |
| 5 | `git lfs pull` (126.7 MiB) | `:71` | 12 s | 20 s | ESTIMATED from measured payload | harness-FIXED |
| 6 | setup-uv cache restore | `:73-77` | 12 s | 20 s | ESTIMATED | partly OURS |
| 7 | `uv sync --group cpu` | `:81` | 20 s (hit) | 120 s (miss) | ESTIMATED | **PARTLY OURS** |
| 8 | apt update + ffmpeg | `:85` | 65 s | 90 s | ESTIMATED | harness-FIXED |
| 9 | `unzip -o` | `evaluate.sh:44` | <1 s | 1 s | **UNMEASURED** | trivial |
| 10 | **inflate.sh** | `evaluate.sh:47` | — | — | — | **HEAVILY OURS → §4** |
| 11 | `.raw` existence loop | `evaluate.sh:50-64` | <1 s | 1 s | ESTIMATED | FIXED |
| 12 | **`evaluate.py` 600 pairs** | `evaluate.sh:69` | **300 s** | **400 s** | **DERIVED** from 176.3 s MEASURED @ 8-core | **IMMOVABLE FLOOR** |
| 13 | upload-artifact | `:91-98` | 10 s | 15 s | ESTIMATED | FIXED |
| | **non-inflate total** | | **≈ 7.8 min** | **≈ 12.6 min** | | |
| | **T_residual(CPU) = 30 − total** | | **≈ 22.2 min** | **≈ 17.4 min** | | |

Term 12's basis is **MEASURED**: full 600-sample `upstream/evaluate.py --device cpu` = **176.3 s** on a
Modal Linux x86_64 **8-core** gVisor container
(`clickpolish_pr110_phase2_modal_runbook_20260710.md`), independently corroborated at **174.9 s**
(`pr106_latent_sidecar_dual_axis…`). The 8→4 core projection is **DERIVED**, not measured — the scorer
forward is thread-parallel but not perfectly linear, so I use a 1.7–2.3× band → 300–400 s. Older
macOS-CPU numbers (408–442 s) are a *different* harness/host and are **not** used as the projection basis.

### The time table — CUDA axis (`linux-nvidia-t4`)

| # | Term | Typical | Worst | Label | FIXED / OURS |
|---:|---|---:|---:|---|---|
| 1-5 | as CPU, + `nvidia-smi` (`:54`) | 60 s | 90 s | ESTIMATED | harness-FIXED |
| 6 | setup-uv restore of a **multi-GB** cache | 120 s | 180 s | ESTIMATED | partly OURS |
| 7 | `uv sync --group cu128` (**3.19 GB** measured + torch + DALI unsized) | 120 s (hit) | **420 s** (miss) | ESTIMATED | **PARTLY OURS** |
| 8 | apt + ffmpeg | 65 s | 90 s | ESTIMATED | harness-FIXED |
| 9,11,13 | unzip / existence / upload | 12 s | 17 s | ESTIMATED | FIXED |
| 12 | **`evaluate.py` on T4** (DALI decode + GPU forward) | 120 s | 180 s | **UNMEASURED BY US** | IMMOVABLE FLOOR |
| | **non-inflate total** | **≈ 8.3 min** | **≈ 16.3 min** | | |
| | **T_residual(CUDA)** | **≈ 21.7 min** | **≈ 13.7 min** | | |

**Self-hosted caveat:** `linux-nvidia-t4` is a custom runner label. If it is self-hosted with a persistent
workspace, apt and uv caches may persist across runs, collapsing terms 4–9 toward zero. **We cannot observe
this from here.** The CUDA worst case is therefore wider-tailed than the CPU one, not narrower.

### The level set

> **The compliance predicate is `t_inflate ≤ T_residual(axis, cache_state)`, with
> `T_residual(CPU) ∈ [17.4, 22.2] min` and `T_residual(CUDA) ∈ [13.7, 21.7] min` — not 30 on either axis.**

**What moves this level set:**
- **Downward (erodes our room):** a uv cache miss (largest single mover, CUDA axis, up to −5 min);
  leaderboard growth — every merged submission adds to the depth-1 checkout pack, and any merged
  LFS-tracked asset adds to the 126.7 MiB `git lfs pull` (a real precedent exists: `damir_bearclaw_003`
  contributed 972,590 B of PNGs). Two submissions vendored ~32.7 MB binaries each, though git blob-dedup
  absorbs the identical copies.
- **Upward (recovers room):** choosing the CPU axis when the CUDA install payload is the binding term;
  fewer/smaller declared dependencies (§5).
- **Orthogonal:** `t_inflate` itself — §4.

---

## 4. SURFACE C — `t_inflate` is OURS, and the ladder is MEASURED

This is the term the operator's correction is about. **Never "infeasible on time."** Always: *at this
coordinate it needs N×; here is the ladder; here is what has been demonstrated; here is the cheapest rung
that closes it.*

### The demonstrated ladder (all receipts in-tree)

| Rung | Task | Serial → optimized | **Multiplier** | Identity grade | **Available in the contest runtime?** |
|---|---|---|---:|---|---|
| R1 multiprocess `INFLATE_WORKERS` (M5, early ckpt) | #214 `db264bb2f` | 88 min → ~8 min | **10.8×** | **BIT-EXACT** | **YES** — stdlib `multiprocessing` + numpy |
| R2 numpy-fp64 4-worker (**explicit contest-4-core proxy**) | #214 FEED-05z | 48.9 → **13.9 min** | **3.52×** | **BIT-EXACT**, 2-run deterministic, `.raw` sha `358bd6eb…` == serial == 2nd run | **YES** — the honest 4-core number |
| R3 torch-fp32 CPU | #214 FEED-05z | 48.9 → **6.59 min** | **7.42×** | **SCORE-PRESERVING** (argmax 99.9995%, 3 flip px / 589,824; d_pose Δ3.2e-10; uint8 maxΔ1) — *not* bit-exact | **YES** — torch is in the contest venv for free (§5) |
| R4 torch-fp32 **T4** | #214 FEED-05z | 48.9 → **<0.5 min** | **~98×** | — | **CUDA axis only; PROJECTED, NOT RUN** (Modal `gpu="T4"` smoke staged, never fired) |
| R5 process-pool base decode | #592 | 111.62 s → 21.17 s (16 workers) | **5.27×** | **BYTE-IDENTICAL**, raw sha `6b550f16…` | **YES** (worker count); the SSD plane-cache is a *local dev* accelerator, **not** a contest-runtime artifact |
| R5b batched factor-2 scorer projection | #592 | 0.7729 → 0.3217 s | **2.40×** | byte-identical sha `5f4a6d1a…` | **YES** — pure numpy batching |
| R6 Rust inflate | #282/#283 | — | — | — | **available-UNBUILT** for decode; #214 deliverable (d) concluded "Rust NOT needed" at its coordinate |

**Superseded rung, recorded so it is not re-litigated:** at the earlier `db264bb2f` coordinate, fp32 was
**REJECTED** as not bit-exact (±12 LSB, 0.014% px). R3's later real-weight measurement re-admitted it under
the weaker-but-sufficient **score-preserving** grade. The two are not in conflict — they are different
identity grades at different coordinates.

### ⛔ The rung that must NOT enter a contest-axis budget

`#212 / #356 / #478` — the custom **Metal** kernel suite (metal-VJP 3.06× fwd+bwd, 5.12× bwd-only;
`TAC_MLX_CUSTOM_GROUPED_BACKWARD` 16.9×). **Apple Metal/MLX exists on neither contest runner** —
`ubuntu-latest` is Linux x86_64 CPU, `linux-nvidia-t4` is Linux + NVIDIA. These are **training-substrate**
multipliers. They are *doubly* inapplicable to decode: most are **backward-pass** speedups, and decode is
forward-only. Quoting any of them against `t_inflate` would be a false-authority leak.

### Where our current vehicle stands on this surface

Coordinates from #214 FEED-05z, n600, M5 Max 4-thread/4-worker as the contest-4-core proxy
(`[macOS-CPU advisory]`, non-promotable):

| Decode implementation | t_inflate | vs `T_residual(CPU)` **worst 17.4 min** | vs **typical 22.2 min** |
|---|---:|---|---|
| numpy-fp64 **serial** | 48.9 min | needs **2.81×** | needs **2.20×** |
| numpy-fp64 **4-worker** (bit-exact) | **13.9 min** | **INSIDE**, margin **1.25×** | **INSIDE**, margin 1.60× |
| **torch-fp32 CPU** (score-preserving) | **6.59 min** | **INSIDE**, margin **2.64×** | **INSIDE**, margin 3.37× |
| torch-fp32 T4 (projected) | <0.5 min | vs CUDA worst 13.7 min: margin ~27× | — |

**Read this the right way.** The serial decoder needs **2.81×** — and the ladder has *already shipped*
**3.52× bit-exact** and **7.42× score-preserving** at exactly this coordinate. The requirement sits
**below** what is built. Nothing here closes anything.

**What the 30 → 17.4 min correction actually costs us:** it does not push any shipped rung outside the
wall. It eats most of the **bit-exact** rung's margin (1.60× → **1.25×**), while the score-preserving rung
keeps 2.64×. That is the one actionable consequence, and it is a *measurement* consequence, not a design
one — see M1 below.

### The general rule to use from now on

For a candidate with measured serial decode `t_s`: **`N_req = t_s / T_residual`**, then place `N_req`
against the ladder:

| `N_req` | Placement |
|---|---|
| ≤ **3.5×** | closed by an **already-shipped bit-exact** rung (R2) |
| ≤ **7.4×** | closed by an **already-shipped** rung (R3) carrying a score-preservation proof obligation |
| ≤ **~98×** | plausible on the **CUDA axis** (R4) — but that multiplier is **PROJECTED, never measured**; buying it requires M3 below |
| **> ~98×** | needs unbuilt rungs — Rust (R6), or algorithmic restructuring |

**A candidate needing 3× when we have shipped 7.4× is at a completely different coordinate from one
needing 3,000×. The ratio is the finding.**

---

## 5. SURFACE D — the free dependency surface (the e4 / #666 coupling, as a TRADE)

**MEASURED (`eval.yml:89`):** `uv run --group "$UV_GROUP" bash evaluate.sh …`.
**DERIVED (uv semantics):** `uv run` prepends the project venv's `bin/` to `PATH` for the child. `evaluate.sh`
then invokes `bash inflate.sh` (`:47`) and `python evaluate.py` (`:69`) as children of that environment.

> **Therefore `inflate.py` executes inside the contest venv and may import, at zero install cost and zero
> archive bytes: `torch`, `torchvision`, `numpy`, `einops`, `timm`, `safetensors`,
> `segmentation-models-pytorch`, `tqdm`, `pillow`, `av`, `charset-normalizer`, `requests`, `urllib3`
> (`pyproject.toml:5-17,20-24`) — plus `nvidia-dali` on the CUDA groups.** `ffmpeg` is on `PATH`
> (`eval.yml:85`). This is what makes ladder rung R3 (torch-fp32 decode) free.

**The trade, stated as a trade and not a prohibition.** A *declared* extra dependency (the e4/#666
brotli decision) costs resolve+download+install time inside the wall and buys whatever bytes/score the
dependency earns. It is priced on this surface:

- **CPU axis:** the sized closure is ~78 MB. One additional small pure-python/C-extension wheel is **noise**
  against a 22-minute residual. The trade is essentially free.
- **CUDA axis:** the install term is already the **dominant** non-eval cost (3.19 GB measured). An extra
  dependency's *marginal* cost is small, but the *base* is what compresses the residual.
- **The always-cheaper alternative** (CLAUDE.md lesson 4, preference order): **vendor the OSS decoder
  source inside `inflate.py`** — zero install risk, zero resolve time, rule-118-clean.

**Which way it falls:** the budget does **not** argue against declaring a dependency. It argues that the
declared-dep path should be **priced** against the vendored-source path, and on the CUDA axis it argues for
attention to the *base* install, which is not ours to move at all.

---

## 6. SURFACE E — defenses vs unguarded surface, file by file

Two columns matter equally. **Every negative statement below carries its exact search scope.**

### `evaluate.py` (read in full, 113 lines)

| Guarded | Receipt |
|---|---|
| Per-batch frame geometry `[seq_len, 874, 1164, 3]` | `:77` assert |
| GT/compressed batch-shape equality | `:78` assert |
| Distortion output shape == batch size | `:80` assert |
| Eval runs under `torch.inference_mode()` | `:73` |
| Score components printed at `:.8f` (score itself only `:.2f`) | `:95-100` — recompute-from-components remains mandatory |
| **Sample count is printed** — `f"=== Evaluation results over {batch_sizes:.0f} samples ==="` | `:94` |

| Unguarded / what the absence permits | Receipt + scope |
|---|---|
| **No integrity check of any kind on `archive.zip`** — no hash, no size cap, no signature. `grep -niE "hashlib\|sha256\|sha1\|md5\|checksum\|digest\|shasum\|cksum\|blake"` over `evaluate.py`, `evaluate.sh`, `frame_utils.py`, `modules.py` returned **zero matches** (rc=1). The scored bytes are exactly whatever `curl` fetched. | EXHAUSTIVE over those 4 files |
| **`zip(dl_gt, dl_comp)` truncates to the shorter stream** (`:71`); `TensorVideoDataset` derives `N = file_size // frame_bytes` by integer division (`frame_utils.py:227`). A short `.raw` is scored on the surviving **prefix**, and `:78`'s equality assert can never fire because `zip` stops first. The **only** detection surface is the printed sample count at `:94`, which lands in `report.txt` → the PR comment (`eval.yml:129`). | **ALREADY RECORDED** — `ADVISORY_RESTART_HANDOFF_v752_v753_v8_20260710.md:125`; already exploited as the `--batch-size n` free-eval trick (`sub015_DAG…:6707`). Not my finding. |
| Rate is computed on rank 0 from a **dynamic** `rglob('*')` denominator | `:63-65` — **us1/#812, landed**, not re-derived |
| Device auto-detect falls through to **MPS** with no `--device` | `:21-28` — us1 E4 |

### `evaluate.sh` (read in full, 77 lines)

| Guarded | Receipt |
|---|---|
| `set -euo pipefail` — any failure aborts the whole chain | `:2` |
| `archive.zip` must exist | `:31-34` |
| `inflate.sh` must exist | `:36-39` |
| Extraction dir wiped fresh each run (`rm -rf` + `mkdir -p`) | `:42-43` |
| Every named video must yield `inflated/<base>.raw`, counted, non-zero ⇒ `exit 1` | `:50-64` |
| Unknown CLI arg ⇒ `exit 2` | `:18-21` |

| Unguarded | Receipt + scope |
|---|---|
| The `.raw` check is **existence only** — `[ ! -f "$RAW_PATH" ]` (`:55`). No size, content, mtime, or "did inflate.sh actually write it" test. **Scope: `evaluate.sh` lines 1-77 read in full; no size/content test on `.raw` appears in that file.** Size is only *implicitly* consumed downstream by the integer division above. | scoped |
| `unzip -o` (`:44`) overwrites without prompting. Info-ZIP `unzip` strips leading `/` and prunes `../` traversal by default — **INFERRED from documented Info-ZIP behavior; I did not test it, and testing it is not in scope for a read-only audit.** | labeled INFERRED |
| No in-script timeout | us1 C1 — the 30-min wall is entirely external (§2) |

### `.github/workflows/eval.yml` (read in full, 145 lines)

| Guarded / SOUND | Receipt |
|---|---|
| `workflow_dispatch` **only** (`:3-4`) — maintainer-triggered; nothing auto-runs on PR. `pr-welcome.yml:33` states this to submitters. | **SOUND**: no untrusted auto-execution |
| Duplicate `submission_name` refused against `origin/master`, with a literal `"baseline"` exemption for testing | `:41-48` |
| `runner` is a closed `choice` — only `ubuntu-latest` / `linux-nvidia-t4` | `:18-25` |
| `UV_GROUP`/`EVAL_DEVICE` derived from the runner, not user-supplied | `:32-33` — **SOUND**: device and dependency group cannot be spoofed by the submitter |
| `timeout-minutes: 30` on the job | `:30` — **SOUND**: this is the *right* place for it, because the PR's own `inflate.sh` runs with runner privileges (`:39` checks out `refs/pull/{N}/merge`). An arbitrary decode program is the contest's design; a job-level wall is the correct containment. |
| Artifacts + PR comment emitted `if: always()` | `:92`, `:102` — failures are visible, not silent |
| `permissions:` scoped to `pull-requests: write` on the comment job | `:104-105` |

| Unguarded | Receipt + scope |
|---|---|
| `submission_url` is an **arbitrary URL** with `curl -L` (follows redirects). **Scope: eval.yml read in full, 145 lines — no domain allowlist, no `--max-filesize`, no size cap, no hash input token appears.** The 30-min job wall is the only bound on a hostile/slow URL. | scoped |
| The `test` job declares **no `permissions:` block** (`:28-34`); only the `comment` job does (`:104`). Repo-default `GITHUB_TOKEN` permissions therefore apply to the job that executes submitter code. **Scope: `permissions:` occurs exactly once in the file, at `:104`.** | scoped |
| Both `apt-get update` steps (`:65`, `:85`) and `git lfs pull` (`:71`) sit **inside** the timed job — §3 | — |

### `pr-welcome.yml` (read in full, 47 lines) — **SOUND, and worth naming**

Uses `pull_request_target` (`:4`) — the privileged-context trigger — but performs **no checkout** and runs
only `actions/github-script` to comment and assign. **This is the correct-by-construction use of
`pull_request_target`:** the classic vulnerability is checking out and executing PR code under that
trigger, and this workflow does neither. Repo-scoped by `:10`.

### `update-leaderboard.yml` (read in full, 45 lines)

Daily cron (`:5`) → `wget` `comma.ai/leaderboard` → `pup` extract → `awk` splice between
`<!-- TABLE-START -->` / `<!-- TABLE-END -->` → commit+push README.md.

**This resolves ddm_us1's R7 as a puzzle.** us1 flagged the README leaderboard as a "STALE pre-HNeRV-wave
snapshot." It is not stale *upstream* — it is refreshed daily by this workflow. It is stale **because our
snapshot is pinned at HEAD `11ad728f` / 2026-04-13**. us1's operational conclusion (never cite it as
score-to-beat) is **unchanged and correct**; the mechanism is now named. Note the pinned HEAD's own commit
message is literally `ci: update leaderboard tables` — the snapshot was taken right after one of these runs.

**Unguarded (scope: the 45 lines of this file):** no verification that `wget` returned a leaderboard page or
that `pup` produced a non-empty table; an upstream markup change would splice an empty table into README.
No effect on scoring — README is not read by `evaluate.py`.

### `pyproject.toml` / `uv.lock`

| Guarded | Receipt |
|---|---|
| `requires-python = "~=3.11"` pinned against `.python-version` = `3.11` | `pyproject.toml:3` |
| `[tool.uv] conflicts` makes all 5 device groups **mutually exclusive** (10 pairwise rules) | `:27-38` — **SOUND**: cannot accidentally co-install cpu+cu128 |
| `[tool.uv.sources]` + `explicit = true` on all 5 indexes — torch/torchvision/dali resolve **only** from their pinned index, never PyPI | `:40-85` — **SOUND**: closes the "unpinned torch grabs the wrong CUDA wheel" trap that CLAUDE.md names as its own historical bug class |
| `uv.lock` fully resolves 119 packages | MEASURED |

**Note, not a defect:** the `mps` group (`:24`) has **no** `[tool.uv.sources]` override, so it resolves
`torch 2.10.0` from PyPI — the 873.2 MB CUDA-bundling manylinux wheel. Irrelevant to both contest axes
(neither uses `--group mps`), recorded for completeness.

### `.gitattributes` (read in full, 66 lines)

65 LFS filter rules. Load-bearing consequence: `*.zip` (`:34`), `*.raw` (`:40`), `*.safetensors` (`:26`),
`*.mkv` (`:59`) are LFS-tracked, so **any** archive or inflated raw committed to the repo would join the
`git lfs pull` payload inside the timed job. **MEASURED: it hasn't happened** — `git lfs ls-files` returns
exactly 5 objects and none is a submission archive. This bounds the growth model of §3: the LFS term grows
only when maintainers merge LFS-matching assets (precedent: 2 PNGs, 972,590 B).

### `.devcontainer/devcontainer.json`, `download_and_remux.sh`, `.gitignore`

- **devcontainer**: `postCreateCommand` installs git-lfs + ffmpeg + uv — the same three the CI installs.
  Confirms those are *not* assumed pre-present. Dev-only; not on the eval path.
- **download_and_remux.sh**: provenance of `videos/0.mkv` — `ffmpeg -f hevc -framerate 20 -r 20 -i <hevc>
  -c copy -metadata segment=…` (`:56`) from the comma2k19 segment named in `public_test_segments.txt`.
  Guarded: `set -euo pipefail` (`:2`); missing source ⇒ `exit 1` (`:50-53`). **Not on the eval path.**
- **.gitignore**: 232 lines, standard Python/tooling ignores. **Read by pattern class, not line-by-line —
  the one file in my scope I am not asserting exhaustively.** Nothing there affects scoring.

---

## 7. UNMEASURED TERMS → the cheapest measurement that closes each

| # | Unmeasured | Cheapest closure | Cost | Value |
|---|---|---|---|---|
| **M1** | `evaluate.py` on **4 vCPU** Linux (currently DERIVED 300–400 s from 176.3 s @8-core) | One Modal CPU container pinned to 4 cores (`cpu=4`, `torch.set_num_threads(4)`) running `upstream/evaluate.py --device cpu` on any existing byte-closed archive | ~$0.02–0.10, ~10 min | Removes ±100 s — the widest band in the CPU residual |
| **M2** | **Every CI setup term (1–8), both axes** | `workflow_dispatch` `eval.yml` **on our own fork** with `submission_name: baseline` — explicitly exempted from the uniqueness check at `eval.yml:45`. GitHub prints **per-step durations** in the Actions log. | **$0**, one dispatch | **Highest value/cost ratio in this memo.** Replaces ~8 ESTIMATED rows with MEASURED ones, and reveals real uv-cache hit/miss behavior |
| **M3** | The **~98× T4 multiplier** (R4) — projected, never run | Fire the Modal `gpu="T4"` smoke already **staged** by #214 deliverable (c) | ~$0.20 | Converts the entire CUDA-axis ladder from PROJECTED to MEASURED |
| **M4** | `unzip -o` term | `time unzip -o` on a real archive locally | $0, <1 min | Trivial; closes row 9 |
| **M5** | `linux-nvidia-t4` vCPU count; whether it is self-hosted with persistent caches | **Not observable from here.** M2 on a fork only exercises `ubuntu-latest`. | — | Named as a standing UNKNOWN, not guessed |

---

## 8. ROUND-1 ADVERSARIAL REVIEW OF MYSELF

**What I tried to refute, and what survived.**

1. **"Maybe `timeout-minutes` at job level really is per-step."** Refuted my own doubt: the workflow's own
   failure path (`:132-133`) maps the timeout to `needs.test.result == 'cancelled'` — a **job** result. The
   authors' own code reads it as a job wall. **SURVIVES.**
2. **"Maybe the 176.3 s → 4-core projection is wrong enough to change the conclusion."** Stress-tested: even
   at a pessimistic 400 s, `T_residual(CPU)` = 17.4 min and the shipped bit-exact 13.9-min rung still fits.
   The conclusion is **insensitive** to this projection over its plausible band. **SURVIVES**, and M1 closes
   it cheaply anyway.
3. **"Is the 3.19 GB real, or an artifact of my version-picking?"** Partly an artifact-risk: I selected
   max-version per NVIDIA package. I re-derived it the honest way — walking `torch==2.9.0+cu128`'s **own
   declared dependency list** and filtering to x86_64/cp311 wheels. That is where 3,190,213,124 B comes from.
   It **excludes** torch's own wheel and DALI (unsized in the lock), so it is a **lower bound**, not an
   inflated one. **SURVIVES as a floor.**
4. **"Did I invent the `zip()` truncation finding?"** Tried hard to refute my own novelty and **succeeded**:
   an exhaustive targeted grep found it recorded at `ADVISORY_RESTART_HANDOFF…:125` and already exploited as
   the `--batch-size n` trick. **My claim was withdrawn before it was made.** This is the
   negative-existence discipline paying for itself.
5. **"Am I smuggling a kill criterion back in?"** Checked every sentence about `t_inflate`. §4 states
   required-speedup ratios against demonstrated multipliers and nowhere says a representation is
   infeasible. The one place the correction bites hardest — the bit-exact rung's margin dropping
   1.60× → 1.25× — is framed as a *measurement* consequence (buy M1), not a design verdict. **SURVIVES.**
6. **The ESTIMATED rows are the weak part of this memo.** Eight of thirteen CPU rows are ESTIMATED, and
   I have no receipt for GHA apt/network throughput. I did **not** dress them as measurements, and M2
   converts nearly all of them for $0. Anyone consuming §3 should treat the *structure* (which terms exist,
   which are FIXED vs OURS, the measured payloads) as the durable content, and the *seconds* as
   provisional.
7. **Axis purity.** I computed the two axes independently and never inferred one from the other. The Metal
   rungs are explicitly fenced out of both. The `[macOS-CPU advisory]` provenance of the entire #214 ladder
   is stated at its table rather than laundered — those multipliers are *ratios* measured on a 4-thread Mac
   proxy, and a real Linux 4-core ratio could differ.

---

## 9. WHAT THIS FEEDS

- **MAIN's finding** — verified at the primary artifact, and **broadened**: the conflation is in ~20
  research memos, not only `CLAUDE.md:302` / `:929`. The doctrinally correct phrasing is
  *"a 30-minute **full-evaluation** budget, of which ~17–22 min is the residual available to inflate."*
- **e4 / #666 declared-dep decision** — priced as a **trade** on §5, not gated. CPU axis: noise. CUDA axis:
  the base install already dominates. Vendored-source remains the cheaper path on both.
- **#214 / #592 decode ladder** — the multipliers now have a *denominator* (`T_residual`) instead of a
  mythical 30 min. R2's margin is thinner than believed (1.25× at worst case), which is the argument for M1.
- **Byte-close / export chain** — `archive.zip` receives **no integrity check anywhere in the harness**
  (exhaustively verified over 4 files): our own sha256 custody is the *only* thing standing between a
  corrupted upload and a scored row.
- **ddm_us1 R7** — mechanism named (`update-leaderboard.yml` daily cron; staleness is our pin, not
  upstream's). Operational conclusion unchanged.
- **ddm_ua1 (weights)** — untouched by design; no duplication.

---

## 10. INCIDENTAL FINDING — the pinned snapshot has PRE-EXISTING drift from its own HEAD

Found while verifying that my read-only audit left `upstream/` untouched. **REPORTED, NOT ACTED ON**, per
the charter: a finding requiring an upstream touch is reported, never acted on. **I changed nothing.**

**Evidence that this predates my session (MEASURED):**
- `upstream/` HEAD = `11ad728f`, snapshot materialized **2026-04-13 16:50**.
- `git -C upstream status --porcelain` reports **36 modified files**.
- mtimes of the modified scripts (`evaluate.py`, `evaluate.sh`, `modules.py`, `frame_utils.py`,
  `download_and_remux.sh`) are all **2026-04-13 16:50** — 3.5 months before this session (now 2026-07-31 17:55).
- `uv.lock` mtime is **2026-07-26 17:33** — **5 days before this session**, not mine.
- My entire toolset on `upstream/` was read-only: `cat`, `ls`, `find`, `stat`, `du`, `grep`, `sed -n`,
  `git status/diff/show/log/rev-list/lfs ls-files`, and Python heredocs that only *opened files for reading*.

**Two distinct drift classes:**

| Class | Files | Nature | Content-identical? |
|---|---|---|---|
| **A — mode-only, benign** | 35 files (`evaluate.py`, `evaluate.sh`, `modules.py`, `frame_utils.py`, submission scripts, `ffmpeg-new`) | `git diff --stat` = **0 insertions, 0 deletions** — permission-bit drift from snapshot materialization; plus one dropped symlink (`submissions/av1_roi_lanczos_unsharp/lib/libSvtAv1Enc.so.2`, ` D`) | **YES** — verified: `git show HEAD:evaluate.py` sha256 `7da71a84ce24…` == worktree `evaluate.py` sha256 `7da71a84ce24…` |
| **B — REAL content drift** | **`uv.lock` only** | **+296 / −194 lines**; **116 packages at HEAD → 119 in worktree** (3 packages added) | **NO** |

**Why class B matters and why it did not change my numbers.** I originally computed §3's CUDA payload from
the **worktree** `uv.lock`. On finding the drift I recomputed against the **pinned-HEAD blob**
(`git show HEAD:uv.lock`, extracted read-only to scratchpad — nothing written into `upstream/`):

```
HEAD (pinned authority): 23 deps, SIZED SUBTOTAL = 3,190,398,780 B = 3.19 GB, 116 pkgs in lock
WORKTREE               : 23 deps, SIZED SUBTOTAL = 3,190,398,780 B = 3.19 GB, 119 pkgs in lock
```

**Byte-for-byte identical**, and the resolved torch version set is identical in both
(`2.10.0`, `2.10.0+cpu`, `2.9.0+cu126`, `2.9.0+cu128`, `2.9.0+cu130`). **§3's headline CUDA number is
robust to this drift.** All figures in this memo now cite the HEAD-authority computation.

**The part that is genuinely worth operator attention:** `git status --porcelain upstream/` **from the
parent repo returns empty** — `upstream/` is a nested git repo, so the parent tracks none of its contents.
**This drift is invisible to every parent-repo gate and to `preflight_all()`.** A future agent re-deriving
dependency facts from `upstream/uv.lock` would silently read the 07-26 mutated file rather than the pinned
authority. That is a staleness-confound surface of exactly the named class
(`staleness_is_a_named_confound_class_freshness_at_consumption`).

**Cheapest closure (operator decision, not mine to take):** a read-only integrity check —
`git -C upstream diff --stat` — wired as a warn-only preflight row that fires when any `upstream/` file
shows **non-zero** content churn against its pinned HEAD. Mode-only drift (class A) would be filtered out
by the `0 insertions, 0 deletions` test, so the check would have a live count of exactly **1** today
(`uv.lock`) rather than 36. **No upstream mutation is required to implement it, and none is proposed.**

---

**Pointer 0.1910828242 [contest-CPU] UNMOVED.** This memo moved no score and claims none.
