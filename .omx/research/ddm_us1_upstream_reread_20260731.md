# ddm_us1 — Upstream snapshot + dependency full fresh-eyes re-read (task #811)

**Actor:** ddm_us1 · **Date:** 2026-07-31 · **Mode:** READ-ONLY on `upstream/` (pinned snapshot immutable).
**Operator directive (verbatim 07-31):** *"look at upstream again and all of its dependencies because
there's signal there that you've forgotten since you're just recalling from working memory."*
**Method:** PREDICT-THEN-DIFF per surface — record recorded-law prediction, read the artifact line-by-line,
classify every material fact `{MATCHES-RECALL | DRIFTED | FORGOTTEN | NEW-SIGNAL}` with file:line receipts.
**Pointer honesty:** 0.1910828242 [contest-CPU] UNMOVED. This read is MEANS, not a score-mover.

---

## 0. HEADLINE — drift scorecard

**Working memory did NOT drift on the load-bearing physics. Zero CRITICAL law-basis failures.**
Every one of the 13 canonical cross-check laws' upstream bases was re-derived from the primary artifact and
HOLDS. The drift is entirely in the *operational envelope* (dynamic-vs-constant mechanisms, hardware specs,
version pins) and in *precision detail our live doctrine flattened*, never in the core scorer geometry.

| Class | Count | One-line meaning |
|---|---:|---|
| **MATCHES-RECALL** | 13 core laws + formula + full architecture | recall faithful; bases re-derived from primary artifact |
| **DRIFTED** (recall wrong in a way that matters) | 2 | rate-denominator-as-constant; T4 = VRAM-only |
| **FORGOTTEN** (recorded somewhere, absent from live doctrine) | 4 | dynamic rate denom · camera_fl=910 · MPS default branch · DALI-pad/AV-drop |
| **NEW-SIGNAL** (never recorded / not in live doctrine) | 11 | scorer sizes · timm drift · yuv6 polyphase · AllNorm/gelu_tanh · baseline anchor · stale leaderboard · inflate contract · … |
| **CRITICAL law-basis failures** | **0** | — |

**Verdict [magnitude-ok]:** low drift on the science, moderate forgotten/new on the operational surface.
The primary artifacts confirm the campaign's frozen-scorer factorization spine is honestly grounded.

---

## 1. CRITICAL law-basis failures

**NONE.** All 13 cross-check laws verified below (§4). This absence IS a finding: the recalled scorer
geometry (frozen factorization, blind coordinates, rank-4 head, seq_len=2, rate formula, rule-118) is
faithful to `upstream/` as pinned. Do not re-open any of them on suspicion of drift.

---

## 2. Environment / venv custody (shared-venv hijack law, verified BEFORE probes)

- Contest-replica venv = **`upstream/.venv`** → `/opt/homebrew/opt/python@3.11` (py **3.11.15**), matches
  `upstream/.python-version` (3.11) + `upstream/pyproject.toml:3` (`requires-python = "~=3.11"`).
- Lab venv = **`.venv`** → uv cpython-**3.13**; `tac.__file__ = src/tac/__init__.py` (NO worktree hijack).
- **As-installed** module `__file__`s all resolve inside `upstream/.venv/lib/python3.11/site-packages/`.
- Canonical pins (`upstream/uv.lock`) — replica HONORS them; lab venv DRIFTS from them:

| module | replica `upstream/.venv` (=uv.lock) | lab `.venv` | drift |
|---|---|---|---|
| torch | 2.10.0 | 2.12.1 | yes |
| torchvision | 0.25.0 | — | — |
| timm | **1.0.22** | **1.0.27** | yes (parity-probed §3) |
| segmentation-models-pytorch | 0.5.0 | 0.5.0 | no |
| numpy | 2.3.4 | 1.26.4 | **major (2.x vs 1.x)** |
| av | 17.0.0 | 17.1.0 | minor |
| einops | 0.8.1 | — | — |
| safetensors | 0.6.2 | — | — |

---

## 3. The 4-class signal ledger (every row with file:line receipts)

### 3A. `upstream/evaluate.py`

| # | Fact (file:line) | Class | Note / consumer |
|---|---|---|---|
| E1 | `compressed_size = (submission_dir/'archive.zip').stat().st_size` (`evaluate.py:63`) | MATCHES | rate numerator = archive.zip stat ONLY. Reconfirms rate law. |
| E2 | `score = 100*segnet_dist + math.sqrt(posenet_dist*10) + 25*rate` (`evaluate.py:92`) | MATCHES | exact score formula. **No time term anywhere** (grep: 0 timing refs). |
| E3 | **`uncompressed_size = sum(f.stat().st_size for f in uncompressed_dir.rglob('*') if f.is_file())`** (`evaluate.py:64`); `rate = compressed_size/uncompressed_size` (`:65`) | **FORGOTTEN** | rate DENOMINATOR is a **dynamic rglob sum**, NOT the hardcoded `37_545_489`. Measured `sum(rglob) == stat(videos/0.mkv) == 37,545,489` today (match=True). A stray file under `videos/` (`.DS_Store`, AppleDouble `._*`) would inflate the denom → silently LOWER rate. Consumer: byte-close/export + rate coder stack. |
| E4 | device auto-detect: cuda → **elif `torch.backends.mps.is_available()` → `mps`** → else cpu (`evaluate.py:21-28`) | **FORGOTTEN** | On a Mac with no `--device`, upstream DEFAULTS TO MPS (the noise axis). Must pass `--device cpu`. Basis of the MPS-never law made concrete at the default. |
| E5 | cuda → `DaliVideoDataset`; else → `AVVideoDataset` (`evaluate.py:39,42`) | MATCHES | GT decode class forks by device — the DALI(NVDEC)/PyAV CUDA-vs-CPU drift-hypothesis basis. Compressed side always `TensorVideoDataset` (`:67`). |
| E6 | `compute_distortion` returns `(posenet_dist, segnet_dist)` — POSENET FIRST (`evaluate.py:79`) | MATCHES | assignment order law (Catalog #222). |
| E7 | assert comp shape `[seq_len, camera_size[1], camera_size[0], 3]` = (B,2,874,1164,3) NHWC (`evaluate.py:77`) | MATCHES | frame layout. |
| E8 | `seed` default 1234 → passed ONLY to dataset ctors (`:15,58,67`); no `torch.manual_seed` / `use_deterministic_algorithms` | NEW-SIGNAL | eval determinism = argmax+MSE under `inference_mode` (`:73`); seed is effectively inert on CPU/AV/Tensor decode. |
| E9 | multi-GPU: `all_reduce(SUM)` over ranks (`:84-87`); score printed `:.2f` but components `:.8f` (`:95-100`) | MATCHES | recompute-from-components law (`:.2f` rounds → lies). |

### 3B. `upstream/modules.py`

| # | Fact (file:line) | Class | Note / consumer |
|---|---|---|---|
| M1 | SegNet = `smp.Unet('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)` (`modules.py:105`) | MATCHES | exact. |
| M2 | SegNet input = LAST frame `x[:, -1, ...]` then `interpolate(size=(384,512), mode='bilinear')` (`:108-109`) | MATCHES | frame_0 seg-free + resize-to-(512,384). No antialias/align_corners. |
| M3 | SegNet distortion = `(argmax != argmax).float().mean(...)` (`:112-113`), comment `# accuracy` | MATCHES | argmax disagreement on last frame; note label is "accuracy" but it's the error/disagreement rate. |
| M4 | PoseNet = `timm.create_model('fastvit_t12', pretrained=False, num_classes=2048, in_chans=12, act_layer=gelu_tanh)` (`:66`, consts `:22-25`) | MATCHES | VISION_FEATURES=2048, IN_CHANS=12. |
| M5 | PoseNet preprocess: `interpolate(size=(384,512), 'bilinear')` THEN `rgb_to_yuv6` → `b (t c)` c=6 t=2 → 12 ch (`:73-74`) | MATCHES | **BOTH scorers resize to the SAME `segnet_model_input_size`** → the A_seg≡A_pose factorization basis HOLDS. |
| M6 | PoseNet normalize: `_mean=255/2=127.5`, `_std=255/4=63.75` over IN_CHANS=12, applied `(x-_mean)/_std` (`:64-65,77`) | MATCHES | mean/std applied uniformly to all 12 yuv6 channels (chroma centered ~128 → ~0). |
| M7 | Head = `Head('pose', hidden=32, out=12)` (`:26`); distortion uses `[..., : h.out//2]` = first 6 (`:84`) | MATCHES | d_pose = MSE on first 6 of 12. |
| M8 | Hydra topology: ResBlock(512) → per-head in_layer `Linear(512,32)`+relu → res_layer `32→32→32`+relu → final `Linear(32,12)` (`:51-58`) | NEW-SIGNAL | exact PoseNet head graph (richer than recalled "12-dim pose"). |
| M9 | `AllNorm` = `BatchNorm1d(1)` on `x.view(-1,1)` — GLOBAL-SCALAR BN; `BN_EPS=0.001`, `BN_MOM=0.01` (`:20-21,28-33`) | NEW-SIGNAL | non-standard norm in ResBlock/Hydra; relevant to any PoseNet torch-parity/gradient work. |
| M10 | `act_layer='gelu_tanh'` override on FastViT (`:25,66`) → 19× GELUTanh modules (probe) | NEW-SIGNAL | tanh-approx GELU; a real parity surface for torch vs MLX pose. |
| M11 | `DistortionNet.forward` `# TODO run in bfloat16?` (`:152`); `compute_distortion` is `@torch.inference_mode()` (`:154`) | NEW-SIGNAL | scorer runs fp32, no autocast; gradients severed in eval (training must re-derive). |
| M12 | `preprocess_input` takes NHWC `b t h w c` → `b t c h w`, `.float()`; both scorers read same x (`:143-148`) | MATCHES | shared input; segnet slices last frame, posenet both. |

### 3C. `upstream/frame_utils.py`

| # | Fact (file:line) | Class | Note / consumer |
|---|---|---|---|
| F1 | `seq_len = 2`, `camera_size = (1164, 874)` (W,H), `segnet_model_input_size = (512, 384)` (`frame_utils:10,11,13`) | MATCHES | 600 pairs · camera 1164×874 · scorer grid 512×384. |
| F2 | **`camera_fl = 910.`** (`frame_utils:12`) | **FORGOTTEN** | camera focal length — never in live pose doctrine. Consumer: terminal pose solve (se(3) ego-screw geometry). |
| F3 | `rgb_to_yuv6` is `@torch.no_grad()` (`:50`) + in-place `.clamp_(...)` (`:61-63`) | MATCHES | the grad-severing yuv6 (PR95/106 monkey-patched it). |
| F4 | yuv6 layout = `[y00,y10,y01,y11,U_sub,V_sub]` — luma is a **2×2 POLYPHASE** of Y (4 half-res phases), chroma 2×2-averaged (`:65-78`) | NEW-SIGNAL | NOT 4 identical luma. The exact structure any luma/chroma carrier writes into. Consumer: terminal pose solve + chroma d_seg lever. |
| F5 | yuv6 crops to even H,W first (`rgb[..., :2*H2, :2*W2]`, `:53-54`) | NEW-SIGNAL | drops last row/col for odd dims; (384,512) even → no-op. |
| F6 | BT.601 consts kYR/kYG/kYB=0.299/0.587/0.114; U=(B-Y)/1.772+128; V=(R-Y)/1.402+128 (`:60-63`) | MATCHES | chroma <2px invisible = 2×2 subsample (`:65-72`). |
| F7 | **`yuv420_to_rgb`** GT decode = BT.601 **LIMITED** range + **bilinear** chroma upsample `align_corners=False`; `(y-16)*255/219`, `(u-128)*255/224`, r=yf+1.402·vf, g=yf−0.344136·uf−0.714136·vf, b=yf+1.772·uf, `.round().to(uint8)` (`:159-183`) | MATCHES | the phantom-pose law's exact basis (PyAV rgb24 ≠ this). The precise coefficients (16/219/128/224/1.402/0.344136/0.714136/1.772) are the DECODE AUTHORITY. |
| F8 | seq_len batching: DALI `num_sequences = frames//seq_len`, `last_sequence_policy="pad"` (`:126,138`); AV/Tensor build complete pairs only, DROP trailing single (`:203-204,240-241`) | **FORGOTTEN** | DALI PADS a partial trailing sequence; AV/Tensor DROP it → a CUDA-vs-CPU count divergence for ODD-frame files. 0.mkv even (1200) → no effect, but a latent dual-axis micro-source. |
| F9 | `TensorVideoDataset` reads `.raw` uint8 memmap `(N,H,W,3)`=(N,874,1164,3), `frame_bytes = 874*1164*3 = 3,052,008` (`:218-231`) | MATCHES | compressed/inflated frame format consumed by eval `:67`. |
| F10 | file sharding `all_file_names[rank::world_size]` (`:93`); `prepare_data` asserts files exist (`:107`) | NEW-SIGNAL | multi-rank distributes files; single video → rank0 only. |

### 3D. `upstream/README.md` (rules)

| # | Fact (file:line) | Class | Note / consumer |
|---|---|---|---|
| R1 | `./videos/0.mkv` = "1 minute 37.5 MB dashcam video" (`README.md:16`) | MATCHES | 37.5MB = 37,545,489 = rate denom = the single GT video. |
| R2 | score = `100*seg + 25*rate + √(10*pose)` (`:25`) | MATCHES | same terms (order commutes vs code). |
| R3 | **T4 GPU instance: RAM 26GB / VRAM 16GB; else CPU: 4 CPU / 16GB RAM; 30-min limit** (`:114`) | **DRIFTED** | recall said "T4 16GB-VRAM"; actual T4 host also has **26GB system RAM**. 30-min budget + CPU 4×/16GB MATCH. |
| R4 | rule-118: external libs/tools free UNLESS large artifacts (NN/mesh/pointcloud) → those count in archive. **"This applies to the PoseNet and SegNet."** (`:118`) | MATCHES | the free-code/counted-data boundary, with the scorers named explicitly as the canonical counted example. |
| R5 | "You can use anything for compression, including the models, original uncompressed video…" (`:119`) | MATCHES | scorers + original video legal at COMPRESS time (not inflate). |
| R6 | reference `baseline_fast` report: pose **0.38042614**, seg **0.00946623**, size 2,244,900, rate **0.05979147**, **score 4.39** (`:87-93`) | NEW-SIGNAL | concrete known-good eval fixture (recomputed: 4.3919 ✓). Byte-close/export sanity anchor. |
| R7 | mirrored leaderboard tops at **mask2mask 0.60 (#53)**, then neural_inflate 1.89 … no_compress 25.0 (`:143-457`) | NEW-SIGNAL | **STALE pre-HNeRV-wave snapshot** — must NOT be cited as score-to-beat (live official ~0.172). |
| R8 | ranking = public leaderboard, "no private testing" (`:121`); submit-by May 3 2026 (`:31`, PAST → contest closed) | NEW-SIGNAL | governance context. |
| R9 | `test_videos.zip` = 2.4GB, 64 comma2k19 videos for generalization testing (`:472`) | NEW-SIGNAL | cross-video set exists; low leverage now (we optimize single 0.mkv). |

### 3E. Contract files — `evaluate.sh`, `inflate.sh`, test-name files

| # | Fact (file:line) | Class | Note / consumer |
|---|---|---|---|
| C1 | inflate contract: `inflate.sh <archive_dir> <inflated_dir> <video_names_file>`; unzip→`archive/`; expect `inflated/<base>.raw` (`evaluate.sh:44-59`) | NEW-SIGNAL | exact 3-arg export contract. `evaluate.sh` has **NO in-script timeout** (30-min is external harness). |
| C2 | `.raw` = "flat binary dump of uint8 RGB frames (N,H,W,3), H/W match original, no header" (`baseline_fast/inflate.sh:2-4`); inflate.py invoked `python -m submissions.<name>.inflate SRC DST` (`:27`) | NEW-SIGNAL | authoritative raw format for byte-close/export. |
| C3 | `public_test_video_names.txt` = `"0.mkv\n"` (6 bytes); `format='raw'` → decode target `0.raw` (`frame_utils:221`) | MATCHES | ONE video → 1200 frames → 600 non-overlapping pairs = the "600 samples". |
| C4 | `public_test_segments.txt` = `b0c9d2329ad1606b\|2018-07-27--06-03-57/10/video.hevc` | NEW-SIGNAL | source comma2k19 segment identity. |

### 3F. Dependencies AS-INSTALLED (probes, `[macOS-CPU advisory]`)

| # | Fact (probe) | Class | Note / consumer |
|---|---|---|---|
| D1 | SegNet stem = `model.conv_stem` **stride (2,2)** k(3,3) 3→32 (probe) | MATCHES | "stride-2 stem loses half res immediately". |
| D2 | SegNet encoder `out_channels=[3,16,24,48,120,352]` (efficientnet_b2, TimmUniversalEncoder); SegNet ≈ **9,543,831 params** | NEW-SIGNAL | precise channel ladder + param count. |
| D3 | SegNet `segmentation_head` = `[Conv2d(16→5, k=3, pad=1), Identity, Activation(=Identity)]` (probe) | MATCHES | **rank-4 linear head**: single linear conv 16→5, NO nonlinearity (activation=None) → argmax invariant to +const → 4 DOF. flip-distance `d=|m|/‖Δw‖` basis HOLDS. Decoder final width = 16. |
| D4 | FastViT stem `stem.0.conv_kxk.0.conv` stride (2,2) 12→64; **12× RepMixerBlock, 0× AttentionBlock**; 4 FastVitStage, 19× GELUTanh, 1 SEModule; vision ≈ **8,633,664 params** | MATCHES | "FastViT-T12 is RepMixer/convolutional" — the MPS-drift-is-NOT-attention law's exact basis (zero attention). |
| D5 | frozen scorer sizes: `segnet.safetensors` = **38,502,892 B**, `posenet.safetensors` = **55,835,560 B** | NEW-SIGNAL | **posenet weights (55.8MB) > the entire 37.5MB video.** If shipped: rate += 25·38.5M/37.5M=**25.64** (seg) + **37.18** (pose) = **+62.8** score. Quantifies rule-118 stakes. Consumer: #809 cg1 guard ledger. |
| D6 | strict state_dict load of BOTH scorers under timm **1.0.22 AND 1.0.27**: `missing=0 unexpected=0` for each | NEW-SIGNAL (parity risk → MEASURED-benign) | module keys identical across the timm gap; numeric-forward parity across versions still technically owed but authority = `upstream/.venv`. Not CRITICAL. |

---

## 4. Cross-check list — every canonical law's upstream basis re-derived

| Law (recalled) | Basis re-derived from primary artifact | Verdict |
|---|---|---|
| frozen-scorer factorization A_seg≡A_pose→(512,384) | both `preprocess_input` resize to `segnet_model_input_size=(512,384)` bilinear (`modules.py:73,109`) | **HOLDS** |
| frame_0 structurally seg-free | SegNet reads `x[:,-1,...]` only (`modules.py:108`) | **HOLDS** |
| chroma <2px invisible | yuv6 chroma 2×2 average-subsampled (`frame_utils:65-72`) | **HOLDS** |
| ker(A) 80.67% resize nullity | 1−196608/1017336 = 0.80674 (512·384 / 1164·874) | **HOLDS** |
| rank-4 linear head, flip d=\|m\|/‖Δw‖ | seg head = 1 linear conv 16→5, activation=None (probe D3) | **HOLDS** |
| blind coords 230,904 (22.70%) px/frame | 106 rows·1164 + 140 cols·874 − 106·140 = 230,904 = 22.70%; distinct from the 80.67% rank-nullity | **HOLDS** |
| ERF r50~85px | consistent with stride-2 stem + RepMixer depth; NOT re-measured (needs forward pass; out of scope no-scorer-jobs) | consistent, not re-measured |
| SegNet class order [Road,Lane,Undrivable,Movable,MyCar] | property of trained argmax on data, not architecture; NOT re-measured (out of scope) | consistent, not re-measured |
| d_seg = argmax disagreement LAST frame | `(argmax!=argmax).float().mean` on `x[:,-1]` (`modules.py:108,112`) | **HOLDS** |
| d_pose = MSE first 6 of 12 | `[..., :h.out//2]`, h.out=12 (`modules.py:26,84`) | **HOLDS** |
| rate = archive.zip stat ONLY, no time term | `:63` numerator; `:92` no time; grep 0 timing refs | **HOLDS** (denom dynamic — E3) |
| rule-118 free-code / counted-data | `README.md:118` verbatim + "applies to PoseNet and SegNet" | **HOLDS** |
| seq_len=2 non-overlapping = 600 pairs | `seq_len=2` (`frame_utils:10`), `num_sequences=frames//2`, 0.mkv 1200 frames | **HOLDS** |

11/13 re-derived directly this session; 2 (ERF 85px, class order) are data/forward-pass properties consistent
with the re-read architecture but out of scope for a no-scorer-jobs read. **None failed.**

---

## 5. FORGOTTEN / NEW-SIGNAL ranked by live-consumer leverage

| Rank | Signal | Class | Live consumer | Leverage |
|---:|---|---|---|---|
| 1 | Scorer weight sizes: posenet 55.8MB > video 37.5MB; ship-both = **+62.8 score** (D5) | NEW | **#809 cg1 guard ledger** (rule-118 no-scorer-in-archive) | HIGH — quantifies the guard's stakes; a payload-cleanliness audit anchor |
| 2 | Dynamic rate denominator = `sum(rglob '*')` = `stat(0.mkv)` (E3) | FORGOTTEN | byte-close/export + rate coder stack | HIGH — a stray `videos/` file silently changes rate; export must assert `videos/` clean before any rate claim |
| 3 | Lab timm 1.0.27 ≠ replica 1.0.22 (keys benign D6, numeric-forward parity owed) | NEW | burn-4 endpoint · byte-close · torch-parity | HIGH — any lab-built PoseNet is a staleness-confound risk; authority stays `upstream/.venv` |
| 4 | `camera_fl = 910.` focal length (F2) | FORGOTTEN | terminal pose solve (se(3) ego-screw) | MED — real geometry constant absent from live pose doctrine |
| 5 | yuv6 luma = 2×2 POLYPHASE (y00/y10/y01/y11) + chroma half-res (F4) | NEW | terminal pose solve · chroma d_seg lever | MED — the exact luma/chroma structure the carriers write into |
| 6 | baseline_fast reference (pose .38042614 / seg .00946623 / rate .05979147 / 4.39) (R6) | NEW | byte-close/export sanity | MED — known-good eval fixture |
| 7 | Stale README leaderboard tops at 0.60 (R7) | NEW | gc13 convocation · pointer honesty | MED — never cite as score-to-beat (live ~0.172) |
| 8 | inflate 3-arg contract + `.raw (N,874,1164,3)` no header + evaluate.sh no-timeout (C1,C2) | NEW | byte-close/export chain | MED — exact export contract |
| 9 | MPS default auto-detect branch (E4) | FORGOTTEN | any local advisory eval | LOW-MED — must pass `--device cpu` |
| 10 | DALI-pad vs AV-drop trailing frame (F8) | FORGOTTEN | dual-axis CUDA/CPU drift understanding | LOW — 0.mkv even → inert; latent odd-frame source |
| 11 | AllNorm(BN1d(1), eps .001/mom .01) + gelu_tanh + fp32/no-autocast (M9,M10,M11) | NEW | PoseNet torch/MLX parity | LOW-MED — parity surfaces for pose |

---

## 6. Attack on my own conclusion (§6 craft-manual)

- **"0 CRITICAL" could be complacency.** Two cross-check laws (ERF 85px, class order) are data/forward-pass
  properties I did NOT re-measure — a no-scorer-jobs read cannot. I did NOT prove them; I proved their
  architectural PRECONDITIONS (stride-2 stem, argmax on last frame) hold. Labeled "consistent, not
  re-measured", not "HOLDS".
- **timm parity: keys match ≠ numerics match.** D6 proves `missing/unexpected=0` (structure identical); it
  does NOT prove bit-identical forward across 1.0.22↔1.0.27. Labeled MEASURED-benign at the key level,
  numeric-forward parity OWED. Mitigated because the AUTHORITY venv is `upstream/.venv` (1.0.22) regardless.
- **Rate denom "constant vs dynamic" is not a number error.** `sum(rglob)==37,545,489` today (measured). The
  finding is a MECHANISM/operational hazard, not a wrong value. Framed as such.
- **Scorer-ship +62.8 is a DERIVED upper bound** (25·bytes/denom per weight, summed) assuming both shipped
  uncompressed; a compressed scorer would cost less. Labeled DERIVED, and it's an existence-of-stakes
  argument for the guard, not a live plan.

## 7. Consumers to notify (this memo feeds them)

- **gc13 convocation** (running parallel) — will consult §3/§5; the stale-leaderboard (R7) + scorer-size
  (D5) rows are the load-bearing ones for any rate-axis framing.
- **#809 cg1 guard ledger** — D5 (+62.8 ship-both) is the quantified stake; E3 (dynamic denom) is a second
  payload-cleanliness surface (stray `videos/` files).
- **burn-4 endpoint consumption** — D6 timm parity note (authority = replica venv).
- **byte-close/export + rate coder stack** — C1/C2 exact contract; E3 videos/-cleanliness precondition; R6 fixture.
- **terminal pose solve** — F2 (camera_fl=910) + F4 (yuv6 polyphase) are new geometry inputs.
