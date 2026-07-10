# ADVISORY — PR #128 HNeRV reverse engineering, SDF transfer, and recursive v7.5.3/v8 design — 2026-07-10

`research_only=true`

**Authority:** advisory means artifact. No training, dispatch, official evaluation, pointer move,
run signal, or run termination was performed. The exact public archive was acquired and parsed on
the SSD tier, its deterministic packer was replayed, and a single-pair decoder smoke was used only to
verify the parsed tensor contract. The public score remains unratified.

**Answer first:** PR [#128](https://github.com/commaai/comma_video_compression_challenge/pull/128)
is **unambiguously an HNeRV-family descendant, not a new representation family**. It preserves the
PR #95 HNeRV-style decoder and its learned pair embeddings, PR #98 render biases, PR #101 quantized
payload lineage, PR #110 FEC6 selector, and PR #112 container/coder. Its actual contribution is a
new *optimizer/packing treatment of the existing representation*: delete a 607-byte floating latent
sidecar, absorb its useful effect into the native uint8 latent grid, and polish that grid with claimed
exact-score-gated `+-1/+-2` coordinate moves on the CPU axis. No decoder architecture, learned
weight tensor, selector sequence, frame grammar, or representation family changed.

The useful Pact lesson is therefore not “switch to PR128/HNeRV.” It is: **after a receiver-closed
SDF/level-set vehicle exists, expose its charged integer carrier coordinates and finish it with a
resumable, exact-through-R, real-coder discrete ratchet.** That finisher cannot substitute for class
birth, topology, edge integrability, Pose carriage, or receiver closure.

## Literal dispositions

| Object/action | Disposition | Reason |
|---|---|---|
| Classify PR128 | **HNeRV family; payload-polish child** | decoder, weights, latent semantics, selector, container, and inflate lineage are inherited |
| Treat PR128 as a new family | **REFUSE** | changing stored coordinates does not change the sufficient-statistic family or decoder map |
| Promote claimed `0.187991` | **REFUSE** | open PR, no maintainer evaluation, no reviews/checks, stale PR body, release-tag/source mismatch |
| Preserve exact PR128 bytes as an external candidate | **GO; custody only** | deterministic packer reproduces the exact current archive and member |
| Run an official/full scorer in this advisory lane | **NOT LAUNCHED** | outside the advisory contract; no new dispatch/eval authority was inferred |
| Transfer exact integer polishing to v7.5.3/v8 now | **HOLD** | first close MLX/NumPy/inflate parity, charged-section consumption, topology, and complete-artifact selection |
| Build the transfer mechanism after receiver closure | **GO as a future bounded implementation** | it directly optimizes the legal realized objective and complements continuous training |
| Current v7.5.3 A2/A3 launch | **REFUSE** | trained texture/hidden state is not one receiver-consumed vehicle |
| Current v8 increment-1a launch | **REFUSE** | trainable state is not fully decoupled, receiver ignores the new head, and gates lack receipts |
| `owed16v2_rebalanced_ON_20260710T114759Z` | **COMPLETED NATURALLY; this lane sent no signal** | owner harvested ep700 after the earlier live observation; rebalanced warm-start was formulation-scoped NO-GO |
| Shared v7.5.2 pilot | **EXTERNAL OWNER LAUNCHED; this lane did not launch/signal** | `levelset_v752_pilot_20260710T154100Z` was observed live at the final state refresh; preserve its owner/custody boundary |

## 1. Public authority, custody, and chronology

### 1.1 Exact objects inspected

| Field | Exact value / scope |
|---|---|
| Upstream PR | `commaai/comma_video_compression_challenge#128`, open and unmerged |
| Base | `991b317c41fe3aac657e0f0cb88fd831b2e4185a` |
| Head branch | `submission/rhnerv_latent_polish` |
| Head commit | `3eb39cac8261075888b1c562e9d9c2a7f1c7aebf` |
| Submission tree | `efd66a7f56e6eb580801507758fbd6ba52924c28` |
| Author fork | [a12dongithub/comma_video_compression_challenge](https://github.com/a12dongithub/comma_video_compression_challenge) |
| Release | [`rhnerv-latent-polish-20260709`](https://github.com/a12dongithub/comma_video_compression_challenge/releases/tag/rhnerv-latent-polish-20260709) |
| Current release asset | 176,531 B; SHA-256 `cfd941de10e5c27a5c855f97b0c84e39f6171f23c53c150e4afd90915f41e395` |
| ZIP member `x` | 176,431 B; SHA-256 `8f7b808e34c0f679fc7fd4fa5b58395acb03d76f981cd183bbae2453f65f6f22`; CRC32 `341ecef1` |
| Forensic source | `/Volumes/VertigoDataTier/pact/public_pr128_intake_20260710/source` (detached, external input only) |
| Downloaded candidate | `/Volumes/VertigoDataTier/pact/public_pr128_intake_20260710/archive.zip` |
| PR112 comparator | 177,136 B; SHA-256 `dd4f3899b91f5b59df90b4bf4fc4d903099a286548339f5f65ff91e4b8146aa4` |

Local-first search found no pre-existing PR128 archive custody. The public fork and exact asset were
therefore acquired to `/Volumes/VertigoDataTier`, not to the source tree. The current `compress.py`
reconstructed the member and ZIP byte-for-byte; no payload was regenerated or normalized in Pact.

### 1.2 Public state is internally inconsistent

- The PR is open, has zero reviews, and has no evaluation check on the current head. Its only public
  comment says a maintainer must trigger evaluation.
- Head docs claim `0.187991`, but the PR title/body/template still describe the prior `0.188532`
  commit and prior `fae8d338...` archive.
- Release tag `rhnerv-latent-polish-20260709` points permanently at `ea478f64...`, whose packer
  expects `fae8d338...`. The asset currently served under that tag is the later head archive
  `cfd941de...`. The tag source therefore cannot rebuild its own currently served asset.
- There is no tag at `3eb39cac...`. A second older `v1` release also has a body-to-current-asset
  digest mismatch. Public API evidence cannot determine whether assets were deleted/re-uploaded or
  added later, so the mechanism is `UNKNOWN`; the custody break itself is measured.
- The author account had 29 public repositories at inspection time. The challenge fork was the only
  challenge-related repository and GitHub search found no other challenge PR by this author. No
  additional implementation lineage was found in the author's other repositories.

Required promotion predicate: bind `{PR number, head SHA, submission tree, archive SHA/bytes,
member SHA, release tag target, committed report, official evaluation receipt, axis}` and refuse if
any required field disagrees.

### 1.3 Four-commit evolution

| UTC | Commit | Publicly described change |
|---|---|---|
| 2026-07-09 07:51 | [`61f47a2`](https://github.com/a12dongithub/comma_video_compression_challenge/commit/61f47a28328e44bc2d7b356f79627b95840620b4) | initial claimed CPU `0.189227`; no encoder/raw inputs |
| 2026-07-09 09:12 | [`53955e2`](https://github.com/a12dongithub/comma_video_compression_challenge/commit/53955e23baecc176fc775d064fb9f75107d2feb5) | deterministic packer plus three Git-LFS raw inputs |
| 2026-07-09 17:06 | [`ea478f6`](https://github.com/a12dongithub/comma_video_compression_challenge/commit/ea478f64f230111e20f78f736673933c15b8ca49) | claimed CPU `+-1/+-2` reselection; `0.188532`; latent LFS object changed |
| 2026-07-10 09:01 | [`3eb39ca`](https://github.com/a12dongithub/comma_video_compression_challenge/commit/3eb39cac8261075888b1c562e9d9c2a7f1c7aebf) | four CPU rounds claimed complete; `0.187991`; only latent payload/docs/hash/report changed materially |

The final branch is exactly four commits ahead and zero behind the inspected base. `master` in the
fork remains at the base; this feature branch is the sole PR128 production surface.

## 2. Family lineage: inherited representation versus actual novelty

The primary [HNeRV paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html)
defines the family by content-adaptive learned embeddings fed to a neural video decoder. PR128 still
stores one learned 28-dimensional embedding per frame pair and feeds it to the unchanged HNeRV-style
decoder. Removing the training-time encoder and shipping the learned embeddings does not turn the
decoder into a new representation family.

It is also **not HiNeRV**. [HiNeRV](https://hmkx.github.io/hinerv/) is a distinct hierarchical INR
with multiscale positional encodings, depthwise/MLP/interpolation blocks, patch-and-frame coordinates,
and its own pruning/quantization pipeline. PR128 has none of those structures: it retains the flat
28-d pair embedding, `6x8` HNeRV stem, and PixelShuffle decoder. HiNeRV's official implementation is
MIT and remains a lawful separate control/campaign input; its PSNR/bpp results are not evidence for
PR128 or for Pact's scorer-native objective.

| Ancestor | What survives in PR128 | Identity evidence |
|---|---|---|
| PR #95 | HNeRV-style `28 -> 6x8` stem, six upsampling stages, refinement, two RGB heads | executable model code is unchanged; PR128 adds provenance text |
| PR #98 | quantized weight/latent lineage and fixed channel biases (`f0 R-1`, `f0 B-1`, `f1 G-1`) | inherited decode behavior |
| PR #101 | quantized decoder streams, latent grid, and 607-byte correction-sidecar ancestry | decoder raw remains byte-identical; sidecar is removed/folded |
| PR #110 | 600-entry FEC6 K=16 selector and post-round transform order | raw selector and source module are byte-identical |
| PR #112 | version-1 three-section ctx container and all entropy-coder implementations | `codec.py`, `codec_ctx.py`, `frame_selector.py`, and decoder/selector sections are identical |
| PR #128 | different native latent coordinates, no sidecar, deterministic final-input packer | only distinctive charged source input is `polished_latent_raw.bin` |

PR [#127](https://github.com/commaai/comma_video_compression_challenge/pull/127) is a parallel
HNeRV/PR112 payload-polish descendant using gradient/straight-through exact-grid language. It is not
an ancestor of PR128 and PR128 contains no PR127 code attribution. Both may be independently useful
optimization treatments; neither is a family break.

Taxonomic test:

```text
same decoded sufficient statistic z_p
+ same decoder F_theta(z_p)
+ same pair/frame/render/selector grammar
+ only q_p values and packing change
= same HNeRV representation family, new payload optimizer
```

## 3. Exact archive grammar and every charged payload

### 3.1 ZIP and member map

The archive is deterministic and deliberately uncompressed at the ZIP layer: one unencrypted
`ZIP_STORED` member named `x`, DOS epoch `1980-01-01`, no extra field, no archive comment.

| Archive offsets | Bytes | Meaning |
|---|---:|---|
| `0..30` | 31 | local ZIP header including one-byte filename |
| `31..37` | 7 | ctx container header |
| `38..161141` | 161,104 | decoder section |
| `161142..176213` | 15,072 | latent section |
| `176214..176461` | 248 | selector section |
| `176462..176508` | 47 | central directory |
| `176509..176530` | 22 | end-of-central-directory |

ZIP overhead is exactly 100 bytes. The seven-byte header is
`17 50 75 02 e0 3a 00`:

- `0x17`: version `1`; lower bits 0/1/2 select ctx coding for decoder/latent/selector;
- `50 75 02`: little-endian decoder length `161104`;
- `e0 3a 00`: little-endian latent length `15072`;
- selector consumes the remainder.

The implementation uses one coder-ID bit per section, not two as some prose suggests, and does not
reject reserved bit 3.

| Section | Coded bytes | Decoded bytes | Decoded content | SHA-256 anchor |
|---|---:|---:|---|---|
| decoder | 161,104 | 229,014 | 228,958 uint8 weight codes + 28 fp16 scales in seven streams | raw `83598024bdb4d60463610db23934cdee60c3b6a81158a97e0dd55ea621833fcd` |
| latent | 15,072 | 16,912 | 28 fp16 minima + 28 fp16 scales + 16,800 temporal-delta bytes | raw `a7eba9722beb7f5bfe6027175307fa374dc5c081fa31bb805e02755787f7a98c` |
| selector | 248 | 249 | `FEC6`, `u16(600)`, 243 Huffman bytes | raw `fc5c431b5d793c33e2f320076fe6f0dd76c2d91e3826ae4b05abfb4f86f453ca` |

Section hashes are decoder `35fd7beaaeac8a0fe4f74c15e6a7a906b153dd18b9d2e66bf99bab435508cece`,
latent `24c7ff0595849483b1fbfc6b5975e23b90810dc1040ea26c28e94b5e0ede0a41`, and
selector `d7b277473bfd7235d35b879ab1adccf92ba77eb0ba3e58d5266646470a147ae1`.

### 3.2 Decoder architecture and weight reconstruction

`model.py` contains a 228,958-parameter HNeRV-style pair decoder:

```text
z_p in R^28
  -> Linear(28, 36*6*8), reshape 36x6x8, sin
  -> six [Conv3x3(out*4) -> PixelShuffle2 + bilinear skip -> sin]
     channels [36,36,36,27,20,18,18]
  -> dilated two-conv residual, gain 0.1
  -> separate sigmoid RGB heads for frame0 and frame1
  -> 2x3x384x512, scaled to [0,255]
```

The seven raw streams have lengths `974, 1460, 157680, 488, 584, 19442,
48386`. Twenty-eight tensors are stored in a fixed order. Each tensor reconstructs as

\[
W_k = \operatorname{fp32}(\operatorname{map}^{-1}_k(b_k))
      \operatorname{fp16}(s_k).
\]

Most byte maps are zigzag; tensors 9/14 use negated zigzag, tensor 20 uses two's-complement, and
tensor 27 uses offset-binary. Conv tensors carry fixed storage permutations. The ctx coder uses 25
adaptive 256-ary models; tensor indices `{7,5,1,3}` share one model. It chooses geometric-prior and
adaptation parameters encoder-side, signals them compactly, and entropy-codes fp16 scale high bytes
separately from their low bytes.

### 3.3 Latent representation and coder

For absolute uint8 latent coordinate `q[p,d]`, the decoded value is

\[
z_{p,d}=m_d+s_d q_{p,d}, \qquad q_{p,d}\in\{0,\ldots,255\},
\]

with `m_d,s_d` stored as fp16. The wire grid is dimension-major temporal differences:

\[
u_{d,0}=q_{0,d},\qquad
u_{d,p}=128+q_{p,d}-q_{p-1,d}\pmod {256}.
\]

After fixed dimension reordering, cumulative summation reconstructs `q`. The ctx layer predicts the
centered temporal-difference sequence with Q6 coefficients: an own AR(1) term plus an optional own
lag-2 and up to four earlier decoded dimensions. Residuals use selected discrete-Gaussian models;
PR128's final latent header has 35 cross features and no adaptive residual flags. The section is 228
bytes of metadata plus a 14,844-byte range body. All 28 absolute dimensions are active; the final
absolute grid differs from PR112 in every dimension.

A `q[p,d] += delta` click normally changes two *wire* differences, at `p` and `p+1`; it may also
change multiple predictive residuals and the encoder-selected model plan. Consequently, “two deltas
change” is not an exact byte-cost law. Exact re-encoding is mandatory.

### 3.4 FEC6 selector payload

The selector ctx stream regenerates the exact 249-byte FEC6 Huffman payload. All 1,944 payload data
bits are occupied. Mode counts are:

| Mode | Count | Mode | Count |
|---|---:|---|---:|
| identity | 134 | frame0 blue-chroma amp 1 | 35 |
| frame0 blue-chroma amp 3 | 129 | frame0 luma `+1/-1/-2/-4` | `9/25/13/11` |
| frame0 RGB `(-2,+1,+1)` | 71 | frame0 RGB `(-4,+2,+2)` | 10 |
| frame0 RGB `(0,-1,+1)` | 24 | frame0 RGB `(0,-2,+2)` | 7 |
| frame0 RGB `(0,+1,-1)` | 16 | frame0 RGB `(0,+2,-2)` | 6 |
| frame0 RGB `(+2,-1,-1)` | 92 | frame0 RGB `(+4,-2,-2)` | 17 |
| frame0 roll `(dx=0,dy=+1)` | 1 | any frame1 mode | 0 |

This is inherited content, not PR128 novelty.

## 4. All submitted source, scripts, data, and information

The current submission tree contains 17 files. The three payload inputs are Git-LFS objects; the
archive itself is not committed.

| Path | Bytes | SHA-256 | Role |
|---|---:|---|---|
| `LICENSE` | 1,245 | `abd4a2dcda8bfa1f8a727d4e30ad9e4e7ba7890fb96f8359a9a428e9e5362310` | contest MIT text/copyright surface |
| `README.md` | 4,892 | `6309a8c3305c3ab2a5aeb66284c42517dedf66744708e8724c65729c0cb43143` | current public claim, custody table, quick runbook |
| `METHOD.md` | 8,622 | `27d5df4b1259bbc573e9c1d365be4303d028b64e503643a7e8852cf844a96028` | prose search algorithm; stale final counts |
| `THIRD_PARTY_NOTICES.md` | 5,337 | `8ab6c84da000daf2fd5605c0272a11a6489a1409468bda20207c0b488fd8268f` | PR95/98/101/110/112 attribution |
| `model.py` | 2,453 | `5cd9c4ed7f9b30e6181ceda0a362b3a2875ed75c33bff6a4010589588b981fd3` | HNeRV-style decoder |
| `codec.py` | 5,656 | `979b0e8f098600e5e0c56dae5d48a88fc4f72fe5c54d8be3c0bc818e6fd88e41` | weight/latent tensor reconstruction |
| `codec_ctx.py` | 27,764 | `adb5f255fec2e1ea4459b12866b8f570d147a6df56bc0f7bfb81faae75ac30af` | PR112 decoder/latent/selector range coder |
| `frame_selector.py` | 8,011 | `a84941518836f312ab8c5bdf21975c970fdc0d89c41eed935642701e833cda2a` | inherited FEC6 transform helpers |
| `inflate.py` | 9,122 | `c6c4897c952df775da13e59da10e6841353b150359beebc53377253e190489bc` | CPU receiver; PR112 path minus sidecar |
| `inflate.sh` | 849 | `e5f8a63756f6e1e67a8c8c0abb4290412a0fd91deecc90753217a0aee7bd0ee7` | harness wrapper and Python selection |
| `compress.py` | 4,094 | `1eed8db7919987ac8e1a9ef07143f4e5a249112ed9514621a2dda5db14ea1f3a` | deterministic range-code + ZIP packer |
| `compress.sh` | 518 | `58963bfcdee315bcbf987e63ba3a01d612aac5af7977828d3ecca32fc9d637fc` | packer wrapper |
| `encoder/decoder_streams.bin` | 229,014 | `83598024bdb4d60463610db23934cdee60c3b6a81158a97e0dd55ea621833fcd` | frozen quantized decoder raw |
| `encoder/polished_latent_raw.bin` | 16,912 | `a7eba9722beb7f5bfe6027175307fa374dc5c081fa31bb805e02755787f7a98c` | PR128's distinctive charged latent raw |
| `encoder/selector_payload.bin` | 249 | `fc5c431b5d793c33e2f320076fe6f0dd76c2d91e3826ae4b05abfb4f86f453ca` | frozen FEC6 selector raw |
| `expected_output.sha256` | 65 | `0f6600f07847cb6ec38f70560667bcb52379a1f18b789a0d6344d5f6ca4c55aa` | author-machine raw hash `8c5774c3...` |
| `report_cpu.txt` | 737 | `73b9f674eac9103535fc36117cc1e979e9d8b47ce6323ccf86306927a6c76ec7` | self-run Windows CPU report |

Not included: the actual latent-search program, proposal ledger, rejection ledger, pure sidecar-fold
baseline, search checkpoints, requirements file, pinned environment, official evaluation receipt,
checked-in `archive.zip`, or a `report.txt` matching the PR template. `compress.py` reproduces the
final artifact from already-polished inputs; it does not reproduce how those inputs were found.

## 5. Inflate DAG and runtime behavior

```text
ZIP member x
 -> parse 7-byte ctx container
 -> range-decode 28-tensor HNeRV state, 600x28 latents, 600 selector codes
 -> for p=0..599 in 16-pair batches:
      HNeRV(z_p) -> two RGB frames at 384x512
      bicubic resize to 874x1164, align_corners=False
      f0.R -= 1; f0.B -= 1; f1.G -= 1
      clamp [0,255] and round
      apply selected frame0 FEC6 transform
      clamp and round again
      uint8 -> NHWC -> streaming write
 -> 1,200 frames; expected 3,662,409,600 raw bytes
```

`inflate.sh` prefers `PACT_PYTHON_BIN`, then `python`, then `python3`. For each nonempty file-list
entry it reads shared member `x` or falls back to `<base>.bin`. The Python path is pinned to CPU and
requires NumPy, Torch, and `constriction`. Source contains no network access, scorer weights, GT mask
table, pickle load, dynamic code execution, or uncounted learned/video tensor. Charged bytes contain
the decoder, scales, latents, and selector. Fixed architecture/coder/transform tables are receiver
grammar inherited from the prior PRs.

One parsed pair produced shape `(1,2,3,384,512)` with finite `[0,255]` output, confirming tensor and
container compatibility. A full 3.66-GB inflate/evaluator run was intentionally not launched.

## 6. Exact PR112 delta and score accounting

| Component | PR112 | PR128 | Delta |
|---|---:|---:|---:|
| ZIP overhead | 100 | 100 | 0 |
| decoder ctx | 161,104 | 161,104 | 0 |
| latent ctx | 15,070 | 15,072 | +2 |
| selector ctx | 248 | 248 | 0 |
| trailing float sidecar | 607 | 0 | -607 |
| total archive | 177,136 | 176,531 | **-605** |

Decoder and selector coded sections and decoded raw bytes are identical. The fp16 latent min/scale
header is also identical. Only the temporal-delta grid changes.

Archive-verifiable absolute-grid comparison against PR112:

- 2,656 of 16,800 absolute coordinates differ;
- 598 of 600 pairs and all 28 dimensions differ;
- absolute L1 step distance is 3,111; range is `-5..+5`;
- signed coordinate histogram is
  `-5:1, -4:1, -3:45, -2:143, -1:1212, +1:1080, +2:141, +3:26, +4:6, +5:1`;
- the temporal-delta wire grid differs in 4,664 bytes.

The PR112 sidecar contains one possible correction per pair: 597 non-noops and three noops, using
floating values in `+-{0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10}` on all 28 dimensions. Those
corrections are generally not native-grid multiples. Therefore “folding” is a discrete
re-optimization, not value-preserving syntax substitution.

Historical LFS objects explain the documentation drift without validating it:

| Payload state | Direct absolute-grid delta versus PR112 | Shipped prose at that time |
|---|---:|---|
| `53955e2` / `5b2b...` | 2,031 coordinates / 596 pairs | about 1,565 adjustments |
| `ea478f6` / `106e...` | 2,308 / 598 | 1,802 / 577 |
| final / `a7eb...` | 2,656 / 598 | README 2,162 / 583; METHOD still 1,802 / 577 |

The prose counts plausibly refer to search clicks or net polish relative to an unshipped pure-fold
baseline, not total change from PR112. Because that baseline and search ledger are absent, `2,162 /
583` is **source-unverifiable**, not disproved by the final-byte comparison.

Using the displayed components, the score arithmetic is:

| Term | PR112 | PR128 claim | Delta | Share of absolute claimed gain |
|---|---:|---:|---:|---:|
| `100*d_seg` | `0.056023` | `0.053309` | `-0.002714` | 86.59% |
| `sqrt(10*d_pose)` | `sqrt(10*0.00002943)` | `sqrt(10*0.00002937)` | `-0.0000174964` | 0.56% |
| rate | 177,136 B | 176,531 B | `-0.0004028447` | 12.85% |
| total | `0.1911257657` | `0.1879914246` | **`-0.0031343410`** | 100% |

This is arithmetic consistency, not independent score evidence. `report_cpu.txt` is a Windows CPU
run at batch 16, two threads, seed 1234 and prints only rounded component values and final `0.19`.

Rate geometry at n600:

\[
c_B=\frac{25}{37{,}545{,}489}=6.6585895312\times10^{-7}\ S/\text{byte},
\]

\[
c_{\rm segcell}=\frac{100}{600\cdot384\cdot512}
=8.4771050347\times10^{-7}\ S/\text{corrected cell}.
\]

Thus one corrected Seg cell can pay at most 1.273108 bytes; each extra byte needs more than 0.785479
corrected Seg cells if Pose is unchanged.

## 7. Reverse-engineered search: what is sound, what is not proved

### 7.1 Claimed algorithm

For fixed latent dimension `d` and step `delta in {+-1,+-2}`, the author describes constructing a
batch in which every pair's own `q[p,d]` is changed, rendering all 600 pairs once, and treating row
`p` as the output of the candidate that changes only pair `p`. This diagonal batching is valid for
the HNeRV renderer because there is no temporal or cross-pair state in `F_theta(z_p)`.

Each round then allegedly:

1. records the current per-pair Seg/Pose errors and coded size;
2. enumerates up to `600*28*4 = 67,200` local candidates;
3. estimates byte deltas;
4. keeps improving proposals, at most one per pair per round;
5. applies the set, performs a real re-encode and full exact re-score;
6. accepts only a strict full-score improvement, otherwise bisects;
7. banks a shippable archive and repeats to a zero-accept plateau.

The public source ships neither the implementation nor its logs, so enumeration completeness,
batch layout, bounds handling, candidate counts, rejected moves, bisection behavior, and “plateau”
cannot be independently reproduced.

### 7.2 Exact mathematical factorization

Let pair-local errors be `e_p(q_p)` and `r_p(q_p)` and `N=600`:

\[
d_{\rm seg}=\frac1N\sum_p e_p,\qquad
d_{\rm pose}=\frac1N\sum_p r_p.
\]

Rendering and the two distortion *means* factor by pair. The total score does not add per-pair
candidate scores:

\[
\Delta S(A)=\frac{100}{N}\sum_{p\in A}\Delta e_p
+\sqrt{10\left(d_{\rm pose}+\frac1N\sum_{p\in A}\Delta r_p\right)}
-\sqrt{10d_{\rm pose}}
+c_B\Delta B(A).
\]

Two couplings remain:

1. the square root makes Pose score increments set-dependent even though raw Pose errors add;
2. the predictive/adaptive range coder and its refitted model choices make `Delta B(A)` globally
   set-dependent.

Therefore the README/METHOD statement that accepted clicks on different pairs “compose exactly” is
too broad. **The renderer and raw distortion sums compose; the full score does not.** The claimed
real re-encode/full-score acceptance step is the correct cure, if actually executed.

### 7.3 Why the method can still work

The selection proxy need only rank proposals. Exact acceptance on the exact serialized candidate
gives a monotone ratchet:

\[
A_{k+1}=\begin{cases}
A'_k,&S(A'_k)<S(A_k),\\
A_k,&\text{otherwise}.
\end{cases}
\]

Because the integer state space is finite and every accepted state strictly decreases `S`, a
deterministic search with no revisits must terminate. Its terminal certificate is only a local
minimum with respect to the enumerated move set (`+-1/+-2` coordinates and any registered blocks),
not a global optimum. Expand the move grammar with bounded multi-coordinate/section moves when
barriers are measured; do not use the word “optimal” without stating the neighborhood.

PR128 also reports useful formulation-scoped negatives:

- 7-bit/dead-zone weight requantization saved bytes but lost much more Seg score;
- entropy-regularized straight-through QAT improved its surrogate while true Seg worsened;
- gradient-ranked individual weight-code clicks were rejected;
- half-grid latent moves were not worth reintroducing a sidecar;
- GPU-selected moves lost claimed gain on CPU because bicubic LSBs changed;
- `+-2` proposals were needed where the `+-1` midpoint was worse.

None of these kills HNeRV weights, QAT, gradients, or sidecars as families. They are negative only for
the tested payload/formulations near this basin. The durable lesson is to profile slack per charged
section and scope every negative.

### 7.4 Corrections to the concurrently landed intake/design memos

The parallel Claude memos `.omx/research/pr128_intake_reverse_engineering_20260710.md` and
`.omx/research/clickpolish_to_witness_design_20260710.md` correctly identify HNeRV lineage, diagonal
render batching, and the high value of a native-code finisher. Five statements need narrowing before
their in-flight implementation can become authority:

1. PR112's committed full-precision Seg component is `0.00056023`, not `0.00056032`.
2. The PR101 sidecar values are generally not exact multiples of the native fp16 grid scales. Folding
   is not a value-preserving, zero-distortion syntax rewrite; it is an exact-gated discrete
   re-optimization whose pure-fold baseline is absent.
3. Pair-local rendering does not make full-score gains exactly additive because Pose uses a square
   root and archive bytes are coder-coupled.
4. A witness frame1 FiLM/code click changes a frame read by both SegNet and PoseNet. Its screening
   receipt must return both per-pair Seg and raw Pose deltas, not `d_seg` alone; the selected set then
   needs a real re-encode and full-S confirmation.
5. The public search counts and CPU score are author claims, not reproducible search/contest receipts.

The efficient implementation remains diagonal rendering, but its authority boundary is:

```text
pair-local candidate pixels and raw per-pair error deltas
 -> set-aware nonlinear Pose recomposition
 -> real coder/archive bytes
 -> exact full-S accept/reject
```

This is an additive hardening of the concurrent `clickpolish-build` lane, not a request to stop,
absorb, or duplicate its owned files.

## 8. Reproducibility, parser, compliance, and attribution findings

### 8.1 What is strong

- Current pack inputs, member SHA, ZIP metadata, archive SHA, and bytes are deterministic and were
  reproduced exactly.
- All learned/video-specific arrays needed to inflate are charged inside the archive; no scorer
  weights, GT labels, frame tables, or hidden network fetch were found.
- The receiver emits the right logical cardinality by construction: exactly 600 pair iterations and
  1,200 frame writes.
- Third-party notices trace the principal PR95/98/101/110/112 lineage and include the contest/PR110
  MIT texts.

### 8.2 Remaining defects

- Author docs explicitly say `torch.interpolate(..., bicubic)` yields different raw LSBs on
  different CPU microarchitectures. The committed `expected_output.sha256` is therefore not a
  portable decode authority and violates Pact's bit-identical cross-host receiver goal.
- `constriction` is unpinned in project dependency metadata and not custody-fixed by the committed
  lock state; a fresh unfrozen sync may resolve a different environment.
- Section range streams are not required to be consumed exactly. Trailing words can be ignored, and
  truncated selector experiments could still decode 600 altered codes without an internal integrity
  error. The fixed public ZIP CRC/external SHA protects the known artifact, but the grammar is not
  fail-closed for arbitrary inputs.
- Empty/truncated members fail through incidental exceptions; reserved header bit 3 is accepted;
  final raw byte count/hash is not asserted; output writes are non-atomic and non-resumable; no
  storage preflight protects a 3.66-GB output.
- Search reproducibility is missing even though final packing reproducibility is strong.
- `model.py` is not literally byte-identical to PR95 because provenance text changed; executable
  model behavior is unchanged.
- The tree-wide license says Matt Neel while new commits/payload/docs are authored by Samarth
  Singhal. Repository-license coverage is plausible under the contest MIT grant, but new-contribution
  authorship attribution is ambiguous.

These are custody/robustness findings, not evidence that the fixed candidate is malicious or contest
invalid. Official harness execution remains the contest-compliance authority.

## 9. Transfer to the original SDF/level-set witness paradigm

### 9.1 Transfer map

| PR128 technique | SDF/level-set adaptation | Admission boundary |
|---|---|---|
| pair-local diagonal batching | expose per-pair `xi`, modulation, palette, carrier, or residual integer blocks; batch candidates only for proven block-independent rows | shared weights, argmax competition, global scorer support, and coder state are not pair-local |
| native-grid `+-1/+-2` moves | mutate post-quantization field, carrier, palette, texture, and pose codes; include multi-step/barrier moves | only after NumPy/inflate decode the same codes and through-R output |
| exact full-score accept | use exact `Delta S`, real archive re-encode, fresh raw reload, and axis-tagged scorer | never admit on CE, margin, mask, PSNR, or entropy proxy alone |
| sidecar folding | project sparse correction effects into existing native carrier syntax when exact score improves | do not collapse geometry/paint/pose ownership merely to remove a section |
| CPU reselection | propose cheaply on MLX/GPU but reselect on each promotion axis | no CPU-to-CUDA or macOS-to-contest inference |
| bank every acceptance | atomically save exact payload, receiver, archive, score receipt, mutation log, and optimizer/search state | every stage and accepted move must remain resumable |
| section-local slack audit | separately screen G weights, per-pair codes, class biases, `xi`, T, fills, selectors, and residuals | one dead section does not kill the vehicle/family |

### 9.2 The correct generalized finisher

Let `a` be the vector of charged integer coordinates and `Pack(a)` the deterministic archive/receiver
pair. Define the authority objective

\[
J(a)=100d_{\rm seg}(R(\operatorname{Decode}(\operatorname{Pack}(a))))
+\sqrt{10d_{\rm pose}(R(\operatorname{Decode}(\operatorname{Pack}(a))))}
+c_B\,|\operatorname{Pack}(a)|.
\]

A reusable finisher must:

1. load a complete byte-closeable stage checkpoint and freeze its receiver grammar;
2. enumerate allowed integer moves from typed section manifests;
3. use JVP/margin/entropy models only to order candidates;
4. build the actual candidate archive and parse it back;
5. evaluate exact realized components on the target axis;
6. accept only `J(a') < J(a) - epsilon_authority`;
7. atomically bank `{a', archive, receiver, hashes, components, axis, mutation, predecessor}`;
8. resume from that bank with RNG/proposal queue/coder state preserved.

Where effects can overlap, construct a conflict graph. Two coordinates may share a diagonal batch
only after a receipt shows disjoint decoder support *and* scorer-safe interaction below the declared
tolerance. Color the graph and exact-check each selected independent set. For v8, class-local
parameter Jacobians may be block diagonal before composition while the tropical argmax and SegNet
remain coupled; the composition gate is never optional.

### 9.3 What must not transfer

- the 28-dimensional HNeRV latent coordinates or their numeric moves;
- the HNeRV decoder widths, tensor order, byte maps, FEC6 table, or channel-bias constants;
- PR128's claimed score, CPU report, or GPU/CPU selection behavior as authority for Pact;
- any inference about island birth, rare-class geometry, SDF gauges, temporal screws, or edge
  carriers—PR128 has none of those structures;
- the unsupported claim that different-pair full-score deltas add exactly;
- code from a repository with no applicable license. A public no-license repository is research
  evidence only; implementation must be clean-room/re-derived or permission obtained.

## 10. Corrected recursively optimized v7.5.3 target

The optimal evidence-bounded v7.5.3 remains one coherent object `W=(G,xi,T)`, but the current T path
must be replaced, not merely tuned.

### 10.1 Target representation

- **G — geometry:** inherit the corrected v7.5.2 single-trunk SDF/partition vehicle. The completed
  owed16v2 bounded warm-start rebalanced arm measured `0.006409/0.004286/0.004213` at ep650/675/700
  versus OFF `0.006295/0.004244/0.004181`: rebalanced is `+0.77%..+1.81%` worse. This removes the
  along-26 rebalance from the current formulation and selects self-orient OFF; single-seed/noise-floor
  limitations leave only a future from-scratch formation test open. Do not convert it into a
  family-level basis verdict.
- **xi — pose:** select a complete, receipt-compatible pose artifact. Frame0 is Pose-only; spend no
  Seg texture there. Banked-R1 versus a new finish is a complete-artifact selector, never a weight
  graft across incompatible EMA states.
- **T — texture/paint:** frame1-only residual applied **after final nonlinear RGB composition**,
  projected into the exact local Pose-preprocess kernel, then lifted through a bounded integer
  camera-grid preimage and verified after raw reload. It owns no DC/palette bias.

At the 384x512 scorer grid, a first-order Pose-preprocess-null 2x2 block satisfies, for each pixel,

\[
0.299\Delta R+0.587\Delta G+0.114\Delta B=0,
\]

and over the block

\[
\sum\Delta R=0,\qquad \sum\Delta B=0.
\]

The six-dimensional local kernel is spanned by three zero-sum 2x2 Haar patterns crossed with

\[
c_U=(0,-0.3441362862,1.772),\qquad
c_V=(1.402,-0.7141362862,0).
\]

Projection before sigmoid, class blending, clamp, resize, or uint8 is insufficient: those operations
can leak back into Pose. Project after the last learned RGB nonlinearity, solve the actual integer
preimage, and require the first six Pose preprocessing outputs to remain identical after fresh raw
decode. “High-frequency chroma” alone is not an exact-null theorem.

### 10.2 Evidence-conditional typed design

```yaml
vehicle: crucible_v753_exactD_target
inherits: crucible_v752
geometry:
  self_orient: OFF_for_current_warmstart_formulation
  owed16v2_result: MEASURED_NO_GO_REBALANCED_WARMSTART_FORMULATION
  from_scratch_basis: HELD_SEPARATE_FORMULATION
texture:
  enabled: BUILD_ONLY_UNTIL_RECEIVER_CLOSED
  frame: 1
  insertion: post_final_rgb_nonlinearity
  basis: exact_pose_preprocess_kernel_2x2_haar_x_chroma
  camera_preimage: bounded_integer_exact_D
  palette_or_dc_bias: forbidden
  placement_gradient_default: stopgrad
  placement_coupled_mode: explicit_matched_probe
  transport_modes: [stationary, xi_advected]
  support: geometric_sdf_annulus
  activation: event_after_birth_stable_and_geometry_plateau
  deterministic_basis_samples_in_archive: forbidden
  charged_payload: learned_coefficients_and_regeneration_metadata_only
optimizer:
  stage_order: geometry_then_texture_then_exact_discrete_finisher
  texture_group: separate
  geometry_frozen_at_texture_birth: true
  joint_unfreeze: matched_probe_only
resume:
  event_state: required
  all_optimizer_groups: required
  proposal_queue_and_accept_log: required
  preserve_every_stage_ema_checkpoint: true
selection:
  object: complete_receiver_closed_artifact
  metric: exact_full_S_on_declared_axis
```

No unmeasured width, learning rate, ramp, annulus width, or frequency band is promoted here. Candidate
bands `{period 4,6,8}` are a probe grid, not a sealed optimum. Choose coefficient rank, class
allocation, and frequency by exact score-units-per-byte waterfilling after receiver closure.

### 10.3 Curriculum

1. **G formation:** coarse partition, area-Lagrange/Chan-Vese counterforce, birth/persistence,
   eikonal gauge, tie-locus/margin, temporal screw; T output and gradient exactly zero.
2. **Geometry bank:** save a complete EMA stage checkpoint and fresh receiver receipt.
3. **T event:** fire only when island births are stable and the G plateau criterion is met; persist the
   event and independent optimizer state.
4. **Exact-D T fit:** freeze G by default; compare stationary versus `xi`-advected coordinates and
   stop-gradient versus coupled placement at exact matched bytes.
5. **Quantize/parse back:** regenerate generic basis code-side; serialize only learned coefficients.
6. **Head solve:** optimize the small native output/palette/carrier tensors while maintaining the
   exact-D constraint.
7. **PR128-style discrete ratchet:** `+-1/+-2` over receiver-visible native codes, with full Seg,
   Pose, and real-coder acceptance and stage-resumable banks.
8. **Pose selector/finish:** compare complete compatible artifacts; retain rollback to the banked
   floor.
9. **Byte close and exact axes:** final cardinality, decode time, hash, contest-CPU, and contest-CUDA
   remain separate receipts.

### 10.4 Current v7.5.3 blockers

1. MLX trains T while canonical NumPy verdict and inflate omit it; those are different vehicles.
2. Byte close can charge a deterministic bank (`430,878` isolated Brotli bytes, about `0.286904 S`)
   that inflate ignores. Fixed generic samples must be regenerated; learned/video samples must be
   counted and consumed.
3. Current T is unconstrained pre-sigmoid RGB on both frames, not frame1 exact-D texture.
4. T is live from initialization in the global optimizer; event, independent optimizer, and resume
   state are absent.
5. T bias duplicates the palette gauge; softmax placement silently couples T gradients into G.
6. Confidence/tau gating is not a geometric signed-distance annulus.
7. `OutTexHidden` is not implemented by the NumPy/inflate receiver and dimensional parity is not
   established for a hidden width different from the ordinary output head.
8. No exact-archive-byte A1/A2/A3 control exists.
9. The current MC finisher omits Pose from its admission objective and does not preserve all adaptive
   state needed for bit-faithful resume.

Until all nine close, “optimal v7.5.3 config” means the conditional target above, not runnable
science and not a launch command.

## 11. Corrected recursively optimized v8 target

### 11.1 Resolve K-class versus E-edge ambiguity with integrable potentials

Use K class potentials with disjoint trainable state:

\[
\phi_c(x,t)=F_c(x,\xi_t,z_{t,c};\theta_c),\qquad
P(x,t)=\arg\max_c\phi_c(x,t).
\]

Requirements:

- `theta_c` and `z[t,c]` are disjoint; no shared trainable code;
- `xi_t` is frozen/exogenous and bound to a selected complete Pose receipt;
- a frozen generic dictionary may be shared and regenerated;
- fix the additive gauge with a pinned reference or weighted zero-sum potential/bias law;
- derive every live edge preference as `g_ij=phi_i-phi_j` receiver-side.

If an edge stream is serialized, it must lie in the gradient/cut space of the class graph. With
incidence matrix `B`,

\[
g=B^T\phi,\qquad
P_{\rm grad}=B^T(BB^T)^+B,\qquad
\|(I-P_{\rm grad})g\|=0
\]

after quantization and decode. Arbitrary independent edge fields carry nonzero cycle/curl components
and cannot be a globally consistent class-potential partition. The clean design serializes
gauge-fixed class potentials or a spanning-tree basis and derives all other edges for free.

For class-local optimization, freeze a Jacobi snapshot of competitors:

\[
m_c=\phi_c-\operatorname{stopgrad}\max_{j\ne c}\bar\phi_j.
\]

Then apply class-local SDF, eikonal, birth/completion, area-Lagrange, topology, temporal-screw,
quantization, and rate terms. Loss weights change only at stage boundaries. The decoded tropical
composition is still globally scored; pre-composition block diagonality is not evaluator
factorization.

### 11.2 Carrier homes and evidence scope

| Class/term | Target home | Current evidence boundary |
|---|---|---|
| Undrivable | default/top-region basin plus compact lateral boundary carrier | curve/frozenness and receiver bytes remain owed |
| Road | compact horizon geometry plus flat/per-pair fill | whole-region period-4 grating was strongly harmful; flat fill is the safer measured pin |
| Lane | analytic ground-frame band as a trained geometry lever plus local residual | post-hoc neutrality does not decide trained-in value; exact A/B owed |
| Movable | birth-aware sparse sites/contours with area and completion forces | current sparse-site payload is 6,289 B; coverage/topology remains the enemy |
| MyCar | majority/static core plus measured boundary XOR if it pays | exact encode/decode and non-derivability owed |
| paint | reuse v7.5.3 frame1 exact-D T after geometry freezes | no blanket region grating; waterfill by measured scorer gain/byte |
| Pose | selected complete `xi` artifact, exogenous to class fields | fixed xi does not by itself make arbitrary RGB composition Pose-safe |

Do not call arbitrary neural scalar fields a Laguerre/power diagram unless the decoder actually has
weighted-site affine/quadratic structure. “Tropical class-potential argmax” is the honest general
name.

### 11.3 v8 stage graph

1. Train each class-potential block against frozen competitor snapshots with SDF/birth/area/topology
   and temporal constraints; preserve per-class checkpoints.
2. Decode all K potentials, compose `argmax`, and compare against an exact matched shared-control
   artifact. Report every class, component birth, junction, tie flicker, `d_pose`, and bytes.
3. Parse back the gauge/integrability receipt. Reject nonzero decoded cycle residual.
4. Freeze geometry and apply the same frame1 post-RGB exact-D paint stage as v7.5.3.
5. Allocate residual bytes across class/edge/paint/pose sections by realized marginal score per byte,
   with pairwise non-derivability and unique ownership.
6. Run the generalized exact integer ratchet, first on pair/class-local coordinates and only later on
   global weights if a bounded screen shows slack.
7. Select among complete receiver-closed artifacts; perform final byte-close/axis receipts.

### 11.4 Current v8 blockers  # MAGNITUDE_DISMISSAL_OK: items below are STRUCTURAL receiver-closure/completeness defects (receiver ignores head; charged-state not minimal; incomplete carrier sets are not vehicles), not eyeball-magnitude dismissals — no ΔS was judged small; verdict_scope: instance per item

1. Current class heads share a trainable code, so `d phi_c / d code` couples every class even though
   `w_out[c]` blocks are separate. Existing block tests perturb only output weights and miss this.
2. The tied-equivalence test makes all fields identical and selects class 0; it does not reproduce a
   real shared-head output.
3. The deployed receiver ignores the decoupled head and still decodes shared `out_sdf`.
4. The ON artifact retains/charges the unused shared head; charged-state ownership is not minimal.
5. The prose kill inequality has the opposite direction from the evaluator predicate.
6. Gate rows can be satisfied by naked numbers: hashes, exact artifacts, axis, and control receipts
   are absent.
7. “Byte closed” is a boolean rather than a parse-back manifest; some analytic smoke carriers are
   target-derived in memory, not encoded.
8. Aggregate `d_seg` omits required per-class birth/topology/junction/tie-flicker noninferiority.
9. Current rate tables mix complete and dominant-only carrier sets; low incomplete totals are not
   shippable vehicles.
10. Default widths/depths and reported seed “spread” are not measured specifications.

## 12. Recursive/fractal optimizer across every scale

“Fractal optimal” should mean the same legality and exact-admission law repeats at every scale, not
that every knob is simultaneously activated.

| Scale | State optimized | Exact invariant | PR128-derived move |
|---|---|---|---|
| coordinate | one quantized code | valid native range and receiver parse | `+-1/+-2`, strict `Delta S<0` |
| local block | pair/class/edge/2x2 kernel | support, exact-D, gauge, topology | diagonal/conflict-colored batch, exact set recheck |
| tensor/carrier | weights, palette, xi, T, field coefficients | unique home, no dead charged bytes | slack screen then native-grid ratchet |
| section | G/xi/T/fill/residual/coder | encode/decode conservation and non-derivability | sidecar fold/remove only if exact net gain |
| artifact | checkpoint + receiver + archive | complete compatible state; reproducible hash | bank every acceptance atomically |
| config | stage graph and active levers | typed DSL, event/resume state, matched control | choose conditional branch by receipts |
| vehicle | v7.5.3 or v8 | full Seg + nonlinear Pose + exact bytes | select complete artifacts only |
| campaign | public/internal frontier | axis-separated promotion and custody | no pointer move without official exact row |

The continuous stage energy may contain evaluator-aligned terms such as

\[
E=E_{\rm margin}+\lambda_AE_{\rm area}+\lambda_EE_{\rm eikonal}
+\lambda_HE_{\rm topology}+\lambda_TE_{\rm screw}
+\lambda_PE_{\rm pose}+\beta\widehat R,
\]

but it is a basin-finding surrogate. Stage-boundary weights and controllers produce a quantized
candidate; the exact discrete objective `J` decides whether it ships. KKT/waterfill allocation uses
measured one-sided marginals:

\[
-\frac{\partial (100d_{\rm seg}+\sqrt{10d_{\rm pose}})}{\partial b_i}
\gtrless c_B
\]

with section `i` admitted only when the realized benefit of its next byte exceeds `c_B` after
interactions. Never add local scorer gains as if independent.

## 13. Primary-source and OSS reuse ledger

| Primary source / official repository | Research or implementation signal | License/reuse disposition | Pact use |
|---|---|---|---|
| [PR128 submission](https://github.com/a12dongithub/comma_video_compression_challenge/tree/submission/rhnerv_latent_polish/submissions/rhnerv_latent_polish) | ctx coder, deterministic packer, FEC6 receiver path, polished payload and exact-gate design prose | submission tree is MIT under the contest repository; retain full MIT and lineage notices; its license cannot cure separately imported unlicensed material | source may be reused with notices after review, but the absent search implementation must be built independently |
| [HNeRV paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html) / [official repo](https://github.com/haochen-rye/HNeRV) | content-adaptive latent embeddings and decoder-capacity argument | paper is research authority; official repo has no root license or license API result, so code/checkpoints/assets are **DO-NOT-COPY** | cite the method and independently implement only if a new HNeRV control is needed |
| [NeRV paper](https://proceedings.neurips.cc/paper/2021/hash/b44182379bf9fae976e6ae5996e13cd8-Abstract.html) / [official repo](https://github.com/haochen-rye/NeRV) | image-wise neural video decoder and model-compression framing | official repo likewise has no declared license; do not copy source/checkpoints/data | research comparator only; independent implementation if needed |
| [HiNeRV](https://hmkx.github.io/hinerv/) / [official repo](https://github.com/hmkx/HiNeRV) | hierarchical positional encodings, patch/frame decoding, prune/quantize pipeline | MIT; retain license/copyright notices and audit acknowledged dependencies | distinct licensed family control; do not relabel PR128 or transfer PSNR claims |
| [COIN++](https://openreview.net/forum?id=NXB0rEM2Tq) / [official repo](https://github.com/EmilienDupont/coinpp) | frozen shared INR plus small quantized/entropy-coded per-signal modulations | MIT; retain notice and separately audit data/checkpoints | strong licensed pattern for shared generic dictionaries plus tiny pair/class modulations |
| [C3](https://c3-neural-compression.github.io/) / [official repo](https://github.com/google-deepmind/c3_neural_compression) | per-signal multiresolution latents, small entropy/synthesis networks, soft-rounding and non-uniform quantization-noise controls | software Apache-2.0; other repository materials CC-BY-4.0; retain license/NOTICE/attribution and audit datasets | licensed texture/carrier-rate control; constrain to exact-D and exact scorer/archive acceptance |
| [Cool-Chic](https://github.com/Orange-OpenSource/Cool-Chic) | overfit multiresolution latent grids and low-complexity entropy-coded decoder | BSD-3-Clause; retain notice/disclaimer | licensed alternative T/carrier grid; isolate from G and measure actual archive bytes |
| [SIREN](https://github.com/vsitzmann/siren) | sine initialization, differential operators, SDF/Poisson reference machinery | MIT, copyright Vincent Sitzmann; retain notice | licensed candidate for an isolated smooth SDF/basis control, not presumed score-positive |
| [WIRE](https://github.com/vishwa91/wire) | localized complex/real Gabor INR modules | MIT, copyright Vishwanath Saragadam; retain notice | licensed candidate for localized annular/edge texture versus the current global Fourier bank |
| [constriction](https://github.com/bamler-lab/constriction) | queue range/ANS primitives and categorical models used by PR128 | project offers MIT, Apache-2.0, or BSL-1.0 choice; prefer MIT for source reuse and audit bundled `LICENSE.html` if distributing wheels | lawful real-coder backend candidate; pin exact version/runtime and keep exact packer receipts |
| [OR-Tools](https://github.com/google/or-tools) | CP-SAT/MIP machinery for bounded integer set selection | Apache-2.0; retain license/notices, mark modifications, honor NOTICE | optional offline selector over pre-measured moves; never receiver dependency or score authority |
| [scikit-fmm](https://github.com/scikit-fmm/scikit-fmm) | fast marching and signed-distance/Eikonal reinitialization | BSD-3-Clause; retain notice/disclaimer and no-endorsement clause | offline SDF initialization/reinitialization reference, not a differentiable-training claim |
| [scikit-image](https://github.com/scikit-image/scikit-image) | Chan-Vese and active-contour reference implementations | mixed/file-specific BSD-3/BSD-2/MIT; inspect the exact file before copying | prefer dependency/reference use; do not flatten mixed licenses into one assertion |
| [PyDEC](https://github.com/hirani/pydec) | discrete exterior derivative, Hodge star, codifferential, Hodge decomposition | BSD-3-Clause; retain notice/disclaimer | offline v8 edge-flow `gradient + coexact/curl + harmonic` diagnostic; learned-field transfer needs an empirical projection gate |
| [Ballé entropy bottleneck](https://openreview.net/forum?id=rJxdQ3jeg), [scale hyperprior](https://openreview.net/forum?id=rkcQFMZRb), [CompressAI](https://github.com/InterDigitalInc/CompressAI) | QAT/additive-noise rate surrogates, entropy priors, reference coders | CompressAI BSD-3-Clause-Clear; inspect exact components/checkpoint/third-party licenses | training-time rate control only; exact candidate bytes remain the admission authority |
| [Entropy-constrained neural video representations](https://openaccess.thecvf.com/content/CVPR2023/html/Gomes_Video_Compression_With_Entropy-Constrained_Neural_Representations_CVPR_2023_paper.html) | joint distortion, quantization, and estimated-rate training rather than purely post-hoc compression | paper authority; no official licensed implementation found in this audit | derive the loss independently, then retain real packer bytes as verdict authority |
| [NVRC](https://arxiv.org/abs/2409.07414) / [official repo](https://github.com/hmkx/NVRC) | quantized INR weights/latents/entropy parameters and end-to-end RD hierarchy | MIT; current README still lists result/evaluation-bitstream/YUV apparatus TODOs | reusable implementation ideas, not receiver-closed replay authority |
| [NVRC++](https://arxiv.org/abs/2606.28163) | multiple high-resolution feature grids, grid entropy model, scalable decoder complexity | paper has no official licensed code link at audit time | research concept only until a licensed source appears |
| [VINRB/WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Gwilliam_How_to_Design_and_Train_Your_Implicit_Neural_Representation_for_WACV_2026_paper.html) / [repo](https://github.com/mgwillia/vinrb) | controlled INR component/equal-wall-time comparisons; public warning that hybrid-INR compression path can zero unsaved modules | repository has no declared license; do not copy | research/apparatus authority: reinforces parse-back and receiver-closed evaluation |
| [HodgeRank](https://arxiv.org/abs/0811.1067) | graph-edge flows decomposed into globally consistent gradient plus cyclic residuals | paper authority; PyDEC supplies the licensed implementation reference above | mathematical support for v8 integrability/cycle-debt gate |
| [DeepSDF](https://github.com/facebookresearch/DeepSDF) | compact latent-conditioned continuous signed-distance/zero-level-set representation | MIT; repository is archived, so pin any reuse and retain notice | licensed SDF reference/control, not evidence that its 3-D reconstruction loss transfers |
| [Tropical geometry of ReLU networks](https://proceedings.mlr.press/v80/zhang18i.html) | ReLU maps as tropical rational maps and piecewise-linear complexes | paper-level mathematics | supports honest tropical-argmax language; does not prove a Laguerre/site parameterization |
| [Osher-Sethian level sets](https://math.berkeley.edu/~sethian/Papers/sethian.osher.88.pdf) / [fast marching](https://pmc.ncbi.nlm.nih.gov/articles/PMC39986/) | implicit fronts, topology changes, Eikonal distance construction | mathematical authority; paper access is not code-reuse permission | supports SDF/topology priors and offline reinitialization, never scorer authority |

The official HNeRV/NeRV repositories being public does not make them OSS. Conversely, the contest
PR chain is MIT-governed in-tree; preserve its `THIRD_PARTY_NOTICES` and do not introduce a new copy
from the unlicensed official repositories. Model weights and datasets require their own license and
custody review even when the surrounding code is permissively licensed.

Useful licensed reuse sequence:

1. use `constriction` or a deterministic equivalent only behind a pinned, byte-exact packer contract;
2. test COIN++-style small modulations before duplicating trunks, with every modulation charged;
3. compare SIREN, WIRE, HiNeRV, and NVRC-style grids only as isolated controls after the receiver
   path exists;
4. use scikit-fmm only for offline SDF initialization/reinitialization;
5. use PyDEC to measure decoded v8 edge curl/cycle debt before deciding whether an edge stream is
   admissible;
6. use OR-Tools only to choose among already measured discrete moves when interactions make greedy
   selection inadequate, or to solve bounded 2x2 camera preimages offline; emit no solver runtime and
   always exact-reencode/re-score plus fresh PyTorch-fp32/uint8 replay.

No paper's PSNR, reconstruction, topology, or compression result transfers as a Pact score claim.
Every imported technique still needs the proof matrix in section 14.

One current public-apparatus warning is directly relevant: VINRB's official README says hybrid-INR
compression results are unreliable in its present path because `set_zero()` can zero model parts that
are not saved, producing noise; bypassing that behavior is itself cautioned pending refactor. That is
not evidence against HNeRV as a family. It is independent evidence for Pact's rule that a compressed
result has no authority until the exact saved state is parsed back and the real receiver is scored.

The highest-EV existing-line synthesis from the online pass is therefore:

```text
entropy-constrained/QAT basin finding
 -> receiver-closed integer quantization
 -> PR128-style exact native-code ratchet
 -> actual coder + archive parse-back

v7.5.3 T candidate: constrained WIRE/C3/Cool-Chic coefficients in the exact-D home
v8 G candidate: gauge-fixed integrable class potentials with small COIN++-style modulations
```

This is a ranked design hypothesis, not a launch disposition. It inherits every blocker and proof
gate already stated above.

Licensing rule: a paper/repository may be authoritative research even when its code cannot be copied.
GitHub's official licensing guidance states that absent a license, default copyright applies; public
visibility permits viewing/forking through GitHub, not general reproduction/modification/distribution.
For an unlicensed source, retain the citation and independently derive/implement the mathematics, or
obtain permission. For licensed OSS, preserve copyright/license notices, satisfy attribution/source/
patent obligations, and separately verify model-weight/data licenses.

## 14. Smallest convincing proof matrix

1. **Custody:** source tree, LFS OIDs, packer, member, archive, release, and official-eval hashes bind.
2. **Receiver parity:** MLX, deterministic NumPy fp32 authority, and fresh-process inflate produce
   identical bytes for every enabled section.
3. **Payload conservation:** every charged byte is consumed exactly once; every video-derived value is
   charged; every deterministic generic bank is regenerated.
4. **Cardinality/runtime:** exactly 1,200 frames/3,662,409,600 bytes; decode stays within the contest
   budget on the named hardware.
5. **v7.5.3 exact-D:** post-nonlinearity kernel rank, bounded integer preimage, raw reload, and Pose
   preprocessing identity all pass.
6. **v8 isolation:** off-block Jacobians include weights, biases, codes, temporal state, and every
   optimizer-visible tensor; decoded edge-cycle residual is zero.
7. **Topology:** per-class births/components/junctions/turnover/tie flicker and rare-class recall have
   explicit noninferiority gates.
8. **Resume:** an interrupted stage and interrupted discrete search reproduce the next loss/proposal/
   accepted archive; all stage EMA checkpoints remain preserved.
9. **Matched science:** control and treatment compile from typed configs and match exact archive bytes
   within a preregistered tolerance; paired seeds/temporal block bootstrap decide formulation scope.
10. **Final admission:** exact

\[
\Delta S=100(d'_{\rm seg}-d_{\rm seg})
+\sqrt{10d'_{\rm pose}}-\sqrt{10d_{\rm pose}}
+c_B(B'-B)<0
\]

on the exact evaluated archive, with axis-tagged uncertainty and no inferred equivalence.

## 15. Exact remaining blockers

### PR128/public candidate

- maintainer-triggered official evaluation on current exact head/archive;
- coherent PR title/body/report/release tag/current asset;
- search code, pure-fold baseline, accepted/rejected ledger, and resume receipts;
- pinned dependency/runtime custody and portable deterministic decode or explicit axis-specific hash
  receipts;
- fail-closed range-stream/schema integrity and atomic cardinality-checked output;
- explicit new-contribution authorship/licensing clarification.

### v7.5.3

- one receiver-consumed T implementation across MLX/NumPy/inflate;
- exact-D frame1-only formulation, no palette gauge duplication;
- deterministic-bank exclusion and exact charged-coefficient grammar;
- texture event/separate optimizer/resume state;
- A1/A2/A3 exact-byte controls and stationary/advected plus stopgrad/coupled probes;
- Pose-inclusive, bit-resumable exact discrete finisher;
- complete-artifact pose selector and exact axes.

### v8

- genuinely disjoint class trainable state, not shared code;
- gauge-fixed K-potential or projected integrable edge grammar;
- receiver consumption and removal of dead charged shared-head state;
- corrected receipt-bound kill predicate;
- exact encoded carrier manifests and complete rate accounting;
- per-class/topology/temporal gates;
- v7.5.3 exact-D paint reuse after geometry closure;
- complete-artifact selection and exact axes.

### Shared apparatus

- no new launch was made by this advisory;
- owed16v2 completed naturally and was harvested by its existing owner; this advisory never signaled
  it. Its result is a rebalanced-warm-start formulation NO-GO, not a from-scratch/family verdict;
- the shared v7.5.2 pilot was launched by its existing owner while this advisory was active and was
  live at the `2026-07-10T15:45:31Z` refresh. This advisory neither launched nor signaled it;
- `clickpolish-build` concurrently owns the untracked `src/tac/click_polish.py` and
  `tools/click_polish_exact_search.py`. This advisory did not absorb, stage, modify, or certify that
  implementation; receiver/Pose/coder/resume proof remains required after its owner lands it;
- PR128 remains external/open-claim evidence and does not move the canonical frontier pointer;
- implementation work should land only after current shared-tree ownership is rechecked.

## Stores consulted

- full Pact preflight: `CLAUDE.md`, `AGENTS.md`, top-10 Claude memory, lane/conflict/directive and
  canonical frontier/dispatch/task surfaces;
- `.omx/research/ADVISORY_RESTART_HANDOFF_v752_v753_v8_20260710.md` and its three completed first-pass
  advisories;
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`;
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`;
- `.omx/research/fullstack_fractal_optimal_synthesis_20260710.md`;
- `.omx/research/t5_crucible2/ADVISORY_v753_texture_trunk_fresh_eyes_20260710.md`;
- `.omx/research/owed16v2_verdict_20260710.json` and the final shared process/ownership refresh;
- concurrently landed `.omx/research/pr128_intake_reverse_engineering_20260710.md` and
  `.omx/research/clickpolish_to_witness_design_20260710.md`, reconciled adversarially in section 7.4;
- current v7.5.3/v8 DSL, generator, texture, decoder, byte-close, finisher, and test surfaces;
- PR128 PR metadata, commits, release objects, fork branches, author-repository search, and public
  discussion through GitHub primary surfaces;
- detached PR128 head and all 17 submission files, hydrated LFS inputs, exact release archive, and
  PR112 comparator archive on `/Volumes/VertigoDataTier`;
- official HNeRV/NeRV paper/project sources and the primary-source/OSS license ledger below;
- GitHub's official repository-licensing guidance.

**Pointer delta:** none. **Launch delta:** none. **Run signals:** none. **Official score delta:** none.
**Owned repository write:** this advisory only.
