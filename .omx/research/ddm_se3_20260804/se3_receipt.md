# ddm_se3_20260804 SE3 receipt

Status: scorer-free description-side receipt. No SegNet/PoseNet forward, no archive.zip, no full-n600 scorer job.

Baseline from charter/hot state: own-vehicle LIVE BEST `S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`. Borrowed contest pointer remains unmoved.

## Leg A - Road/Lane partition price

Command:

```bash
.venv/bin/python experiments/ddm_se3_edge_partition_price.py --store-best --hash-inputs
```

Inputs:

- GT argmax: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy`, sha256 `b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d`
- Current argmax stand-in: `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy`, sha256 `5e903de650e60ec6a64b34eb455fa1bc911223551d0b31e9ae45cc906e1490be`
- Selection mode: `n600 all pairs; no prefix`
- Total Road/Lane flips: `235,148 / 117,964,800` scorer slots
- Rate exchange: `1.2731082153320312 B/flip`

Primary receipt:

- JSON: `.omx/research/ddm_se3_20260804/se3_edge_partition_price.json`, sha256 `3526a48c590c50caf0bd7c3f77c7ac70a6814fb3699239997630d50de867a1a8`
- Script: `experiments/ddm_se3_edge_partition_price.py`, sha256 `6dc125584171d19c29caf9164cc4a8af21598bffd2d07961e4d293097a9bd0a5`
- SSD payload directory: `/Volumes/VertigoDataTier/pact/ddm_se3_20260804/edge_partition_payloads/`

Best rows, all MEASURED description bytes and DERIVED score arithmetic:

| row | bytes | codec | captured flips | break-even survival | net S if perfect | own S if perfect |
|---|---:|---|---:|---:|---:|---:|
| `road_lane_band_r1_edit_bits_side_implied` | 81,365 | Brotli q11 | 161,660 | 0.395339 | -0.082863 | 0.671117 |
| `road_lane_band_r1_edit_plus_direction_bits` | 100,904 | Brotli q11 | 161,660 | 0.490276 | -0.069853 | 0.684128 |
| `road_lane_band_r3_edit_bits_side_implied` | 110,689 | Brotli q11 | 176,895 | 0.491500 | -0.076252 | 0.677728 |
| ED1 section baseline | 169,149 section / 169,351 archive delta | byte-closed archive section | 191,005 | 0.696430 | -0.049153 | 0.704828 |

Interpretation:

- The 81,365 B side-implied row exactly reproduces the sg3 cheap-address row and beats ED1 on byte price and break-even survival, but only under `ASSUMPTION(receiver-field)`: the receiver must derive the Road/Lane band/side chart from the public generator field.
- The safer explicit-direction row costs 100,904 B. It still beats ED1 on bytes and break-even survival, but it does not close the receiver-field blocker because the band coordinates are still derived from the stand-in current argmax.
- lr2 already measured the simple anchor-paint legal value-realizer at this price point as dead at n32 (`AP_pair` survival -1.0190, `AP_null` +0.0501 vs 0.3956 bar). Therefore the production route is not "paint this band"; it is receiver-field closure or a new legal value-realizer.

Verdict: Leg A description pricing is FIRED and complete for this arm. Adopt the 81,365 B / 100,904 B table as the priced target, but do not promote until receiver-field closure and scorer validation exist.

## Leg B - comma10k micro-student

Adversarial legality ruling:

- ALLOWABLE only if the model weights are trained/frozen from public non-contest data such as comma10k and are not selected, fine-tuned, distilled, or adapted using `upstream/videos/0.mkv`, cached GT argmax, SegNet targets, PoseNet targets, or contest-score feedback.
- If any contest-video-derived table, finetune, correction head, or selected checkpoint enters the weights, those bytes become counted archive payload and cannot be hidden in `inflate.py`.
- If the public model is large but contest-video-invariant, this is `ECONOMIC-unpriced`, not a free score row: it still needs license/custody, deterministic decode budget, and 30-minute runtime evidence.
- It must not ship scorer weights or a SegNet/PoseNet clone trained on contest labels.

Size/quality estimate, INFERRED from literature and model cards:

- comma10k is MIT-licensed public driving segmentation data with 10,000 PNGs and labels including road and lane markings. Source: <https://github.com/commaai/comma10k>
- The public comma10k-segnet card is an EfficientNet-B2 U-Net with 5 output classes. Source: <https://huggingface.co/commaai/comma10k-segnet>
- Real-time segmentation architectures show the compute regime is plausible for a tiny student: BiSeNet V2 reports a bilateral detail/semantic design for real-time segmentation (<https://arxiv.org/abs/2004.02147>), and Fast-SCNN reports high-resolution embedded segmentation with no required large-scale pretraining (<https://arxiv.org/abs/1902.04502>).
- Knowledge-distillation sources support a smaller student imitating a teacher for semantic segmentation, including APD's local-perspective distillation (<https://hub.hku.hk/handle/10722/332255>).

Smallest honest prototype:

1. Run the public comma10k-segnet or a hand-sized public-data-only student on <=32 existing generated frames, no scorer, no training, no contest labels.
2. Compare only the derived Road/Lane band chart against the cx1 stand-in chart: overlap, captured Road/Lane flip eligibility, and whether the same 81 to 101 KB streams can be addressed from receiver-derived coordinates.
3. If chart overlap is poor, FOLD the micro-student as a receiver-field source for SE3. If overlap is good, then train a tiny comma10k-only student with frozen recipe and custody logs; still no contest-video finetune.

Verdict: Leg B is QUEUED-WITH-FIRE-ORDER after Leg A receiver-field closure attempt. Do not spend a big training run before the cheap chart-overlap prototype.

## Leg C - literature routing

| source family | ruling | local consumer |
|---|---|---|
| MPEG-4 binary alpha / CAE shape coding and BCAE | ALREADY-HAVE as a warning and fallback coder family. The literature validates context-coded binary shape masks, but our current band edit bitstreams already use real coders and the live blocker is receiver realization, not entropy theory. Sources: <https://www.sciencedirect.com/science/article/abs/pii/S0923596599000478>, <https://ir.lib.nycu.edu.tw/handle/11536/23821>, <https://researchoutput.ncku.edu.tw/en/publications/block-conditioned-context-based-arithmetic-coding-for-efficient-r/> | Leg A coder fallback only if RF closure passes and the 81 KB target needs shaving. |
| Chain code, digital straightness, polygon boundary codecs | N-A for Road/Lane flip dust. Local sg3 found `112,077` Road/Lane components with median 1 px and per-island 700,394 B, so straight-line boundary coders attack the wrong object. Freeman chain coding remains relevant to smooth semantic lanes, not isolated correction dust. Source: <https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/AD720042.xhtml> | Possible gt2 semantic-lane generator, not SE3 correction payload. |
| Coding for machines | ADOPT as objective discipline. The SE3 route is exactly "send task-relevant structure, not RGB." Source: <https://pubmed.ncbi.nlm.nih.gov/32857694/> | Leg A and Leg B framing. |
| Tiny segmentation and segmentation KD | ADOPT only for the cheap receiver-field prototype. Sources: comma10k, comma10k-segnet, BiSeNet V2, Fast-SCNN, APD links above. | Leg B micro-student chart source. |
| Steganography/watermarking known-channel placement | ADOPT cost-map language, not the full STC implementation. UNIWARD frames embedding as payload under a distortion/cost map with practical near-bound codes, which matches our "where can a bit survive R/SegNet" question. Source: <https://link.springer.com/article/10.1186/1687-417X-2014-1> | se2/SE3 placement maps and survival targeting, after receiver closure. |

## Fire order

FOLDED:

- ED1 as the next description-side price target. It is still a useful byte-closed receiver-consumed baseline, but 169,149 B / 0.696 break-even is dominated by the SE3 band rows until receiver closure fails.
- Chain-code/polygon/digital-straightness as correction-dust coders for Road/Lane flips.
- Anchor-paint realization at the 81 KB band price point, per lr2 n32 survival.

QUEUED-WITH-FIRE-ORDER:

1. `SE3-RF1`: scorer-free receiver-field closure. Derive the Road/Lane band from the live public generator/inflated class field, re-run `experiments/ddm_se3_edge_partition_price.py` with that field replacing `cx1_argmax_n600.npy`, and require an explicit row <=101 KB with captured flips near the current 161,660 before any archive work.
2. If `SE3-RF1` passes, build the smallest legal receiver-consumed archive for the explicit-direction stream first, then append one full-n600 scorer spec to `.omx/research/scorer_batch_20260804.md` behind the existing `sq2` owner slot. Do not fire a scorer job while `sq2` owns the slot.
3. If `SE3-RF1` fails, run the Leg B <=32-pair comma10k chart-overlap prototype. If overlap fails, fold the micro-student route for SE3; if it passes, train only a public-data-only tiny student and repeat RF closure before archive work.

## Verification

- `.venv/bin/python -m py_compile experiments/ddm_se3_edge_partition_price.py` - PASS
- `.venv/bin/python experiments/ddm_se3_edge_partition_price.py --store-best --hash-inputs` - PASS
- `.venv/bin/python tools/review_tracker.py scan` - PASS, script ingested after moving to `experiments/`
- `.venv/bin/python tools/review_tracker.py mark-file experiments/ddm_se3_edge_partition_price.py --status reviewed` - PASS, twice
- `.venv/bin/python tools/review_tracker.py policy-check experiments/ddm_se3_edge_partition_price.py` - PASS, 22 entities compliant, 0 violations

No `/tmp` evidence. SSD evidence is under `/Volumes/VertigoDataTier/pact/ddm_se3_20260804/`.

## NEXT-IF-RESUMED

Start at `SE3-RF1`. Do not reprice sg3/SE3 band rows unless the receiver-derived class field changes. The next new information is whether the receiver can derive the Road/Lane band chart legally from the public generator output; without that, the 81 KB and 101 KB rows are only description-side targets, not score rows.

Own-vehicle frontier line: own-vehicle LIVE BEST remains `S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; SE3 did not move the exact pointer or build an archive.
