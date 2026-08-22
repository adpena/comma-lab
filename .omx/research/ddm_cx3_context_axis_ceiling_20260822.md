# ddm_cx3 context-axis ceiling: the tested model axis supplies 0 B, not 42,382 B

**MEASURED `[macOS-CPU advisory / scorer-free lossless diagnostic]`, n600:** the best
model-cost-inclusive challenger in the named conditional-entropy ladder is **125,210 B**, which is
**11,433 B worse** than the shipped **113,777 B** RC64 stream. Even its hindsight ideal data term is
**117,224 B**, already **3,447 B worse before model cost**. The context axis therefore supplies
**0 of the required 42,382 B in this formulation**. The preregistered 71,395 B falsifier did not
fire, and the borrowed-group refit gate did not open.

This is a `FORMULATION`-scoped negative: the tested causal token summaries and retained learned-
predictor summaries on the pinned DX2 field. It does not kill a differently trained HPAC network,
a new learned representation, or a model that consumes the continuous five-class probability
vector rather than the named summaries.

## Inherited state and refusal pins

All four charter pins reproduced before measurement. The TO2 decoded field was reused; it was not
decoded again.

| object | bytes | SHA-256 | status |
|---|---:|---|---|
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | MATCH |
| decoded token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | MATCH |
| RC64 token stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | MATCH |
| token checkpoint receipt | 3,511 | `c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9` | MATCH |

The learned-signal rows also consume FS2's retained predictor trace on the identical `cc10...`
token field. FS2's pre-corrector HPAC logit and CDF digests equal DX2's retained decoder receipt.
The pinned trace objects are:

| object | bytes | SHA-256 |
|---|---:|---|
| FS2 replay receipt | 3,014 | `edf65f114cd01d468109a42bccba284b96365cd1aac77ad609cc5dfaddec8863` |
| corrected coding-row argmax field | 117,964,928 | `93cdf71daedd39505c5031aca7cf8524a6358fc862ce838acfbcc1cc73dcae33` |
| confidence-index field | 235,929,728 | `74470f44a5333b27b131fcd0cf5d17fd41d82cc219d5fbd1b0557feb8825295f` |

Scope boundary: those two arrays predate DX2's final 70 B corrector improvement. Their underlying
HPAC logit/CDF digests and token field are identical to DX2, so they are valid conditioning
coordinates. They are not claimed to reproduce DX2's final continuous probability row.

## The incumbent context law

The archive contains a **13,515 B counted Brotli-coded HPAC blob** whose decoded model object is
17,952 B, followed by the **113,777 B RC64 stream**. The combined counted token subsystem is
127,292 B.

Denominator for every bits/symbol figure is exactly **117,964,800 five-class symbols**:

| counted object | bytes | realized bits/symbol |
|---|---:|---:|
| RC64 token stream | 113,777 | 0.007715996636 |
| HPAC model blob | 13,515 | 0.000916544596 |
| token stream + HPAC blob | 127,292 | 0.008632541233 |

Decode is frame outermost, then `g=0..189`, then stable raster positions inside each group, where
`g=(x mod 64)+2*(y mod 64)`. The map collapses 4,096 tile-phase cells onto 190 groups: 3,906 phase
collisions. Across the 384x512 plane, a group contains 48 to 1,536 sites.

The base HPAC conditions on a learned frame-index embedding, the complete previous token frame
through a learned 3x3 convolution, already-decoded current-frame groups through masked 7x7 plus
dilated 5x5 and 3x3 patch-local convolutions, fixed patch coordinates, learned frame scale, and SPM.
The 19 adaptive hit-event members are:

| member | conditioning set |
|---|---|
| `shipped_joint` | predicted class x surprise x t-1/t-2 agreement x run x previous-frame boundary distance |
| `temporal_spatial` | predicted class x t-1/t-2 agreement x causal left/up agreement |
| `surprise_only` | predicted class x surprise |
| `spatial_surprise` | predicted class x causal left/up agreement x surprise |
| `spatial_boundary` | predicted class x causal left/up agreement x boundary distance |
| `run_surprise` | predicted class x run x surprise |
| `boundary_surprise` | predicted class x boundary distance x surprise |
| `temporal_surprise` | predicted class x t-1/t-2 agreement x surprise |
| `shipped_fast256` | `shipped_joint`, counts halved above 256 |
| `shipped_fast4096` | `shipped_joint`, counts halved above 4,096 |
| `surprise_fast256` | `surprise_only`, counts halved above 256 |
| `spatial4_surprise` | predicted class x four-neighbour agreement x surprise |
| `homog_surprise` | predicted class x distinct causal-neighbour classes x surprise |
| `homog_boundary_surprise` | predicted class x homogeneity x boundary x surprise |
| `spatial4_boundary` | predicted class x four-neighbour agreement x boundary |
| `homog_spatial4` | predicted class x homogeneity x four-neighbour agreement |
| `spatial4_temporal` | predicted class x t-1/t-2 agreement x four-neighbour agreement |
| `homog_surprise_fast256` | `homog_surprise`, counts halved above 256 |
| `spatial4_surprise_fast256` | `spatial4_surprise`, counts halved above 256 |

Their fixed-point logistic mixer has 4,000 online weight cells keyed by predicted class, boundary
bucket, t-1/t-2 agreement, neighbour homogeneity, and an eight-way surprise bucket. MA1 adds the
within-miss relative law keyed by causal up, up-right, left, and previous-frame token, 1,296 cells.
All adaptive state is rebuilt causally and transmits zero bytes.

## Conditional-entropy ladder

Each row sees all **117,964,800 symbols**. `Ideal data` is hindsight empirical conditional entropy,
rounded up to bytes. `Static model` is the smallest exact retained sparse count-model payload among
Brotli q11, LZMA1, and zlib9; every variant and deterministic repeat is retained and parse-backed.
`Static total` is ideal data plus that counted model. `KT` is the exact Dirichlet-1/2 prequential
ideal length with zero transmitted table. `Best ideal total` is `min(static total, KT)`.

None of these ideal lengths is a finite-precision token payload, a shippable size, or a universal
lower bound. They are optimistic diagnostics for the exact conditioning set named in the row.

| conditioning set | active contexts | ideal data B | static model B | static total B | KT B | best ideal total B | net vs 113,777 B |
|---|---:|---:|---:|---:|---:|---:|---:|
| none, order 0 | 1 | 23,822,019 | 39 | 23,822,058 | 23,822,026 | 23,822,026 | +23,708,249 |
| previous token in frame/group/raster event order | 6 | 4,480,491 | 105 | 4,480,596 | 4,480,521 | 4,480,521 | +4,366,744 |
| inherited group identity | 190 | 23,511,425 | 1,943 | 23,513,368 | 23,512,286 | 23,512,286 | +23,398,509 |
| causal left/up, radius 1 | 36 | 291,231 | 306 | 291,537 | 291,354 | 291,354 | +177,577 |
| causal left/up, radii 1 and 2 | 505 | 284,598 | 1,621 | 286,219 | 285,334 | 285,334 | +171,557 |
| same-site t-1 | 6 | 1,272,195 | 108 | 1,272,303 | 1,272,227 | 1,272,227 | +1,158,450 |
| same-site t-1/t-2 | 31 | 1,061,990 | 303 | 1,062,293 | 1,062,098 | 1,062,098 | +948,321 |
| causal left/up x t-1 | 187 | 243,536 | 889 | 244,425 | 243,937 | 243,937 | +130,160 |
| causal left/up x t-1/t-2 | 642 | 230,679 | 2,160 | 232,839 | 231,679 | 231,679 | +117,902 |
| inherited group x t-1 | 1,140 | 1,241,877 | 6,058 | 1,247,935 | 1,245,769 | 1,245,769 | +1,131,992 |
| inherited group x causal left/up x t-1 | 14,546 | 237,099 | 33,030 | 270,129 | 256,635 | 256,635 | +142,858 |
| causal left/up/up-right/up-left x t-1/t-2 | 2,710 | 208,232 | 6,423 | 214,655 | 210,908 | 210,908 | +97,131 |
| learned corrected-row argmax | 5 | 275,185 | 84 | 275,269 | 275,213 | 275,213 | +161,436 |
| learned argmax x 64 half-bit confidence bins | 293 | 138,657 | 1,488 | 140,145 | 139,716 | 139,716 | +25,939 |
| inherited group x learned argmax/confidence | 48,395 | 131,005 | 99,672 | 230,677 | 227,358 | 227,358 | +113,581 |
| **learned argmax/confidence x causal left/up x t-1** | **5,884** | **117,224** | **14,696** | **131,920** | **125,210** | **125,210** | **+11,433** |
| learned argmax/confidence x spatial4 x homogeneity x t-1/t-2 agreement x run x boundary | 60,352 | 121,951 | 102,923 | 224,874 | 183,417 | 183,417 | +69,640 |
| exact raster site x t-1/t-2 | 817,257 | 680,837 | 239,287 | 920,124 | 1,410,223 | 920,124 | +806,347 |

The best row's exact static model is 37,687 B raw and **14,696 B Brotli q11**, sha
`ba6e6246575a5e8cdf133b18a07cfb531ca56831164f5442fc54fe87aca0a174`; its repeat is byte-identical.
Its ideal data term is 0.007949748578 bits/symbol and its KT term is 0.008491321990 bits/symbol.

## Ceiling versus the 42,382 B demand

There are two honest answers, depending on whether “achieved” or “ideal diagnostic” is meant:

- **Best achieved lossless stream:** the incumbent, 113,777 B. No challenger was admitted.
- **Best measured optimistic challenger:** 125,210 B KT, zero transmitted new table, while still
  requiring the already-counted 13,515 B HPAC predictor. It is 11,433 B worse than the incumbent.
- **Best hindsight static alternative:** 117,224 B ideal data + 14,696 B exact model = 131,920 B,
  18,143 B worse than the incumbent.

The goal requires the token member to reach **at most 71,395 B** at unchanged other sections. The
best optimistic challenger misses that ceiling by **53,815 B**. Expressed against the required
42,382 B cut, the measured “saving” is **-11,433 B, or -26.98% of demand**. In useful campaign
language: this formulation supplies **0 B** and moves in the wrong direction.

The prior-law prediction's route consequence is supported: the named model axis supplies less than
17,000 B, indeed no positive byte. Its broader claim that the incumbent is within 15% of the floor
of every affordable context model is not proven; the rows do not enumerate all learned models.
The registered falsifier, model-cost-inclusive size at or below 71,395 B, is false here.

## Borrowed constant gate

The preregistered materiality bar was a base-ladder saving of at least **4,239 B**, 10% of demand.
The measured base best was **-11,433 B**, so the gate did not fire. No group partition was refit.

This refusal is strengthened, but not promoted into a partition-family kill, by the two fixed-map
controls: group identity alone costs 23,512,286 B KT, and adding group identity to learned
argmax/confidence worsens KT from 139,716 B to 227,358 B. Those are conditioning-label results with
the predictor trace fixed; they do not simulate an HPAC forward under a changed causal traversal.

## Candidate and authority boundaries

- No finite-precision challenger token stream was built. The ideal rows already lost, so treating
  one as a candidate would be a fake size claim.
- Consequently no challenger entered the exact-inversion admission gate. The pinned source field
  remained the measurement target; all static model payloads themselves parse back exactly.
- No receiver or archive was edited. No scorer, Metal, Modal, upstream evaluator, or jo1-r9 action
  ran. Losslessness is the only axis considered.
- The result did not move d_seg, d_pose, archive bytes, score, or the canonical pointer.
- Verdict scope is `FORMULATION`: these named causal summaries on this pinned DX2 token field.

## RECALL EVIDENCE

I searched `.omx/research/`, arm receipts, canonical indexes/DAG feeds, design/spec surfaces, the
task ledger, and active lane claims by content using: `conditional entropy`, `context order`,
`context law`, `HPAC`, `RC64`, `group_index`, `deep_context_ladder`, `corrected_cdf_input`,
`probability model`, `mixer context`, and `within miss`. I also ran
`tools/list_canonical_equations.py --json` and searched the returned registry for this surface.
I did not find a duplicate active CX3 lane or a canonical equation that supersedes the measured
comparison in those scopes.

Beyond the charter's seed list, recall found:

- `ddm_dc1_decode_budget_conditional_coding_20260816.md`: an older-object table-free spatial/
  temporal ladder whose KT curve turns around; its later correction narrows its closure to coder
  swap at fixed probabilities.
- `ddm_hm1_model_byte_derivative_20260816.md`: a retained decoder-digest-gated learned-logit asset
  and the warning that hand-designed features do not summarize the five-vector output.
- FS2's exact replay on the **same `cc10...` token field**, with HPAC logit/CDF digests equal to DX2.
- FX1/FX2/FX5 and MA1: the actual fixed-point mixer, 19 member pool, causal feature definitions,
  and within-miss law; those sources corrected the incumbent characterization and prevented a
  token-only ladder from being mislabeled as the whole axis.
- RR1's older free-model augmentation ladder and the later narrowing of broad “coder closed”
  language to probability-model versus wire-coder factors.

This changed the plan materially. The first complete token-only pass (`measurement_v2`) had a best
ideal total of 210,908 B, but it omitted the learned predictor and therefore established only a
token-summary negative. The authoritative v4 pass added the matching learned argmax/confidence
coordinates, retained static model costs, and the zero-table KT comparator. The learned rows improve
the diagnostic substantially to 125,210 B, but still do not beat the incumbent.

## Retention and execution receipts

All new receipts are on the required Vertigo tier under
`/Volumes/VertigoDataTier/pact/ddm_cx3_context_axis_ceiling/`; no new receipt was written to
APDataStore.

- Authoritative result: `measurement_v4/RESULT.json`, 190,323 B, sha
  `6f3db0b75e4ec02da7daa0a5cabd2ac8bd9fe5f6e8eeb5479a5d701cdf8a2fa3`.
- Manifest: `measurement_v4/MANIFEST.json`, 116,122 B, sha
  `49e560f44cda96bbf868b68f055436d0f4f337cf70ab113eadc8d740d218d033`.
- Manifest population: **430 artifacts, 2,250,295,879 B**. A separate completed-resume pass
  rehashed and verified every listed receipt.
- v1 remains retained: two rows; immutable-state rewrite refusal exposed before a conclusion.
- v2 remains retained: thirteen token-only rows; complete but scoped below the final question.
- v3 remains retained: sixteen rows; a float-promoted context ID refused before the richest row.
- v4 is authoritative: eighteen rows; script sha
  `e0a27018d4c2baf4a7e38c20ec2d40c87d6a6c4954d190b81e97c1bb770c0bc3`.

No retained payload was deleted or overwritten. Stage checkpoints, dense and active histograms, raw
models, all three coded model variants, and deterministic repeats remain present.

## Follow-on dispositions

- **FOLDED:** borrowed group-partition refit. Owner: CX3. Consumer store: this memo and
  `measurement_v4/RESULT.json`. Fire trigger was at least 4,239 B of base-ladder saving; measured
  saving was -11,433 B.
- **FOLDED:** receiver integration, byte-close, scorer, and exact-eval chain. Owner: MAIN. Consumer
  store: `.omx/state/main_hot_state.md`. Fire trigger was a retained exact-invertible stream below
  113,777 B; no candidate stream exists.
- **QUEUED-WITH-A-FIRE-ORDER:** a genuinely new learned HPAC representation/capacity allocation,
  not another hand-summary table. Owner: MAIN must assign a non-duplicate model lane. Consumer
  store: `/Volumes/VertigoDataTier/pact/<claimed-model-lane>/RESULT.json`, then MAIN's hot state.
  Fire integration only when an exact-invertible counted **model + token** subsystem is at most
  123,053 B, at least 4,239 B below the current 127,292 B subsystem; prioritize it for sub-0.12 only
  at 84,910 B combined or 71,395 B token bytes with the model unchanged.

**Own-vehicle frontier: UNMOVED at DX2 S=0.14821987563243377 @ 180,368 B `[contest-CUDA T4 n600]`, archive sha `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.**
