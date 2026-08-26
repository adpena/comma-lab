# ddm_jt23 — the coder axis is SPENT, not remaining: 0 B collectible, composition REFUSED at the bar

`date_utc: 2026-08-26` · `arm: ddm_jt23_coder_collection_compose` ·
`axis: [macOS-CPU scorer-free EXACT byte measurement]` · `score_claim: false` ·
`frontier_moved: false` · `verdict_scope: INSTANCE — the gb1/jt21 body's five physical sections
plus ZIP/RX1 framing, raced at exact bytes`

**Verdict: `CODER_AXIS_CLOSED_0B__COMPOSITION_REFUSED_BELOW_BAR`.** No 21st pointer move comes
from this arm. Nothing fires. The jt21 bank stands unchanged.

STORES CONSULTED: `.omx/state/main_hot_state.md` POINTER_LINE + LIVE_PROCESSES ·
`ddm_jt21_joint_21family_reencode_verdict_20260825.md` · `ddm_jt22_mixer_context_race_verdict_20260825.md` ·
`ddm_r012_rate_representation_20260821.md` (the 88 B receipt) ·
`ddm_dx2_cabac_receiver_fold_20260821.md` · `ddm_ar1b_archive_residue_purchase_20260822.md`
(the exact physical census) · `[[dx2-block-ceilings-are-measured-and-sum-to-5-percent]]` ·
retained receipts `S1_encode_gb1_joint21.json` + `S1_control_600.json`.

## ANSWER FIRST

**The 88 B coder ceiling was already collected. It IS the dx2 body.** The charter asked me to
collect "remaining" coder headroom; there is none, and the number that said otherwise is stale.

`rc2 180,456 B → fx5_e1 180,386 B (−70 B token corrector) → dx2 180,368 B (−18 B CABAC
coefficient coder)`. That is `70 + 18 = 88 B`, exactly. r012 projected the full 88 B composition
would land at "180,368 B and S=0.14821987563243377"; dx2 landed at **exactly** those bytes and
that S, to 17 digits. The two rungs r012 named as the composable ceiling are the two rungs dx2
shipped.

I then measured the current body directly rather than inherit any claim. **Every section of the
jt21 candidate is at or below what a real coder achieves. Best available saving: 0 B.**

Composed marginal against the gb1 pointer = jt21 `−23 B` + coder `0 B` = **`−23 B`**, which is
**7 B short of the pre-registered ~30 B solo-fire bar. REFUSED.**

## 1. The measurement — real coders, exact bytes, current body

Two independent races on `candidate_gb1_joint21.zip`
(180,192 B, sha `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3`, verified).

**(a) `tools/audit_archive_coder_axis.py`** — the tool built to extinct exactly this genus
(its docstring: a coder-axis closure "measured once, on the PR130 base ... then cited on cp135,
MC36, e480b and hv1 -- four bases downstream -- without ever being re-measured").

| section | shipped | raw | H0 bound | H1 oracle | H1 adaptive | best race | Δ |
|---|---:|---:|---:|---:|---:|---|---:|
| hpac | 13,515 | 17,952 | 14,961 | 10,487 | 14,550 | brotli-q11 | **+40** |
| semantic | 30,856 | 36,130 | 33,058 | 27,469 | 33,723 | brotli-q11 | **+0** |
| carrier | 22,010 | 22,008 | 21,985 | 16,813 | 22,401 | lzma2 | **+2** |
| token_stream | 113,601 | — | — | — | — | brotli-q11 | **+5** |
| residual_table | 96 | — | — | — | — | not raced | fixed layout |

`VERDICT: coder_axis_closed (best available saving 0 B)`.

**(b) Independent wider sweep — 25 coder configurations per section, raced on the RAW payloads**
(bz2-9 · zlib-9 · six LZMA2 lc/lp/pb variants · LZMA1 · four delta+LZMA2 distances · brotli q11
at lgwin 20/22/24 · brotli TEXT and FONT modes):

| section | raw | shipped | best of 25 | Δ |
|---|---:|---:|---|---:|
| hpac | 17,952 | 13,515 | `lzma2_lc1_lp0_pb0` 13,555 | **+40** |
| semantic | 36,130 | 30,856 | `brotli_q11_w20` 30,856 | **+0** |
| carrier | 22,008 | 22,010 | `lzma2_lc0_lp0_pb0` 22,012 | **+2** |

**Collectible: 0 B.** The semantic tie is exact, not approximate: my
`semantic__brotli_q11_lgwin24.bin` has sha `39d1be52ba629334…` — **byte-identical to the shipped
section**. That section literally *is* brotli-q11 output, so it sits precisely on the brotli bound.

⚠ **A correction I caught on myself mid-arm.** My first extended race re-compressed the *shipped*
(already-coded) bytes and reported a tidy "+4 B everywhere". That is a meaningless measurement —
recompressing compressed data. The table above races the RAW payloads, which is the only
comparison that can find headroom. The discarded run is noted so no reader mistakes it for a
result.

## 2. Framing — measured at its floor, not assumed

r012 row 10 cited "packaging slack 210 B total". ar1b's later exact census on dx2 supersedes it:
**114 B**, and I measured the container directly on the jt21 body:

| span | bytes | disposition |
|---|---:|---|
| ZIP local file header | 31 | 30 B fixed + 1 B filename `p` |
| ZIP central directory | 47 | 46 B fixed + 1 B filename |
| ZIP EOCD | 22 | fixed, archive comment length 0 |
| **ZIP container total** | **100** | **at the theoretical floor** |
| RX1 header | 14 | receiver-required grammar, no quantization axis |

The member is `ZIP_STORED`, filename is one character, extra fields 0, comments 0. 31+47+22 = 100 B
is the minimum a valid single-member ZIP can occupy. **0 B collectible.** The 210 B figure never
applied to this body.

## 3. Cross-check — jt21 touches only the token stream

Against ar1b's independently measured dx2 span shas:

| section | jt21 vs dx2 |
|---|---|
| hpac `602115b3…` · semantic `39d1be52…` · carrier `932b979f…` · residual `8ab2fe74…` | **byte-identical** |
| token_stream `4c9dc10c…` (was `e2af55e6…`) | changed, 113,777 → 113,601 B |

The `S1_encode_gb1_joint21.json` receipt confirms `tokens_changed = 0` and
`control.byte_identical = True`, with the control stream sha matching ar1b's dx2 token sha. The
family is lossless by construction. Decode-identity and the gen-21 C port remain **OWED at fire
time** — unchanged, and not claimed here.

## 4. Arithmetic, recomputed from components (#877, never the rounded display)

`d_seg = 0.00020139`, `d_pose = 6.37e-06`, `N = 37,545,489`, exchange `6.658589531221714e-07 S/B`.

| body | bytes | rate | seg | pose | S |
|---|---:|---:|---:|---:|---:|
| dx2 | 180,368 | 0.1200996477 | 0.020139 | 0.0079812280 | 0.14821987563243377 |
| **gb1 POINTER** | **180,215** | 0.1199977712 | 0.020139 | 0.0079812280 | **0.14811799921260607** |
| jt21 + coder(0 B) | 180,192 | 0.1199824565 | 0.020139 | 0.0079812280 | 0.14810268445668429 |

The gb1 row reproduces the hot-state POINTER_LINE to all 17 digits, so the recompute is sound.

**Composed marginal vs gb1: `−23 B` = `ΔS −1.531476e-05`. Bar ~`−30 B`. Shortfall 7 B. REFUSE.**

## 5. What this changes for MAIN — the bank's disposition, not just its number

The jt21 bank was recorded as "rides the next lossless fire." **I have now measured that the
coder axis cannot supply that ride.** It is empty, on this body, at 0 B.

Both the hot-state POINTER_LINE and the block-ceilings memory carry the stale sum:
*"remaining lossless total ~2,097 B (model ~2,009 + coder 88)"*. The coder 88 is spent.
**Corrected: ~2,009 B remaining from the gb1 pointer, all of it model-axis.** As a share of the
42,229 B demand that is **4.8%, not 5.0%** — the gestalt's conclusion is unchanged and slightly
*strengthened*: there is less headroom inside this object than the headline said.

So MAIN faces a genuine binary, with no third option:

1. **Fire jt21 solo at −23 B**, accepting the gen-21 native C port + decode-identity proof + seal
   + ~$0.20 T4 for `ΔS −1.53e-05`; or
2. **Hold it**, and accept it may never ride, because the only remaining lossless supplier is the
   model axis whose per-family marginal is collapsing (`153 → 23 → 1 B`).

I did not lower the pre-registered bar. That bar is MAIN's, and moving it to fit a measurement is
the failure this arm exists to avoid.

## 6. GESTALT-DELTA

The block table's two additive rows are now **one**. The coder row is not headroom — it is
history, collected on 2026-08-21 and visible in the pointer's own byte trail. This is the
`[[corrections_land_in_bodies_headlines_keep_the_stale_number]]` genus caught on the campaign's
most-quoted arithmetic, and it is the second time a coder-axis ceiling has been carried across
bases without re-measurement — precisely what `tools/audit_archive_coder_axis.py` was written to
stop. The tool works; it simply had not been pointed at this body until now.

Sub-0.12 is untouched by any of this. The demand is 42,229 B; the entire remaining lossless
inventory is ~2,009 B of collapsing model-axis marginal. **A different OBJECT remains the only
route.**

## NEXT_IF_RESUMED

| item | disposition |
|---|---|
| coder axis on gb1/jt21 body | **CLOSED at 0 B, measured two ways, 25 configs.** Do not re-enter without a body change. |
| ZIP/RX1 framing 114 B | **CLOSED.** ZIP 100 B is at its structural floor; RX1 14 B is required grammar. |
| jt21 bank (−23 B, sha `ec0dd68f…`) | **BANKED, unchanged.** Fire/hold is MAIN's binary per §5. Decode-identity + gen-21 C port owed at fire. |
| hot-state POINTER_LINE + block-ceilings memory | **OWED a correction**: drop coder 88 from "remaining"; total ~2,009 B, 4.8% of demand. MAIN owns both surfaces. |
| residual_table 96 B | CLOSED — 0 B in fixed layout (ar1b); a storage-layout change is out of scope. |
| model axis ~2,009 B | Open but collapsing (153 → 23 → 1 B). Not this arm's axis. |

Retained payloads (22 files, 807,870 B) with per-file sha256:
`/Volumes/APDataStore/pact/ddm_jt23/retained/MANIFEST_jt23.json` — includes every raced candidate,
every raw section, `coder_axis_audit.json`, and `jt23_extended_raw_race.json`. Nothing measured was
discarded. Vertigo was not written (8.4 GiB free); APDataStore used per the charter.
