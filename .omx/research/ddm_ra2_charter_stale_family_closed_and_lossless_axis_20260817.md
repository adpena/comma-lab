# ddm_ra2 — the assigned family was already closed six ways; the lossless axis is the only pose-exempt one, and it is nearly closed too

`date_utc: 2026-08-17` · `owner: ddm_ra2 (carrier rank/refit)` ·
`axis: [macOS-CPU advisory, scorer-free, exact byte + exact arithmetic]` ·
`score_claim: false` · `promotable: false` · `frontier_moved: false`

**Own-vehicle frontier, UNMOVED by this unit: hv1 ep0634, S 0.15959729295498598 @ 182,759 B
`[contest-CUDA T4, n600]`, sha256 `80d9c8c6…`.** This unit produced no lower score and did not
attempt one. Cost: $0, no Modal, no dispatch, no launch, no scorer forward, no render.

Receipts: `experiments/ddm_ra2_carrier_pool_census_and_scale_gauge.py` (ruff clean) ·
`/Volumes/APDataStore/pact/ddm_ra2_pool_census_20260817/RA2_SECTION_CENSUS.json` ·
retained payloads (5 files, sha256 + byte count each) under
`/Volumes/APDataStore/pact/ddm_ra2_pool_census_20260817/retained/`.

---

## THE ANSWER, FIRST

**My charter's central premise — "carrier rank/refit … never fired" — is false. The family was
measured and closed six independent ways on 2026-08-16, the day before the charter was written. I
did not re-derive any of it. I verified the closure, corrected four stale constants the charter
carries, pivoted to the one axis that is structurally exempt from the wall that closed it, and
found that axis nearly closed too — with one measured, unfired row left.**

1. **The family is closed at FAMILY scope, with a sphere-wide bound.** Six treatments
   (`ra1`, `ra2c`, `jc1`, `ra2`, `ra3`, `ra2crr`) span it. The decisive pair: `ra1` MEASURED that
   rank-4 returns **14,709 B — 102.1% of the bar, the rate side passes outright** — and that the
   same cut costs 30.6% of carrier field energy under the *least-squares-optimal* refit, which
   lower-bounds every rank-4 refit. `ra2crr` then minimised the true score-relevant functional over
   the **entire sphere** (292/292 descents within 1% of one optimum) and got `Δd_pose = 3.2824e-03`
   against a break-even of `1.05e-06`–`2.19e-06`: a miss of **1,498×–3,139×**, and **828×** even
   after granting the most favourable model-error factor ever observed on this object.
2. **Charter fork branch 3 (POOL IS MIS-STATED) fires — but `ra2crr` fired it first.** Re-running
   it would have been the re-derivation this corpus keeps paying for.
3. **The one axis the closure cannot touch is LOSSLESS recoding**, where `Δd_seg = Δd_pose = 0`
   *by construction* rather than by measurement. The corpus's own summary sentence — *"every byte
   that can be removed post-hoc is load-bearing for pose"* — is **FALSE on that axis**, and should
   be scoped to lossy edits whenever it is cited.
4. **But `ddm_dc1` already closed most of it on hv1.** The token stream is 61.3% of the archive and
   its coder overhead is **+7.80 B (+0.00696%)** — a constant flush cost, not a rate. PR130's
   +1.85% did **not** transfer. Exactly one measured, unfired lossless row survives corpus-wide:
   `ra2`'s CPR1 inner coder, **+263 B raw / ~230 B realised (PROVISIONAL)**.
5. **That row's blocker is a self-imposed gate that `qw1` measured vacuous.** `ra2` gated it *"fire
   only when a rung of ≥2 KB is also in flight"*; nothing ≥2 KB is in flight, and every sibling
   family is closed. **By its own gate the row can never fire. The gate should be retired.**

---

## 1. What I measured (the only new numbers here)

Instrument: `experiments/ddm_ra2_carrier_pool_census_and_scale_gauge.py`, run on the frontier
archive re-hashed at use time. Two independent local copies of the archive hash identically to
`80d9c8c6…` at 182,759 B.

**Exact RX1M section census, closing to the byte:**

| section | coded B | raw body B | body magic |
|---|---:|---:|---|
| ZIP framing | 100 | — | — |
| RX1M header | 14 | — | `RX1M` |
| HPAC | 13,515 | 17,952 | `IHS1` |
| semantic | 34,763 | 36,040 | — |
| carrier | 22,161 | 22,219 | `a57f01d4` |
| residual table | 96 | — | — |
| token stream | 112,110 | — | — |
| **total** | **182,759** | | census closes exactly |

**Control (independent-arm):** my extracted carrier body is 22,219 B, sha256
`065fce08fc3d44e49d29ad624561cbef86d01282cc73dcd32533b5d63115bd9f` — **byte-identical to `ra1`'s
independently cited `carrier.raw.bin` `065fce08…`**. Two arms, two extraction paths, the same
object. The coded-section figures also agree with `ra2` §1 and `pz5` section for section.

**The strict bar, re-derived from the authority eval components** (not quoted):
`d_seg = 0.00029611`, `d_pose = 6.88e-06`, read from
`experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json`
(`score_axis = "contest_cuda"`). Recomposes to `0.159597292954986`, matching the pointer.

* rate `0.12169171641365492` · seg term `0.029611` · pose term `0.008294576541331089`
* score per byte `6.658589531221714e-07`
* **archive ceiling ≤ 168,345 B; bytes required = 14,414** (continuous: 168,345.598 / 14,413.402)

---

## 2. Four stale constants my charter carries (each corrected, each re-issuable to the next arm)

1. **"never been measured" → measured six ways, 2026-08-16.** Receipts in §3.
2. **"−15,157 B rung"** (charter §"Why this rung") is the **e480b 183,502 B** figure. `fb1` exists
   to kill it; `ra2crr` NEXT #2 flagged that MAIN's scope correction was a sixth live site. **My
   charter is the seventh.**
3. **"−14,413.4 B … ceiling 168,345.6 B"** is the *continuous* form and is **one byte short**:
   saving 14,413 B leaves `S = 0.15000026786363613` (NOT sub-0.15); 14,414 B gives
   `0.149999602004683`. The strict form is **14,414 B / ≤ 168,345 B**. `ra1` made this correction
   already. Prefer the ceiling form — it is base-invariant, and `rfo2` reached the same 168,345 B
   from a different base.
4. **The `mz2` "38/38 receiver-required" citation is cross-archive.** `mz2` measured **e480b**
   (183,502 B, semantic body 36,040 B raw / 34,763 B coded here — the coded halves coincide, the
   memo's own bar does not: `mz2` prices against **15,153 B**). Its conclusion may well transfer;
   the citation as written does not carry that warrant.

Separately: **`ra1` cites `d_seg = 0.000289620308`, which is PK2's object, not hv1's.** hv1's
authority value is `0.00029611` — 2.2% higher. Immaterial to `ra1`'s verdict (which misses by 3–4
orders) but it should not be reused.

---

## 3. The six treatments that span my assigned family (I re-derived none)

| treatment | owner | result | scope |
|---|---|---|---|
| α=0 (carrier deleted) | `ra2c` | 350,428× pose | FORMULATION |
| rank-r Frobenius truncation r=1…11 | `ra2c` / `ra1` | rate PASSES at r=4 (14,709 B); MSE 156.9 grey² = 6.3e7× PK2's gate | FAMILY |
| coordinate keep-set + coefficient re-fit | `jc1` | 235.3× / 238.9×; `K = dim(∩ᵢ null(Jᵢ)) = 0` | FORMULATION |
| pose-metric subspace projection r=11 | `ra2` | 111.2× | INSTANCE |
| subspace + trust-regioned per-pair re-fit, realised | `ra3` | **35.5×** — best any carrier arm produced | family closed |
| sphere-wide priced minimum + per-coordinate map | `ra2crr` | **1,498×–3,139×** (828× most-favourable); every coordinate 24,835×–84,984× | FAMILY |

Two structural results worth keeping, because they decide instrument choice, not just this verdict:

* **The family ceiling CLEARS the bar** (rank-4 = 102.1%), so the closure rests **entirely on
  distortion**. `ra3`'s ceiling-based ground is r=11-scoped and does not generalise — cite
  distortion, never the ceiling. (`ra2crr` §4; still owed as a source correction to `ra3` §5.)
* **No model can bound the re-fit half.** `rank(JᵢP) = 6` for all 600 pairs against 11 free
  coefficients, so the first-order re-fit is exactly solvable and the model-optimal damage is
  identically **zero for every direction**. Any re-fit verdict must be REALISED. This is why `jc1`'s
  designer missed by 1,065× and why `ra3`'s realised acceptance was the only admissible instrument.

**What would reopen it:** not a new rank, basis, or radius — all inside the bound. Only a carrier
**retrained from scratch with pose in the training loop**, which `ra1` explicitly left outside its
FAMILY verdict.

---

## 4. The pivot, and its honest result

Reasoning: every rung on `rfo2`'s route is a **lossy** edit, and all four died to the same
mechanism. A **lossless** recode is structurally exempt — the decoded tokens/frames are bit-identical,
so `Δd_seg` and `Δd_pose` are exactly zero and the wall cannot fire. I checked whether that axis had
byte headroom.

**It is nearly closed, and `dc1` closed it — on hv1, not by transfer.**

* **Token stream (112,110 B, 61.3%):** rc64 consumes 112,117.375 B against HPAC's own cross-entropy
  of 112,109.578 B → **+7.80 B (+0.00696%)**, a constant flush cost. Measured against the shipped
  stream the gap is **0.42 B**. Even the *oracle* over 8-spatial+9-prev contexts bottoms at
  144,167 B = **+32,057 B above shipped**. Any coder, any context depth: **ceiling ≤ 7.8 B.**
* **Logit-precision sub-lever — dead by construction.** `HPAC_LOGIT_PRECISION = 8` looks like a
  coarse 1/8-nat quantization, but `IntegerHPAC` is integer-native and its logits are exact
  multiples of 1/8, so `probability_table`'s `round()` is a no-op. There is no quantization loss to
  recover. I checked this at source before spending a decode pass on it.
* **Semantic and carrier-outer generic recode:** byte-identical to shipped (a definitional no-op).
  **HPAC recode:** +40 B loss, and shipped is already 1,446 B *below* its own order-0 floor.
* **Container:** ZIP framing is at its structural floor (100 B = 30 + 1 + 46 + 1 + 22, member `p`,
  STORED, no extras); header+residual+ZIP total 210 B.

**PR130's coder gap did NOT transfer and must never be re-cited on hv1:** PR130 measured +1.85%
(2,128 B) and projected −1,416 B from an ANS swap; hv1's rc64 is **+0.00696%**. `dc1` records that
the PR130 closure was *"measured once and then cited on four downstream bases without
re-measurement"*, and issues a standing instruction: **no arm should propose a coder race on any hv1
section** — run `tools/audit_archive_coder_axis.py` instead.

### What survives on the lossless axis

| row | bytes | status | owner |
|---|---:|---|---|
| `ra2` CPR1 **inner** coder (basis stream, adaptive arithmetic vs static order-0 Huffman; includes a 32 B dead length table) | **+263 raw MEASURED / ~230 realised PROVISIONAL** | unfired | **unowned** |
| `ra1` `basis_scales` gauge (receiver RMS-normalises *after* applying the scale ⇒ positive magnitudes cancel exactly) | −7 MEASURED through Brotli; **ceiling 48** | unfired | **unowned** |
| coefficient stream, adaptive arithmetic | **−415 B — Rice WINS** | do not swap | closed |

Priced at the live operating point (zero distortion, so the whole credit is rate):
**230 B → ΔS −1.531e-04 = 1.60% of the bar**; **278 B (bundled) → ΔS −1.851e-04 = 1.93%**. Both
clear the −3.5e-6 admission bar by 44× and 53×. Neither is decisive; both are free of pose risk.

**Mechanism, and it generalises:** `dc1`'s samples-per-symbol table spans 13.4 million-fold — token
stream 23,592,960 · semantic 141 · carrier 87 · HPAC 70 · **carrier coefficients 1.76**. Whether a
coder swap can pay is decided by alphabet-vs-sample-count, not by "arithmetic beats Huffman". The
basis stream pays because it is spatially white with 27,648 samples; the coefficient stream cannot,
at 1.76 samples/symbol over a 4,096 alphabet.

---

## 5. `coefficient_scales` — the never-named sibling field, DERIVED not gauge

`pz5` parses the carrier as basis codes + int12 coefficients + **"96 B of scales"**, and
`carrier_codec.decode_compact_carrier` shows two `<f4[12]` fields: `basis_scales` **and**
`coefficient_scales`, 48 B each. `ra1` proved the first is pure gauge. **The second has never been
named in any memo.**

**DERIVED at source (not numerically confirmed — see §6): it is NOT a gauge.**
`basis_scales[k]` cancels because the receiver applies it *before* `normalized_basis`, which
mean-subtracts and RMS-divides, so any positive magnitude is absorbed — only the **sign** survives
(`normalise(−x) = −normalise(x)`), and `ra1` measured all 12 signs `+1`, which is why the whole 48 B
is redundant *in this archive* (12 bits in general). By contrast `coefficient_scales[k]` multiplies
`coeff[b,k]`, which multiplies an atom that is **already unit-RMS**; there is no downstream
renormalisation to absorb it, so it sets the atom's actual amplitude. It is load-bearing.

Two corollaries: re-precisioning it (fp32→fp16, −24 B) is **lossy**, so it belongs on the closed
pose-collateral axis, not this one; and its sign is redundant against the coefficient signs (12 bits
≈ 1.5 B, below the 1e-5 S naming bar). **No byte credit here. The hole in the record is closed.**

---

## 6. Honest limits

* **The CPR1 field census is UNRESOLVED and I did not guess it.** The 22,032 B (`ra2`, "different
  objects, all correct") vs 22,155 B (`ra2crr`, coefficient half re-derived at 79,020 bits)
  contradiction stands. The brotli-decoded carrier body carries no `CPR1`/`CAP1`/`F0C1` magic and
  `materialize_cpr1` from the `hv1_base_control` generation runtime refuses that slice directly. The
  generation receipt records `receiver_closed=true` and `parser=legacy`, so the receiver plainly
  does decode it — **my extraction is one legacy-parser step short, which is NOT evidence of a
  different wire.** The instrument reports this as a typed blocker rather than a number.
  Consequence: **§5's gauge verdict is DERIVED from the receiver source, not numerically confirmed
  on the shipped coefficients.**
* **`ra2`'s ~230 B is PROVISIONAL and measured in the wrong domain.** It priced Brotli over the
  *canonical* CPR1 form (22,278 B); the archive ships a compacted body (22,219 B → 22,161 B coded).
  The uncertainty scale is the **117 B** gap between those two. If the realised credit lands below
  ~113 B the row drops under 1% of the bar. **This must be re-measured at the repack layer before
  the row is fired.** `ra1` §"side finding 2" records that the shipped CPR1 encoder in-repo is
  byte-exact against the frontier archive, so real coded lengths are available directly.
* Everything in §3 and §4 is **recalled, not re-measured** by me. I read each at source and cite it;
  I did not re-run any of it, by design.
* No contest row. Advisory/exact-byte throughout.

---

## 7. VERDICT

**`REFUSED — carrier rank/refit supplies no byte on this vehicle; the family was already closed at
FAMILY scope with a sphere-wide bound before this charter was written.`**
`verdict_scope: FAMILY` (inherited from `ra1` + `ra2crr`, verified not re-derived).

**`Lossless axis: NEARLY CLOSED on hv1.`** Token ≤ 7.8 B, semantic/carrier-outer no-op, HPAC +40 B
loss, container at floor. **One measured unfired row remains: `ra2`'s CPR1 inner coder, ~230 B
PROVISIONAL, bundling with `ra1`'s 48 B gauge to ~278 B = 1.93% of the bar.** Zero distortion by
construction; needs a repack-layer re-measurement, not a new idea.

**`coefficient_scales: NOT a gauge (DERIVED). No byte credit. Record hole closed.`**

**The strategic read:** all four `rfo2` rungs and the whole lossless axis are now measured, and
together they supply **at most ~278 B of the 14,414 B bar (1.93%)**. Post-hoc editing of this
archive is exhausted. The remaining routes run through **joint descent** — a carrier retrained with
pose in the loop, or the semantic/FiLM field — not through further byte surgery on shipped bytes.

---

## 8. NEXT_IF_RESUMED

| # | row | owner | fire-condition |
|---|---|---|---|
| 1 | **Retire `ra2`'s "fire only when a rung of ≥2 KB is also in flight" gate.** `qw1` measured nothing ≥2 KB is in flight and every sibling family is closed, so by its own gate the only measured unfired win in the corpus can never fire. Then re-measure its ~230 B **at the repack layer** (uncertainty 117 B), bundle `ra1`'s 48 B `basis_scales` gauge, receiver-close, byte-close. | MAIN to route; unowned | $0, no scorer, immediate |
| 2 | **Nothing further on carrier rank/refit.** Six treatments, a per-coordinate pose map and a sphere-wide bound span it. | — | do not reopen without a carrier retrained WITH pose in the loop |
| 3 | **Stop re-issuing the stale bar.** Live: **14,414 B strict / archive ≤ 168,345 B**. `fb1` tracked five sites, `ra2crr` a sixth, this charter is the seventh. Prefer the ceiling form — it is base-invariant. | MAIN | $0, immediate |
| 4 | **Scope the "every removable byte is load-bearing for pose" law to LOSSY edits.** As written it is false on the lossless axis, and it is the sentence most likely to be inherited. | MAIN | $0, immediate |
| 5 | **Resolve the legacy carrier parser step**, then settle the 22,032 vs 22,155 pool census and numerically confirm §5. Cheap and well-specified; the instrument already isolates the body and reports the exact blocker. | unowned | before the pool is priced again |
| 6 | **The semantic/FiLM FD pose map remains the live blocker for −2,874 B (20% of the bar)** — `mp2`'s banked q3/q4 and FiLM-row candidates are held on pose collateral in the SEMANTIC section, not the carrier. Needs a scorer pass; cheapest honest form is grouped-FD on a seeded RANDOM pair subset (never a contiguous prefix — pose prefixes measure 2.54–4.21× harder). | unowned — MAIN to route | a scorer lane frees; NOT $0 |
