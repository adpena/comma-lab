# LC3 Lane carriage rung — exact cropped SMEVR is 137,670 B; the rung stays closed

`axis: [scorer-free exact rate and receiver measurement]` · `score_claim: false` · `$0` · no scorer · no Modal · no `upstream/` writes

## Result

The newly raced exact carrier is **137,670 B**. Its independent canonical receiver reconstructed the full **117,964,800 B** source token field byte-identically at SHA-256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.

That carrier is:

- **115,971 B over** the 21,699 B sub-0.12 Lane allowance, an arithmetic rate penalty of `+0.07722032865253134 S` at `6.658589531221714e-7 S/B`;
- **101,626 B worse** than GF1's 36,044 B parametric Lane stream; and
- projected to **253,957 B** only when added to D3's retained 116,287 B GB1-era rate-only archive.

The 253,957 B figure is a **projection, not an archive**. LC3 did not jointly re-encode the AFR1 body, did not materialize a composed archive, and did not score anything. The mechanism failed both Stage-2 byte gates before composition, so the charter did not authorize an AFR1 joint re-encode, local scorer race, seal, or MAIN fire-order.

The own-vehicle frontier therefore remains **AFR1: S=0.14797617125559104 at 180,002 B `[contest-CUDA T4 n600]`**, archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`. LC3 did not move it.

## Candidate ledger

| candidate | arithmetic floor before build | built in LC3? | measured bytes | identity | composed archive | projection vs 21,699 B | verdict |
|---|---:|:---:|---:|---|---:|---:|---|
| GF1 parametric refit alone | 36,044 B existing payload | no | 36,044 B prior object | FAIL as an exact carrier; the analytic whole-field fit leaves 1,325,033 mismatches | — | +14,345 B / +0.0095517467 S | BOUNDED-CLOSED, existing GF1/HG1 formulation |
| D3A q8 trunk + D3B lossless residual | 27,440 B counted trunk alone, before residual | no | 92,024 B prior exact packet | exact in D3B | 208,323 B actual GB1-era q8 archive | +70,325 B / +0.0468265309 S | BOUNDED-CLOSED, D3B formulation |
| recalled dash-phase object leg | 29,958 B event section before exact shape | no | 29,958 B prior section | insufficient: phase does not determine exact shapes | — | +8,259 B / +0.0054993291 S before shape | BOUNDED-CLOSED, standalone exact-carrier use |
| BZ2D generator-form transfer | 16,549.13 B optimistic `36,044/2.178` ratio projection | no | unmeasured on exact Lane | unproved; GF1 measured 385,448 B exact residual / 433,051 B total for this inherited form | — | apparent 5,149.87 B headroom is not byte evidence | BOUNDED-CLOSED, existing BZ2D/HG1 form transfer |
| **cropped SMEVR exact Lane** | 35 B framing before semantic payload | **yes** | **137,670 B** | **EXACT** | **253,957 B projection only** | **+115,971 B / +0.0772203287 S** | **DEAD, cropped-SMEVR formulation** |

The machine-readable rows are in `ddm_lc3_lane_carriage_rung_20260831.jsonl`.

## Built mechanism and retained bytes

The D3 exact Lane mask has 691,095 positive pixels. Across all 600 frames, every positive lies in rows `[158, 294)`, so LC3 counted those bounds and encoded only the 136-row crop. Three canonical R7 SMEVR tiles carry the semantic payload:

| rows | measured bytes | SHA-256 |
|---|---:|---|
| 158–205 | 49,012 | `13f6336ebea1a35904b19f0493bc0058c79d2ec19122f9f80c330ba1c53a75af` |
| 206–253 | 64,425 | `f69451a682e862881b774baef211ddf7611a3e940ab08205d47bbcac3b3c441a` |
| 254–293 | 24,198 | `048c019bd6f8ec9d9daa9751bdf8bfd498b826486c90afaba0fc6db7aa4a577a` |
| packet framing | 35 | counted LC3 header and three tile headers |
| **total** | **137,670** | `daf6df3891888368a6fab982a0b86f47fac10c1e72c18c7e8b6e93f0334450d7` |

The packet is retained at `/Volumes/APDataStore/pact/ddm_lc3/retained/cropped_smevr/lane_exact_cropped_smevr.lc3`. Each candidate tile, the packet, the receiver mask, the 117,964,800 B decoded token field, the corruption-control packet, and every stage receipt are retained with bytes and SHA-256.

The receiver parses counted geometry and lengths, canonical-decodes each SMEVR frame, restores zeros outside the crop, and paints Lane only where the receiver-closed D3 quotient is Road. It does not read the source mask while decoding. The resulting full field is exactly the pinned source SHA.

## Both-direction receiver control

The positive control is the retained 137,670 B packet, which reconstructs the exact source field. The negative control flips one bit in the packet's final arithmetic padding byte and retains that 137,670 B corrupt object at SHA-256 `bb112e36259f5a4afed818508365b091ddf88518fdbf1e34bac611c62496eba8`. The digest-only verification rung accepts that semantically inert padding change by design; the production canonical verification rung refuses it as `token frame is noncanonical or has inert bytes`. This distinguishes semantic identity from canonical counted-byte identity and proves the LC3 receiver is sensitive on the rung it actually uses.

Receipt: `/Volumes/APDataStore/pact/ddm_lc3/FALSIFIER_RESULT.json`.

## Why the other candidates did not rebuild

1. **GF1 precision refit alone is not lossless.** Its retained Lane stream is already 36,044 B, 14,345 B over the carriage bar, before any exact residual. GF1's measured formulation ceiling leaves 1,325,033 whole-field mismatches; exactness cost 385,448 B in the best residual order. Lower coefficient precision cannot produce an exact candidate without joining the residual leg.
2. **D3B already executed the trunk-plus-residual race.** Its smallest exact counted Lane packet is 64,276 B. Adding q8 analytic geometry produces 92,024 B; q1 produces 106,584 B. All identities pass, so this is a rate-only formulation closure, not a receiver failure.
3. **Dash phase is already underwater before shape.** The recalled n600 phase-event section is 29,958 B, 8,259 B over the complete-carriage allowance, while phase alone cannot reconstruct the exact Lane pixel shapes. The charter forbids rebuilding the prior merging/implicit composition that already lost.
4. **The BZ2D ratio is not a transferable packet.** BZ2D measured a 2.178× form advantage on its own generated field. Later GF1 evidence showed that the existing HG1 form and its approximately 1.12% fit error are inseparable on a foreign exact target. Applying `2.178×` to 36,044 B yields a useful hypothesis, not a measured byte floor.

## RECALL EVIDENCE

The recall pass searched `.omx/research/`, retained arm receipts, canonical research index/DAG surfaces, the task ledger/hot state, the canonical-equations registry, source, and the falsified-premise registry. Content queries included `Lane carrier`, `class1 mask`, `lossless conditional`, `D3A`, `D3B`, `D3C`, `GF1`, `movable precision`, `horizon`, `dash phase`, `object codec`, `generator form`, `bit-identical`, `SMEVR`, `Brotli`, `LZMA`, `DALI`, `272869`, `36044`, and the AFR1 archive SHA.

Beyond the charter's named seeds, recall found that D3's direct exact Lane-mask race tested Brotli q11 transforms but never the previously winning SMEVR coder. Source inspection then found R7's 16,000,000-value frame bound, which changed the build into bounded row tiles. Exact-mask inspection found the global `[158,294)` support, which removed 248 guaranteed-zero rows without hiding data because the bounds are counted. Recall also found the falsified-premise registry's corrected RL1 crop price—272,869 B, not 26,500 B—and the later GF1 memo that retracts unrestricted transfer of BZ2D's 2.178× form ratio. Those findings caused LC3 to build cropped SMEVR as the only uncovered exact candidate and to close, rather than rebuild, the already-measured D3B, dash-phase, and inherited-HG1 formulations.

No additional under-36,044 B exact Lane carrier was found in the searched corpus.

## Custody and boundaries

- Live body anchor: AFR1, 180,002 B, SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
- Exact source token field: 117,964,800 B, SHA-256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.
- Exact D3 quotient field: 117,964,800 B, SHA-256 `deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07`.
- Exact Lane mask: 14,745,600 B packbits, SHA-256 `6ca82a7883411d0eb27addac7dcf662e84d2f9cc66404c299da2e15761c0e0cf`.
- GF1 incumbent Lane payload: 36,044 B, SHA-256 `7add7be325443bf3834f8058e9997ed01d94f3d4af59c830e419f0c20afcdcb1`.
- No score was run. Zero distortion here means token-field identity only; it does not create a new contest score.
- The 116,287 B D3 rate-only archive and 180,575 B D3B archive are GB1-era actual objects. They are not AFR1 joint re-encodes. LC3 labels every composition using the former as a projection.
- No candidate beat GF1. Therefore no full AFR1 archive, local identity chain, seal, scorer race, dispatch claim, or paid job was created.

## Follow-on dispositions

- `FIRED → DEAD` — cropped exact SMEVR; owner `ddm_lc3`; consumer `/Volumes/APDataStore/pact/ddm_lc3/`; trigger was the uncovered exact-coder race; result 137,670 B with exact identity.
- `FOLDED → BOUNDED-CLOSED` — GF1 parametric refit and D3A/D3B residual composition; owner `MAIN`; consumers the GF1 and D3B retained stores; fire trigger for reopening is a materially different exact predictor with an arithmetic floor below 36,044 B.
- `FOLDED → BOUNDED-CLOSED` — recalled dash-phase standalone exact carriage; owner `MAIN`; consumer `/Volumes/APDataStore/pact/ddm_lc3/`; fire trigger is a proof that a free visibility generator removes at least 8,259 B before exact shape bytes are charged.
- `QUEUED-WITH-A-FIRE-ORDER` — structurally different Lane-specific topology/event generator; owner `MAIN`; consumer `/Volumes/APDataStore/pact/ddm_lc3/`; fire only when its counted generator plus exact residual has a retained arithmetic floor below 36,044 B and the receiver target is the exact `cc10a7b…` field. The BZ2D ratio alone does not satisfy the trigger.

## LIVE-HYPOTHESES

- A Lane-specific topology/event generator may still beat GF1 because BZ2D proves generator form can be cheaper on a bit-identical self-generated field, while GF1 only closes the existing HG1 formulation on a foreign target. It is plausible only if topology births, deaths, dash visibility, and exact shape residuals are jointly represented rather than fit after the fact.
- A free persistence-class visibility generator could reduce the 29,958 B dash-event section because prior object coding spends heavily on birth/rebirth state that a deterministic visibility law might predict. It remains speculative and must first clear an 8,259 B lower-bound deficit before exact shape bytes matter.

## DEAD-ENDS

- Global vertical crop plus exact SMEVR row tiles: 137,670 B, exact, but 101,626 B worse than GF1 and 115,971 B over the target bar.
- Existing D3B HPAC conditional and D3A q8/q1 reference mixers: exact receiver closure, but their counted Lane packets are 64,276–106,584 B.
- Pure GF1 coefficient-precision refit as an exact carrier: the 36,044 B lossy parametric stream already exceeds the bar and does not determine the exact field without a residual.
- Rebuilding the recalled dash-phase merging/implicit composition: the event section alone is 29,958 B and the prior composition already lost; it still omits exact shape.
- Transferring BZ2D's 2.178× ratio as if it were a Lane packet: GF1 measured that the existing generator form and fit error are inseparable on the exact target.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; LC3 ran no scorer and did not move the pointer.`

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `static_packet_custody_byte_delta_score_savings_v1` — `tac.canonical_equations (registry: static-packet custody byte-delta)` (`tac.canonical_equations`). **Relation:** IN-DOMAIN ANCHOR (rate-only ΔS arithmetic, `score_claim: false`).

The 137,670 B exact carrier is priced by ΔS = 25·ΔB/37,545,489: +115,971 B over the 21,699 B Lane allowance = +0.07722032865253134 S. The BZ2D row is an in-domain refusal under `generator_form_fit_error_entanglement_v1` — the 16,549.13 B `36,044/2.178` projection is a ratio, not byte evidence.
