# ddm_cl2 — the HPAC prior-capacity ladder on the SHIPPED fs2 mixer: cl1's unfired λ bracket, re-rooted to the shipped object

**Arm:** ddm_cl2 (Fable, spawned 2026-09-05 ~12:40Z on MAIN's charter
`.omx/research/charters/ddm_cl2_hpac_prior_capacity_ladder_on_shipped_object_20260905.md`).
**Pointer at spawn:** fs2 — S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600], archive sha
`a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6`. Demand at held distortion: −41,817.8 B
(archive ≤ 138,205.2 B). Tokens: `[no-triality] [p0-ledger-ok]`.

Axis of every byte number below: `[macOS-CPU advisory / scorer-free EXACT byte measurement]` (pack + fx1 mixer +
RC64 through the shipped fs2 path; receiver-copy decode identity). Training axis: `[macOS-MPS research-signal]`.
No scorer ran; the token FIELD is bit-identical on every row (sha `cc10a7b0…63efb`), so d_seg / d_pose are HELD by
construction and only model bytes + stream bytes move. Labels: MEASURED / DERIVED / INFERRED as marked.
`score_claim=false` until the decision-rule section says otherwise.

**Result first.** cl1's ladder, fired on the shipped object: the λ=1.0 control reproduces the shipped joint at
**−41 B** (13,466 + 113,419 = 126,885 B vs 126,926; instrument PASS), the **λ=0.5 rung costs +350 B of model for
+156 B of stream (secant +0.446 against the −1 bar; joint +465 B vs shipped)** — the prior law is FALSIFIED at the
first rung and λ=0.25 is not fired. The only pointer-relevant bytes are the control's −41 B, a retrain/pack-size
effect, deterministic (twin byte-identical at every layer), VERIFIED through the shipped container path and
parse-back-identical at the render; it is SEALED for contest-CUDA as a 26th-move candidate (§6). The demand
(41,818 B) is untouched: 0.098 % of it.

---

## 0. The shipped model's exact training law (charter step 0) — MEASURED from the checkpoint's own `run_identity`

The shipped 13,515 B IHS1 section is the **EMA shadow at epoch 634** of an `rx2_mc36`-profile run of
`tools/train_ddm_cl1_hpac_capacity.py` — the same trainer cl1 owns — on **Metal** (`device: mps`), NOT a 60-epoch run:

| field (from `epoch_0634.pt`, sha `5007beae…147ec`, retained by jf1) | value |
|---|---|
| profile / device / epochs declared | `rx2_mc36` / `mps` / **960** (cosine `T_max 480`, `eta_min 6e-5`) |
| resume lineage | epoch 480 parent `qat_stage_end_epoch_0480.pt` (sha `cd89907b…`) → `full_e480b_e960` |
| phase at ep634 | `discrete_qat` (QAT from epoch 481); `_last_lr` = 6e-5 (past `T_max`) |
| init / cache | P64 exact-from-archive init `0e6c30ce…`; MC36 spatial cache `f53db4e8…` (field `9ba2e52b…`, the hv1-era field) |
| selection | post-hoc argmin over periodic checkpoints by `estimated_joint_bytes` (130,393 B; hv1 retarget ep508 → ep634) |

So the shipped weights come from a ~634-epoch Metal burn on an OLDER field with a post-hoc epoch pick — a law that
costs ~6× a 60-epoch run and is not the one any refit arm has used. The **reference refit law** (jf1/jf2, "sealed
60-epoch reference"): the same trainer, profile `jf1_joint_refit`, **warm-started from the ep634 EMA state**
(`epoch_0634_ema_init.pt`, sha `ff2d3e45…2afd9`), 60-epoch cosine, seed 20260716, batch 8, QAT 0.5, λ = 1.0, on the
CURRENT field `cc10a7b0…` — on **CPU**. jf2 MEASURED its control: packed model 13,463 B (−52), stream 114,143 B on the
dx2 path (+366 vs that path's shipped 113,777 B), joint +314 B vs 127,292 B. That is the instrument that reproduces
the shipped family, to within +314 B joint — inside the charter's +500 B tolerance.

**Decision (charter step 0):** the trainer that reproduces the shipped family is `tools/train_ddm_cl1_hpac_capacity.py`
with the JF1 warm-start law. I added profile **`cl2_shipped_ladder`** (commit `e1060eb5c`): identical config to
`jf1_joint_refit` except `device: mps` (so a rung costs ~48 min, not ~3 days), the cl1 λ bracket {1, 1/2, 1/4}
admitted, caller-pinned inputs (`--expected-cache-content-sha256 cc10a7b0…`, `--expected-init-sha256 ff2d3e45…`), and
SSD output. The shipped 13,515 B model is NOT a repack of PR135's weights; it is our own rx2 lineage, so the warm start
IS "from the shipped weights via `--init`" exactly as mc1 planned.

Note on cl1's history: cl1's Gate 1/2 controls DID fire on 2026-08-12 on PR130's object (P64 init, DALI cache):
resume-vs-twin bit-identical (`GATE2_EQUALITY_ADJUDICATION.json`: `SUBSTANCE_PASS_INSTRUMENT_DEFECT`); twin packed
15,088 B / Range 116,716 B (joint 131,804 B, −340 B vs PR130's 132,144 B); training 2,894 s, peak RSS 1,673 MiB. The
λ = 0.5 rung was never fired — that is the gap this arm closes, on the shipped object rather than PR130's.

## 1. Instrument (the shipped path, nothing re-implemented)

`experiments/ddm_cl2_hpac_prior_capacity_ladder.py` (stages `control` / `price` / `report`):

1. **pack** — `ddm_rx2_mc36_identity_race._pack_terminal_ihs1` (the PR130-intake IHS1 packer; idempotency and decode
   determinism asserted) + the Brotli q0–q11 race, exactly as jf1's `_pack_model`.
2. **stage** — a copy of the fs2 fire tree (`ddm_fs2_carrier_resolve/fire_runtime_D_alternation`, archive
   `a8f3a379…`) whose `archive.zip` carries the new hpac section (RX1 header length rewritten; semantic, carrier,
   residual table, ZIP framing untouched).
3. **encode ×2** — `ddm_jg2_tail_reencode.encode_tail` over the exact field: `decode_production_tokens` line for
   line with the decode replaced by an encode of the known symbols (model, group order, boundary buckets, fixed
   table, `FreeCorrector` = the fx1 mixer, RC64). mc1 proved this walk reproduces the shipped stream byte-for-byte
   (113,411 B, sha `5601d6fd…`). Two fresh encodes must be byte-identical.
4. **decode** — a receiver copy (both `inflate.py` pins patched, jf2 #1237) decodes the candidate archive with the
   shipped `decode_production_tokens`; the decoded field must equal the target byte for byte; wall-clock recorded
   next to the shipped archive's own decode through the same path (`control` stage).

Every payload is retained under `/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/` with sha256 +
bytes (checkpoints, raw IHS1, all 12 Brotli representations, both streams, candidate archive, receiver-copy runtime,
decoded field).

## 2. Launch discipline (all MEASURED)

- Metal smoke (`--stop-after-epoch 2`, launcher counter 863): 127 s, peak RSS 1,697 MiB, **system-availability delta
  15.557 GiB** (the Metal allocator is invisible to RSS) — recorded in `.omx/state/measured_peaks.jsonl` under family
  `train_ddm_cl1_hpac_capacity`. Epoch-2 telemetry of the smoke and of the full control run are IDENTICAL
  (bpp 0.008788232009180965, est. tokens 129,588 B): fresh-vs-fresh MPS determinism, re-confirmed on this object.
- One training process at a time on Metal; every long step through `tools/launch_detached_process.py` with a distinct
  `--done-receipt`; CPU pricing jobs overlap the Metal training (they touch no Metal).

## 3. Prior-law prediction and falsifier (charter, counted plainly)

PREDICTION (dc1's mechanism): λ 1.0 → 0.5 grows the packed model by ≤ +1,500 B and cuts the exact stream by ≥ 2× that
(net joint ≤ −1,500 B); 0.25 continues with a shallower slope. FALSIFIER: λ = 0.5's net joint ≥ 0 vs the control, OR
the control cannot reproduce the shipped joint within +500 B (→ INSTRUMENT-REFUSED, not a family verdict).

## 4. Control rung reproduction gap (charter step 1) — MEASURED through the fs2 path

The shipped path's own control first: the fs2 archive decodes through the receiver copy to the exact field
(`cc10a7b0…`) in **1,494.5 s** CPU; hpac section 13,515 B; stream 113,411 B / `5601d6fd…` (identity PASS).

| control row | trainer device | packed model B (Δ vs 13,515) | stream B (Δ vs 113,411) | joint B (Δ vs 126,926) | two encodes | receiver decode | decode s |
|---|---|---:|---:|---:|---|---|---:|
| `cpu_control_jf2_null` — jf2's terminal ep-60 null checkpoint (sha `3aca9dbb…`), priced by THIS arm through the fs2 path | CPU (jf1 profile) | 13,463 (−52; brotli q10, sha `af8bde55…` = jf2's selected model byte-for-byte) | 113,715 (+304) | **127,178 (+252)** | identical (`706f3e6c…`) | PASS | 1,151.8 |
| `lambda_1p0` — this arm's Metal control (ckpt sha `90f7bc38…`) | MPS (cl2 profile) | 13,466 (−49; brotli q11, sha `66801b10…`) | **113,419 (+8)** | **126,885 (−41)** | identical (`e07274ca…`) | PASS | 1,299.3 |

Reading: on the fs2 path (fx1 mixer in the loop) jf2's CPU control misses the shipped joint by **+252 B**; on jf2's own
dx2 path it missed by +314 B. Both are inside the charter's +500 B tolerance, so the warm-start 60-epoch law IS the
shipped family to within a quarter of a kilobyte — and the 60 fresh epochs at lr 0.003 do not recover the shipped fit
either (the shipped weights sit at a point this schedule slightly perturbs). The candidate archive of the CPU control
is 180,275 B (+252). Decode wall-clock is faster than the shipped decode by 342.6 s — the shipped-archive control ran
while two other cells shared the CPU, so this delta is scheduling noise, not a mechanism; no rung is slower than the
shipped decode.

**Control verdict (charter step 1): PASS.** The Metal control reproduces the shipped joint to **−41 B** (model −49 B,
stream +8 B) — not merely within +500 B but below the shipped joint. The instrument is the shipped law on this
object. Candidate archive 179,982 B (−41 B vs 180,023), rate-only ΔS −2.7300e-5 (DERIVED at 6.6586e-7 S/B).

Candidate verification through the SHIPPED container path (`verify --rung lambda_1p0`, all MEASURED):
- section census vs the shipped member: header / semantic (34,763 B) / carrier / residual table (96 B) byte-identical;
  ONLY the hpac section (13,515 → 13,466 B) and the token stream (113,411 → 113,419 B) moved;
- container identity: `up3.parse_shipped_body(receiver copy)` + `up3.build_archive(body, body.codes)` at fs2's own shape
  (ck2 off, brotli q9 / lgwin16) rebuilds the candidate archive **byte for byte** (sha `08ec8533…`, 179,982 B);
  packed metadata and Rice payload identical;
- receiver-copy tree vs the fs2 fire tree: exactly two files differ — `archive.zip` and `inflate.py`'s two pin lines
  (sha + byte count; jf2 #1237 both patched);
- no-op detector: hpac section length + sha changed, stream changed, and the receiver decodes the stream to the exact
  field only under the candidate model (arithmetic coding under the new probabilities) — the new bytes are consumed.

Parse-back through the receiver tree's OWN inflate path (`runtime.f26_inflate.inflate_archive`, CPU, no scorer; the
tree's `inflate.py` pins CUDA so the call goes one level down with `device_name="cpu"`, after the tree's own
`_verify_input` pin check; RC64 backend + native f26 corrector compiled exactly as `inflate.sh` does):
- shipped fs2 tree (control): render `0.raw` **3,662,409,600 B, sha `f86bfaf3…fec4e0`**, 1,547.2 s decode+render
  (MEASURED); pin check PASS.
- λ=1.0 candidate receiver copy (archive 179,982 B, pins re-pinned): render **3,662,409,600 B, sha `f86bfaf3…fec4e0` —
  BYTE-IDENTICAL to the shipped render**, 1,519.9 s (MEASURED); the tree's own pin check PASS. Same field, same
  semantic and carrier sections → the same frames; the receiver accepts the candidate exactly as it accepts the pointer.

## 5. The ladder (charter steps 2–3) — MEASURED, exact bytes through the shipped path

| rung | λ | device | packed model B (Δ vs 13,515) | stream B (Δ vs 113,411) | joint B | Δ vs 126,926 | fraction of the 41,818 B demand | archive B | two encodes | decode identity |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| shipped fs2 | 1.0 (rx2 burn, ep634) | MPS | 13,515 | 113,411 | 126,926 | 0 | — | 180,023 | — | PASS (control stage) |
| `cpu_control_jf2_null` | 1.0 | CPU | 13,463 (−52) | 113,715 (+304) | 127,178 | **+252** | −0.60 % | 180,275 | identical | PASS |
| `lambda_1p0` (control) | 1.0 | MPS | 13,466 (−49) | 113,419 (+8) | 126,885 | **−41** | +0.098 % | 179,982 | identical | PASS |
| `lambda_0p5` | 0.5 | MPS | 13,816 (+301) | 113,575 (+164) | 127,391 | **+465** | −1.11 % | 180,488 | identical | PASS |
| `lambda_0p25` | 0.25 | — | NOT FIRED (pre-registered: fires only if the 1.0→0.5 slope < −1) | | | | | | | |

**Adjacent secant λ 1.0 → 0.5 (the ladder's one measured slope): Δmodel +350 B, Δstream +156 B, slope
Δstream/Δmodel = +0.446** against cl1's break-even −1. The rung grew the model AND made the stream worse; joint
+506 B vs the control. Halving the model-bit multiplier bought weight precision the coder does not use, and the
perturbed fit codes the field less well. The demand fraction of the best rung (the control) is 0.098 % of 41,818 B.

Trainer surrogates for the record (advisory, never a row): λ=1.0 terminal est. model 17,765 / tokens 116,529;
λ=0.5 est. model 18,279 / tokens 116,630 — the surrogate pointed the same way (+514 model, +101 tokens) before the
exact price confirmed it. (Twin control: § 5b.)

### 3′. Prior-law prediction and falsifier — COUNTED
PREDICTION (λ 1.0→0.5: model ≤ +1,500 B, stream ≤ −2× that, net ≤ −1,500 B): **FALSIFIED** — measured +350 B model,
+156 B stream, net **+506 B**. FALSIFIER ("λ=0.5's net joint ≥ 0 vs the control"): **FIRED**. The instrument clause
did NOT fire (control inside +500 B; in fact −41 B). Verdict scope: FORMULATION — fixed C64/P64/delta2/D8 topology,
the 60-epoch warm-start law, multiplier ∈ {1, 1/2}. dc1's "affordable only as learned weights" is not refuted as a
mechanism statement; what is refuted is that the trainer's rate multiplier is the coordinate that buys it.

### 5a. λ=0.5 admissibility — MEASURED: the row stands
Two fresh encodes byte-identical (stream sha `72bc70e1…`, 113,575 B); receiver-copy decode of the 180,488 B candidate
archive returns the exact field (identity PASS, 1,040.1 s); packed model brotli q10, sha `c6853f29…`, raw IHS1 18,284 B
(vs the control's 17,770 B raw — the extra 514 raw bytes are the released bit depths). The +465 B row is admissible;
it lost on exact bytes, not on a defect.
### 5b. Control determinism — the uninterrupted twin (cl1 rung 1) — MEASURED at the tensor level
A second, fresh-root run of the exact control law (same init, cache, seed, config, device; launched 47 min after the
first while my own commits moved HEAD): terminal `qat_stage_end_epoch_0060.pt` — **all 37 EMA-deploy tensors and all
37 live tensors `torch.equal` to the control's; the full 60-epoch telemetry history identical**. The checkpoint
files differ (sha `caea8f02…` vs `90f7bc38…`) ONLY through `run_identity.launch_git_sha` (repo HEAD moved between
launches) — the same provenance-in-the-causal-hash instrument defect cl1's Gate 2 adjudicated on 2026-08-12
(`SUBSTANCE_PASS_INSTRUMENT_DEFECT`). Training time 3,172 s, peak RSS 1.76 GiB. The packed-bytes and stream identity
of the twin through the shipped path — MEASURED: raw IHS1 sha `81728190…` and packed Brotli-q11 sha `66801b10…`
(13,466 B) identical; RC64 stream sha `e07274ca…` (113,419 B) identical; candidate archive sha `08ec8533…`
(179,982 B) identical — **byte-for-byte the control's bytes at every layer**; the twin's own receiver decode of its
candidate archive returned the exact field (identity PASS, 1,068.1 s; MEASURED after the seal, closing the formality).

## 6. Decision rule (pre-registered, charter §4) + MAIN's mid-arm read

Charter rule: best rung's receiver-closed archive < 180,023 B with identity → build through the shipped container
path, identity control, no-op detector, parse-back, seal for contest-CUDA with the single-axis waiver → READY-FOR-T4
(26th-move candidate; MAIN fires). ≤ 138,205.2 B → FIRE ORDER. Else REFUSED.

- Best rung = the **control** (λ=1.0): archive **179,982 B < 180,023 B**, identity PASS → the READY-FOR-T4 branch.
  It is NOT ≤ 138,205.2 B (shortfall 41,776.8 B) → no FIRE ORDER; the rate corner is not reached. The λ ladder itself
  paid nothing (best non-control rung +465 B), well under the charter's own "under 5,000 B: say so plainly" line —
  this is a −41 B pointer move at most, not a corner.
- MAIN's mid-arm read (received while the ladder was in flight): the control is itself a rung; seal it if (a) the twin
  proves the −41 B deterministic (bit-identical packed model + stream) and (b) λ=0.5 prices worse. (b) holds (§5).
  (a): tensors bit-identical (§5b), packed IHS1 raw and Brotli-q11 bytes byte-identical (raw sha `81728190…`,
  packed sha `66801b10…`, 13,466 B); stream sha `e07274ca…` and archive sha `08ec8533…` identical (§5b). **(a) holds.**
- Label, honestly: a **re-train / pack-size effect** (−49 B of model for +8 B of stream), NOT a capacity win; the
  ladder's prior law is FALSIFIED. Projected S if the −41 B holds on T4 (rate-only, DERIVED): 0.14781744131049854.
- **SEAL (READY-FOR-T4, 26th-move candidate; MAIN fires):**
  `/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/SEAL_ddm_cl2_lambda1_control_repack_contest_cuda.json`
  (mirror `.omx/research/ddm_cl2_capacity_20260905/`), seal sha `e42288fc72c8e67a3bd7d0002f0fad021917ff0c9acaba85c8100dd6af8afc60`,
  sealed 2026-09-05T15:52:40Z by `tools/make_candidate_seal.py`: candidate `ddm_cl2_lambda1_control_repack`, axis
  `contest_cuda`, archive `08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e` (179,982 B) inside the
  receiver-copy runtime tree (sha `ce20617d…`, 41 files, 878,428 B; pins `inflate.py` `053d4dc7…` / `inflate.sh`
  `1300e6ee…` = fs2's), admit bar net dS < −2e-5 derived against the fs2 pointer (0.14784474152757654, archive
  `a8f3a379…`, tolerance 0.0), bound falsifier COMPUTED from fs2's base receipt (3.691128e-06 = seg 5.0e-07 + pose
  3.191128e-06), four pre-registered falsifiers (field/distortion held; 179,982 B; rate-only −2.7300e-5 / projected S
  0.14781744131049854; twin determinism), single-axis waiver stated in the notes (the shipped `inflate.py` pins CUDA;
  contest-CPU timed out on this body). Predicted S = 0.14784474152757654 − 41 × 25/37,545,489 = **0.1478174413…**
  (rate-only; distortion held by the decoded-field identity and the byte-identical render).

## 7. What I did NOT do (plainly)
- λ=0.25 was NOT fired: the pre-registered fire condition (1.0→0.5 slope < −1) failed (slope +0.446).
- No scorer ran, no Modal was spent, no Metal cell overlapped another (one training at a time; the twin ran on the
  idle Metal while λ=0.5 was priced on CPU).
- The shipped model's own 960-epoch burn law was NOT re-run (≈ 6× a rung; post-hoc epoch selection); the reference
  refit law was used, and its control reproduces the shipped joint (−41 B).
- The `MANIFEST.sha256` in the receiver copy is the fs2 tree's own (stale for `inflate.py` there too, as fs2 shipped
  it); `inflate.sh` does not read it. Not regenerated, to keep the tree diff at exactly two files.
- The canonical-equations JSONL ledger was NOT appended (mc1's pattern: module + export + guards; the ledger file
  carries another arm's uncommitted edits).
- Decode wall-clock deltas are NOT a mechanism claim: the shipped-archive control decode ran under heavier
  concurrent load than the candidate decodes.

## Custody (ALWAYS KEEP THE PAYLOAD)
Store: `/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/` — `inputs/` (ep634 EMA init `ff2d3e45…`,
cache `f29c479a…`, field `cc10a7b0…`), `smoke_e2/`, `rungs/{lambda_1p0,lambda_1p0_twin,lambda_0p5,cpu_control_jf2_null}/`
(every training checkpoint incl. `qat_stage_end_epoch_0060.pt`, raw IHS1 + all 12 Brotli representations, both
streams, candidate archive, receiver-copy runtime, decoded field, RUNG_RESULT / VERIFY_RESULT), `control/`
(shipped-runtime copy, decoded field, CONTROL_RESULT), `parseback/{shipped,lambda_1p0}/` (the two 3.66 GB renders
with sha, both retained), `LADDER_REPORT.json`. Small receipts mirrored under `.omx/research/ddm_cl2_capacity_20260905/`.
Measured peaks recorded in `.omx/state/measured_peaks.jsonl` (families `train_ddm_cl1_hpac_capacity`,
`ddm_cl2_hpac_prior_capacity_ladder`). Lane `lane_ddm_cl2_hpac_prior_capacity_ladder_20260905` at L2
(`impl_complete`, `real_archive_empirical`).

## Equations leg (`tac.canonical_equations`)
Registered as **`hpac_prior_capacity_slope_v1`** (`src/tac/canonical_equations/hpac_prior_capacity_slope_20260905.py`,
exported from `tac.canonical_equations`; re-derivation guards in `src/tac/tests/test_ddm_cl2_hpac_prior_capacity_slope.py`;
commit `65fd5ffa1`). The law: with the field held, the token subsystem's counted bytes are `J = B_model + B_stream`;
an adjacent rung pays iff `ΔB_stream/ΔB_model < −1` (`rung_pays`); the ladder is admissible only if the λ=1 control
lands within +500 B of the shipped joint (`control_reproduces_shipped_family`). Two anchors, both
VERIFIED_VIA_EMPIRICAL_ANCHOR: control (residual 41 B: J(1) = 126,885 vs 126,926) and the λ 1.0→0.5 secant
(residual 2,006 B: measured net +506 B against the predicted ≤ −1,500 B). `prior_law_prediction_holds(350, 156)`
is False — the falsifier fired.

## Frontier line
**fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]** — unchanged; this arm moved no pointer.
Candidate line: **ddm_cl2_lambda1_control_repack — 179,982 B, projected S 0.14781744131049854 (rate-only, DERIVED;
[macOS-CPU advisory] byte measurement) — READY-FOR-T4**, seal `e42288fc…`; the λ ladder itself: prior law FALSIFIED,
λ=0.25 not fired, capacity door on the multiplier coordinate CLOSED (formulation).
