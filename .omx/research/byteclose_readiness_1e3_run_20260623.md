# Byte-close → exact-eval READINESS for the LIVE 1e-3 Muon run (re-pointed Branch-A, zero-latency)

**Run:** `experiments/results/yousfi_r3_MUONJUMP_stage8_lr1e3_20260623T180100Z`
(bc20, `taper_channels = [16,16,17,19,19,14,10]`, latent_dim 28, n_pairs 600, EMA decay 0.999).
**Supersedes the taper assumption in** `decisive_run_branchA_byteclose_exacteval_runbook_20260622.md`
(that runbook targeted the VENDORED-taper run `yousfi_r3_taper_marginhinge_e5`; THIS run uses an
EXPLICIT configurable taper, which changes the inflate-runtime assembly — see the TAPER VERDICT).
**Authority:** pointer UNMOVED 0.19110. Nothing fires until advisory S crosses. No score claimed.

---

## TAPER RECONSTRUCTION VERDICT — THE fire-time gotcha, RESOLVED (NOT metadata-driven; explicit fix required)

The G3 inflate runtime **does NOT auto-correct for this run's taper.** Two independent proofs:

1. **The archive bytes do NOT carry the taper.** Parsed `best/best_archive.bin` (79,592 B) embedded
   meta = `{n_pairs, latent_dim, base_channels:20, eval_size}` — **NO `channels`/`taper_channels` key.**
   (`codec.build_archive` only writes `meta_dict`, and the driver passes the vendored 4-key meta.)
2. **The G3 inflate path builds a DIFFERENT architecture.** `g3 .../submission_dir/inflate.py` imports
   the vendored `model.HNeRVDecoder`, which DERIVES its taper from a single `base_channels` via the
   fixed HNeRV formula (`model.py:21`): `[C,C,C,int(C*0.75),int(C*0.58),int(C*0.5),int(C*0.5)]`.
   For `C=20` that is **`[20,20,20,15,11,10,10]`** — NOT `[16,16,17,19,19,14,10]`.

Consequence: dropping THIS run's `archive.zip`/`0.bin` into the unmodified G3 packet and running
`inflate.sh` will **FAIL `decoder.load_state_dict(decoder_sd)`** — not just on tensor shapes but on
state-dict KEYS (e.g. stage 1→2 is `16→17`, so a `skips.1.weight` 1×1 conv key EXISTS in the trained
weights but the vendored-taper model builds `skips.1 = nn.Identity()` with no such key, and vice-versa).
**This is a hard architecture mismatch, not a one-line taper-param tweak.**

### The exact fix (clean, well-defined — verified verbatim-key-compatible)

`tac.torch_vehicle.configurable_taper_decoder.ConfigurableTaperHNeRVDecoder` is the FAITHFUL
generalization of the vendored decoder: its `__init__` body + `forward` are **verbatim identical** to
`model.HNeRVDecoder` except `self.channels = list(channels)` replaces the derived formula. State-dict
keys + module order + the Identity-vs-1×1-skip choice are identical at any taper. So the inflate-runtime
fix is a drop-in model swap + threading the taper in out-of-band (it is NOT in the archive):

- **Swap `<pkt>/src/model.py`** so the inflate-side `HNeRVDecoder` is the configurable-taper decoder
  (copy `src/tac/torch_vehicle/configurable_taper_decoder.py` content; expose its class as `HNeRVDecoder`
  OR edit inflate.py's import).
- **Hardcode the taper in `<pkt>/inflate.py`** (since the archive can't supply it):
  `TAPER = [16,16,17,19,19,14,10]` and pass `channels=TAPER` to the decoder ctor.

The byte-close + parity ACTUATOR is ALREADY taper-aware — `tools/verify_e2e_byte_close_eval.py
--taper-channels 16,16,17,19,19,14,10` reconstructs the `ConfigurableTaperHNeRVDecoder` and runs the
G2 fixed-point parity. The gap is ONLY in the assembled INFLATE RUNTIME TREE (`--keep-packet` copies the
PRISTINE VENDORED tree from `data/working/upstream/submissions/hnerv_muon/`, which is fixed-taper).

---

## Fire condition (read off the live summary; do NOT touch the run)
Binding term is d_seg. Current (last_eval, ep 25075): d_seg 0.00212, d_pose 3.3e-4, archive ≈ 79,592 B.
The **pose TERM** = √(10·d_pose) ≈ 0.05745 and the **rate TERM** = 25·79592/37545489 ≈ 0.05300, so
pose+rate **TERM** ≈ **0.1104** (NOT 0.0024 — that earlier figure was the bare rate *fraction* bytes/N
with the ×25 dropped). The live break-even (via the canonical `tac.contest_score.break_even_d_seg`):
* **beat 0.19110 → advisory d_seg < ~8.07e-4** (current 0.00212 is **~2.63× ABOVE** this — NOT close yet)
* **sub-0.15 → advisory d_seg < ~3.96e-4**
Fire when `torch_vehicle_summary.json:last_eval.score` (or `best_score`) **< 0.19110**.
(G3 proved advisory↔contest-CPU ≈ 0.001%, so the advisory cross IS the trustworthy trigger.)

> CORRECTION 2026-06-23 (canonical-score-helper hardening): the prior version of this block read
> "beat → d_seg < ~1.89e-3 (≈1.1× below 0.00212)" and "sub-0.15 → ~1.48e-3". Those were WRONG — they
> computed pose+rate as the bare rate FRACTION (bytes/N ≈ 0.0024) instead of the rate TERM (25·bytes/N ≈
> 0.053) plus the pose TERM (√(10·d_pose) ≈ 0.057). The ×25 on the rate term was dropped. The corrected
> break-even to beat is **8.07e-4** (the run is ~2.63× away, not 1.1×). The 06-22 runbook's 8.1e-4 was
> CORRECT all along (its pose+rate budget was similar once the terms are computed right). Always re-derive
> at fire time via the canonical helper: `from tac.contest_score import break_even_d_seg;
> break_even_d_seg(0.19110, d_pose, archive_bytes)` (it carries the ×25 — a hand-rolled inline formula is
> how the ×25 got dropped in the first place).

---

## Proven assets (verified this turn)
* **Payload:** `<1e3>/best/best_archive.bin` (EMA-shadow, byte-closed by the run; 79,592 B today — will
  change as best/ updates). `best/best_ema_decoder.pt` + `best/best_ema_latents.pt` + `best/best_meta.json`.
  `best_meta.json` carries `base_channels:20`; the TAPER is in `torch_vehicle_checkpoint_manifest.json:
  taper_channels`. ⚠️ `best_archive.bin` may NOT exist / be stale if no best-eval has landed at fire time
  — re-run the byte-close from the EMA `.pt` files (Step 1B) rather than trusting a stale `.bin`.
* **Byte-close + NO-FAKE parity actuator (taper-aware):** `tools/verify_e2e_byte_close_eval.py`
  (`--ckpt-dir <1e3>/best --taper-channels 16,16,17,19,19,14,10 --keep-packet`).
* **Minimal archive→zip:** `tools/build_torch_vehicle_d2_archive_zip.py` (`--input-bin ... --output-zip ...
  --member-name 0.bin`, ZIP_STORED).
* **Inflate runtime tree to CLONE:** `experiments/results/g3_torch_vehicle_bc20_packet_20260618T012713Z/
  submission_dir/` (`0.bin`+`archive.zip`+`inflate.py`+`inflate.sh`+`src/`+`README.md`). inflate.sh is
  taper-AGNOSTIC (just `inflate.py SRC DST`); ONLY `src/model.py` + inflate.py decoder ctor are taper-bound.
* **Vendored src (the fixed-taper source `--keep-packet` copies):** `data/working/upstream/submissions/
  hnerv_muon/{inflate.py,src/model.py}` — confirmed fixed-taper; this is WHY the manual swap is required.
* **Authoritative hosts:** CPU `experiments/modal_auth_eval_cpu.py` (asserts cuda_available==False);
  CUDA `experiments/modal_auth_eval.py --gpu T4`. Paired template: `.omx/research/
  g3_paired_dispatch_plan_20260618T013243Z.json`. Cost ~$1–3 (within $20 Modal budget).

---

## Fire-time command sequence (re-pointed to the 1e-3 run)

Set once:
```bash
RUN=experiments/results/yousfi_r3_MUONJUMP_stage8_lr1e3_20260623T180100Z
TAPER=16,16,17,19,19,14,10
TS=$(date -u +%Y%m%dT%H%M%SZ)
PKT=experiments/results/yousfi_r3_1e3_bc20_packet_${TS}/submission_dir
SCRATCH=/Volumes/VertigoDataTier/pact/.omx_tmp/byteclose_1e3_${TS}   # SSD tier, auto-clean
```

### Step 1 — Byte-close + G2 parity (taper-aware actuator; ALSO assembles a packet)
```bash
OMP_NUM_THREADS=2 .venv/bin/python tools/verify_e2e_byte_close_eval.py \
    --ckpt-dir ${RUN}/best \
    --taper-channels ${TAPER} \
    --keep-packet \
    --out reports/e2e_byte_close_eval_1e3_${TS}.json
# -> reports parity_ok + the assembled packet_dir + the archive.zip st_size (the contest rate input).
```
*(1B fallback if `best/best_archive.bin` is stale/absent: the actuator rebuilds the archive from the EMA
`.pt` files in `--ckpt-dir`, so Step 1 is self-sufficient. The minimal `build_torch_vehicle_d2_archive_zip.py`
path is a backup only if you already trust a current `best_archive.bin`.)*

### Step 2 — Assemble the packet and APPLY THE TAPER FIX (the one fire-time edit)
```bash
cp -r experiments/results/g3_torch_vehicle_bc20_packet_20260618T012713Z/submission_dir ${PKT}
# Drop in the freshly byte-closed bytes from Step 1's kept packet_dir (archive.zip + 0.bin):
cp <STEP1_PACKET_DIR>/archive.zip ${PKT}/archive.zip
cp <STEP1_PACKET_DIR>/0.bin       ${PKT}/0.bin
# THE TAPER FIX — make the inflate runtime build the configurable-taper decoder:
cp src/tac/torch_vehicle/configurable_taper_decoder.py ${PKT}/src/model.py   # verbatim-key-compatible
#   then EITHER expose its class as HNeRVDecoder in src/model.py,
#   OR edit ${PKT}/inflate.py: `from model import ConfigurableTaperHNeRVDecoder as HNeRVDecoder`
#   AND add `TAPER=[16,16,17,19,19,14,10]` + pass `channels=TAPER` to the ctor (line ~33).
```
> If Step 1's `--keep-packet` already emits a runnable submission_dir, prefer THAT as `${PKT}` and apply
> ONLY the taper fix to its `src/model.py`+`inflate.py`. (Verify whether `_assemble_contest_packet` copied
> the fixed-taper vendored tree — it does today — so the taper fix is required either way.)

### Step 3 — Local runtime-closure smoke (GATE, $0; catches the taper mismatch BEFORE spending)
```bash
mkdir -p ${SCRATCH}/archive ${SCRATCH}/inflated
cp ${PKT}/0.bin ${SCRATCH}/archive/0.bin
printf '0.bin\n' > ${SCRATCH}/names.txt
OMP_NUM_THREADS=2 PYTHON=.venv/bin/python \
    bash ${PKT}/inflate.sh ${SCRATCH}/archive ${SCRATCH}/inflated ${SCRATCH}/names.txt
# ASSERT: inflate.sh exits 0 + writes ${SCRATCH}/inflated/0.raw (1200 frames * 874*1164*3 bytes).
# ASSERT (parity): the inflated frames == the run's in-process render of the SAME byte-closed decoder.
#   (verify_e2e already proves byte-close fixed-point; this proves the RUNTIME TREE reconstructs it.)
rm -rf ${SCRATCH}   # ~3.66 GB raw, rebuildable; auto-clean
```
**If inflate.sh raises a load_state_dict / shape / key error here → the taper fix in Step 2 is wrong;
DO NOT spend. Fix and re-smoke.**

### Step 4 — Claim the paired lanes (cross-agent coordination)
```bash
ARCSHA=$(shasum -a 256 ${PKT}/archive.zip | awk '{print $1}')
ARCBYTES=$(stat -f%z ${PKT}/archive.zip)
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id lane_yousfi_r3_1e3_bc20_exact_eval_${TS}_contest_cpu \
    --platform modal --instance-job-id yousfi_r3_1e3_bc20_${TS}_cpu \
    --status active --agent "claude:byteclose_1e3" \
    --notes "1e3 Muon byte-closed exact CPU eval; taper ${TAPER}; sha=${ARCSHA}; bytes=${ARCBYTES}; advisory S crossed 0.19110"
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id lane_yousfi_r3_1e3_bc20_exact_eval_${TS}_contest_cuda \
    --platform modal --instance-job-id yousfi_r3_1e3_bc20_${TS}_cuda \
    --status active --agent "claude:byteclose_1e3" --notes "paired CUDA T4 carry axis"
```

### Step 5 — Fire the paired Modal eval (re-point of the g3 dispatch plan)
```bash
# CPU (leaderboard axis):
.venv/bin/modal run --detach experiments/modal_auth_eval_cpu.py \
    --archive ${PKT}/archive.zip --expected-archive-sha256 ${ARCSHA} \
    --inflate-sh inflate.sh --submission-dir ${PKT} \
    --output-dir experiments/results/modal_auth_eval_cpu/yousfi_r3_1e3_bc20_${TS}_cpu \
    --detach --provider-detach-ack \
    --pair-group-id yousfi_r3_1e3_bc20_${TS} \
    --lane-id lane_yousfi_r3_1e3_bc20_exact_eval_${TS}_contest_cpu \
    --instance-job-id yousfi_r3_1e3_bc20_${TS}_cpu --claim-agent "claude:byteclose_1e3"
# CUDA (carry axis, drifts; NOT the pointer):
.venv/bin/modal run --detach experiments/modal_auth_eval.py \
    --archive ${PKT}/archive.zip --expected-archive-sha256 ${ARCSHA} \
    --inflate-sh inflate.sh --submission-dir ${PKT} --gpu T4 \
    --output-dir experiments/results/modal_auth_eval/yousfi_r3_1e3_bc20_${TS}_cuda \
    --detach --provider-detach-ack \
    --pair-group-id yousfi_r3_1e3_bc20_${TS} \
    --lane-id lane_yousfi_r3_1e3_bc20_exact_eval_${TS}_contest_cuda \
    --instance-job-id yousfi_r3_1e3_bc20_${TS}_cuda --claim-agent "claude:byteclose_1e3"
```
*(Confirm the EXACT `modal_auth_eval.py` arg names with `grep add_argument`/local_entrypoint before firing;
verified for `_cpu`, the CUDA file's surface should mirror it + `--gpu`.)*

### Step 6 — Harvest + recompute S BY HAND (the rounded `final_score` LIES)
For BOTH axes, read d_seg / d_pose / archive_bytes from the harvested JSON and recompute:
```
S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37_545_489
```

### Step 7 — Verdict + pointer
If **contest-CPU S < 0.19110**: run `scripts/pre_submission_compliance_check.py --contest-final --strict`
(`--expected-archive-sha256 ${ARCSHA} --expected-archive-size-bytes ${ARCBYTES}` + the auth-eval JSON),
then update `.omx/state/canonical_frontier_pointer.json` (the FIRST ours-trained pointer move). If only
close: record the calibration row, keep descending (the d_seg loop continues), re-fire on the next cross.

---

## VERIFIED vs FILLS-AT-FIRE

| Item | Status |
|---|---|
| Archive embeds the taper? | **VERIFIED NO** (meta = 4 keys, no taper) — explicit fix required |
| ConfigurableTaper keys/forward == vendored (clean swap)? | **VERIFIED** (verbatim `__init__`/`forward`, only `self.channels`) |
| Byte-close actuator taper-aware? | **VERIFIED** (`--taper-channels`) |
| Inflate runtime tree taper-aware? | **VERIFIED NO** (copies fixed-taper vendored src) → Step 2 fix |
| inflate.sh contract / member name | **VERIFIED** (`inflate.py SRC DST`, member `0.bin`, ZIP_STORED) |
| CPU + CUDA eval files exist | **VERIFIED** (`modal_auth_eval_cpu.py` + `modal_auth_eval.py --gpu T4`) |
| Fire-time break-even d_seg | **RE-DERIVE at fire** via `tac.contest_score.break_even_d_seg(0.19110, d_pose, archive_bytes)` (carries the ×25; current ≈ 8.07e-4, run ~2.63× away — see corrected Fire-condition block) |
| `best/best_archive.bin` current at fire | **FILLS** (rebuild via Step 1 from EMA `.pt` if stale) |
| Archive sha / size | **FILLS** (`shasum`/`stat` at fire) |
| modal_auth_eval.py (CUDA) exact arg names | **FILLS** (grep before firing) |

---

## NO-FAKE ledger
- The taper-reconstruction blocker is PROVEN by parsing the live archive bytes (no taper key) + reading
  both decoder sources (vendored derives taper from base_channels; this run uses an explicit 7-stage list).
- The fix is PROVEN clean (configurable-taper decoder is verbatim-key-compatible). Byte-close + dual Modal
  eval PROVEN for this vehicle class (G3/#127). Advisory↔contest-CPU ≈0.001% (G3).
- NOT claimed: NO score moved; pointer UNMOVED 0.19110. The authoritative row exists ONLY after
  upstream/evaluate.py runs on the byte-closed bytes. Fire condition (advisory S < 0.19110) NOT yet met
  (current best advisory 0.3066). Nothing fires until a real cross.
- RESIDUAL FIRE-TIME RISK: (1) the taper edit (Step 2) is manual — the Step-3 $0 smoke is the mandatory
  gate that catches a wrong edit BEFORE any spend; (2) if a future driver build embeds `taper_channels`
  into the archive meta + ships a metadata-driven inflate, this manual fix becomes obsolete (re-verify).

Cross-refs: `decisive_run_branchA_byteclose_exacteval_runbook_20260622.md` (vendored-taper predecessor) ·
`g3_torch_vehicle_bc20_first_exact_row_20260618T0135Z.md` (the proven dual row) ·
`g3_paired_dispatch_plan_20260618T013243Z.json` (the dispatch template re-pointed here) ·
`tools/verify_e2e_byte_close_eval.py` · `src/tac/torch_vehicle/configurable_taper_decoder.py`.
