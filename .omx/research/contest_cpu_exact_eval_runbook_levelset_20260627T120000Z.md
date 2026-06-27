# Contest-CPU exact-eval runbook — LEVEL-SET witness row (the means→ends linchpin)

**Fixed UTC:** 20260627T120000Z · **Authority:** advisory until a real byte-closed
contest-CPU (Linux x86_64) `upstream/evaluate.py` row lands. Pointer UNMOVED at
contest-CPU **0.19110**. NO score/promotion claim in this memo — it is the *procedure*.

**What this enables:** the moment the running row's per-stage checkpoint lands
(`experiments/results/levelset_n600_wpose1_20260627T103459Z/levelset_ckpt_stageTau_ep300.npz`,
~14h after launch 20260627T103459Z; full run ~70.5h), produce the pointer-relevant
exact row with zero surprises. Do NOT touch the running GPU row (daemon pid 17099 /
worker 17101). CPU-only here; keep ≥10GB free.

---

## 0. Readiness verdict (audited 20260627)

PATH IS **READY** end-to-end on contest-CPU via the **Modal CPU container**
(`experiments/modal_auth_eval_cpu.py`, debian_slim linux/x86_64 = 1:1 contest-CI
family). One runtime-closure bug was found and FIXED this session (inflate.sh used
bare `python`; now `${PYTHON:-python3}` — see §5). Remaining items are *measure-on-the-
real-checkpoint* gates (inflate wall-clock vs the 30-min budget; per-stage ckpt key
schema), not missing infrastructure.

| piece | status |
|---|---|
| byte-close level-set npz → archive.zip + inflate.py + inflate.sh | EXISTS `tools/levelset_byte_close_and_eval.py` (dry-run PASSED) |
| inflate.py self-contained (numpy+brotli+torch+scipy), emits full `(2·n_pairs,874,1164,3)` `.raw` | VERIFIED (12-frame smoke, exact bytes, self-orient/scipy path) |
| compliance: archive.zip = ONLY `0.bin`, NO scorer weights / SegNet / PoseNet / GT | VERIFIED PASS (1 member, 75,584 B) |
| inflate.sh 3-arg interface matches `contest_auth_eval._run_inflate` (`bash inflate.sh DATA_DIR OUT_DIR FILE_LIST`) | VERIFIED |
| contest-CPU host with full dep closure (incl. scipy) | EXISTS — Modal CPU image pins torch(cpu)+numpy<2+brotli+scipy+smp+safetensors |
| contest-CPU eval = `archive.zip → inflate.sh → upstream/evaluate.py --device cpu`, S recomputed from report.txt | EXISTS `experiments/contest_auth_eval.py` (cross-checks rounded fields) |

---

## 1. The one-command sequence (given a checkpoint npz)

```bash
# --- on the macOS control host (this repo), $0, CPU-only ---
cd /Users/adpena/Projects/pact
RUN=experiments/results/levelset_n600_wpose1_20260627T103459Z
CKPT=levelset_ckpt_stageTau_ep300.npz          # or whichever stage ckpt landed; `ls $RUN`

# (A) BYTE-CLOSE: full 600-code archive.zip + a fast 6-pair LOCAL inflate smoke + compliance.
#     --max-pairs 6 only caps the LOCAL validation; archive.zip still encodes ALL 600 codes
#     (the real rate term). --skip-parity avoids the heavy GT decode here (parity is advisory).
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir "$RUN" --npz-name "$CKPT" \
  --max-pairs 6 --skip-parity --keep-packet \
  --out reports/levelset_bc_stageTau_20260627T120000Z.json

# read the packet dir + archive sha/bytes from the report:
PKT=$(.venv/bin/python -c "import json;print(json.load(open('reports/levelset_bc_stageTau_20260627T120000Z.json'))['packet_dir'])")
SHA=$(.venv/bin/python -c "import json;print(json.load(open('reports/levelset_bc_stageTau_20260627T120000Z.json'))['byte_close']['archive_zip_sha256'])")

# (B) COMPLIANCE GATE (hard, fail-closed): archive.zip must be ONLY 0.bin, no scorer/GT.
.venv/bin/python - "$PKT" <<'PY'
import sys,zipfile
n=zipfile.ZipFile(sys.argv[1]+"/archive.zip").namelist()
bad=any(s in m.lower() for m in n for s in ("segnet","posenet","scorer","safetensors","gt","argmax","lstar"))
assert n==["0.bin"] and not bad, f"COMPLIANCE FAIL: members={n}"
print("COMPLIANCE PASS: archive.zip members =", n)
PY

# (C) CLEAN packet for upload (drop the local archive/ + inflated/ subdirs; n600 .raw is 3.6GB):
CLEAN=experiments/results/levelset_stageTau_packet_clean_20260627T120000Z
mkdir -p "$CLEAN" && cp "$PKT"/archive.zip "$PKT"/inflate.py "$PKT"/inflate.sh "$CLEAN"/

# (D) CLAIM the dispatch lane (cross-agent coord; required before paid eval):
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id levelset_n600_contest_cpu --platform modal_cpu \
  --instance-job-id "stageTau_$SHA" --agent claude:contest_cpu_runbook \
  --status active_eval_running --notes "level-set stageTau contest-CPU exact eval; advisory until report.txt"

# (E) CONTEST-CPU EXACT EVAL on Modal CPU (Linux x86_64, ~$0.06-0.15; detached):
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval_cpu.py \
  --archive "$CLEAN/archive.zip" \
  --submission-dir "$CLEAN" \
  --inflate-sh inflate.sh \
  --expected-archive-sha256 "$SHA" \
  --inflate-timeout 5400 --evaluate-timeout 5400 \
  --output-dir experiments/results/levelset_stageTau_contest_cpu_20260627T120000Z \
  --detach --provider-detach-ack \
  --lane-id levelset_n600_contest_cpu --instance-job-id "stageTau_$SHA"
# (harvest detached runs via tools/recover_modal_auth_eval.py if the entrypoint returns before the fn)
```

**Why `--submission-dir $CLEAN` + `--inflate-sh inflate.sh`:** the whole packet dir is
zipped and uploaded so `inflate.py` travels with `inflate.sh` (verified in
`tac.deploy.modal.auth_eval.prepare_modal_auth_eval_request`). `inflate-sh` is relative
to `--submission-dir`. The Modal image already has numpy+brotli+torch(cpu)+scipy.

## 2. Read the row (recompute S, do not trust the rounded field)

```bash
OUT=experiments/results/levelset_stageTau_contest_cpu_20260627T120000Z
.venv/bin/python - "$OUT/contest_auth_eval.json" <<'PY'
import json,sys,math
d=json.load(open(sys.argv[1]))
seg=d["avg_segnet_dist"]; pose=d["avg_posenet_dist"]; bytes_=d["archive_size_bytes"]
# upstream/evaluate.py:92 — rate = compressed / uncompressed(37,545,489 for the public set)
rate=bytes_/37_545_489
S=100*seg + math.sqrt(10*pose) + 25*rate
print(f"recomputed S={S:.5f}  (seg={seg:.6f} pose={pose:.6f} bytes={bytes_} rate={rate:.6f})")
print(f"axis={d.get('score_axis')} grade={d.get('evidence_grade')} n={d.get('n_samples')}")
print("PROMOTABLE row?" , d.get("n_samples")==600 and "contest-cpu" in str(d.get("evidence_grade","")).lower())
PY
```

## 3. If S < 0.19110 (a real lower row): move the pointer

```bash
# only with a 600-sample contest-CPU report.txt on Linux x86_64:
.venv/bin/python tools/refresh_canonical_frontier.py            # inspect
# then update .omx/state/canonical_frontier_pointer.json via the canonical refresher,
# append a MEMORY one-liner, and a DAG feed citing archive sha + bytes + the report.txt path.
# terminal the claim:
.venv/bin/python tools/claim_lane_dispatch.py claim --force \
  --lane-id levelset_n600_contest_cpu --platform modal_cpu \
  --instance-job-id "stageTau_$SHA" --agent claude:contest_cpu_runbook \
  --status completed_contest_cpu_eval --notes "S=<recomputed>; report.txt=$OUT/report.txt"
```

If S ≥ 0.19110: HONEST negative — pointer UNMOVED; record the row as a per-stage advisory
anchor; the descent continues (tau→l7→Muon stages give later, lower rows).

---

## 4. Checkpoint-landing checklist (verify on the REAL tau npz; the byte-close tool fails closed)

1. **npz name** — per-stage ckpts are NOT `*_ema_mlx.npz`; pass `--npz-name <actual>` (`ls $RUN`).
2. **required keys** — the tool requires `code, in_proj.weight, out_sdf.weight, out_tex.weight,
   palette` (+ `film`/`hidden.*`). If the per-stage ckpt stores optimizer state or a different
   layout it raises a clear `ValueError` (NO-FAKE) — then byte-close from the EMA-shadow save block.
3. **cfg scalars** — the tool reads `__cfg_*/__bank_*/__render_hw` from the npz; if the per-stage
   block omits them it WARNS and uses defaults (render defaults 384×512 — may be WRONG). Verify the
   ckpt carries `__render_hw` + `__bank_*` or the deploy render diverges from training.
4. **self-orient** — SEALED config is self-orient (max-bank-freq 64 → in_feat 88, dir_w 8,
   n_dir_freqs 2). inflate needs scipy (Modal CPU image HAS it). The deploy fixed-point converges on
   the FINAL weights → realized d_seg on the inflated frames is the TRUTH (a small gap vs the
   trainer's trajectory number is itself a finding, not a bug).
5. **pose-blind check** — if realized `d_pose` > ~1.0 the render is POSE-BLIND (the w_pose=0 failure);
   the n600 run is `--w-pose 1.0` so expect d_pose → O(1e-3). A stored pose sidecar does NOT fix a
   pose-blind render (the scorer runs PoseNet on the FRAMES).

---

## 5. BLOCKER fixed this session + the one remaining GATE

**FIXED (runtime closure):** `tools/levelset_byte_close_and_eval.py` `_INFLATE_SH` used bare
`python`. `contest_auth_eval._run_inflate` runs `bash inflate.sh ...` and sets `PYTHON=sys.executable`,
but the template ignored it → on macOS / many Linux CPU hosts (`python` absent, only `python3`) the
inflate died `python: command not found` (silent — the loop masked the rc). Now
`PYBIN="${PYTHON:-python3}"`: honors the canonical `PYTHON` env (deps-complete eval interpreter) and
falls back to `python3` (fails LOUD with `ModuleNotFoundError` if deps missing, not silently). Does
NOT change archive.zip bytes (rate term) nor rendered frames (inflate.py untouched) → score unaffected.
VERIFIED: `PYTHON=<venv> bash inflate.sh …` → 0.raw exact 36,624,096 B, rc=0.

**REMAINING GATE (measure on the n600 tau ckpt, do NOT guess):** the **30-min inflate budget**
(`README.md` contest CPU = 4 cores / 16GB). The n6 smoke (render 96×128) inflates fast, but n600 is
600 pairs × 1200 frames at the trained render res (likely 384×512 = ~16× the pixels) + per-pair
self-orient fixed-point. If inflate > 30 min on contest CPU the submission is INVALID (an *advisory*
contest-CPU row can still use `--inflate-timeout 5400`). MEASURE the Modal-reported inflate elapsed; if
> 1800s, the inflate.py numpy forward needs vectorization-across-pairs / fewer `so_iters` / lower
render res BEFORE a *legal* submission — the advisory row is still informative for the score itself.

**Cost:** Modal CPU ~$0.06/hr; 600-sample CPU eval ~60-120 min + inflate ~ tens of min ⇒ ~$0.06-0.15
plus image build/startup. Spend it on the tau/l7/Muon checkpoints — NOT on an early junk checkpoint.

---

## Cross-refs
- DAG: `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` FEED-ee (this), FEED-ed (the launched row), FEED-dy (pose path).
- Tools: `tools/levelset_byte_close_and_eval.py` · `experiments/contest_auth_eval.py` · `experiments/modal_auth_eval_cpu.py` · `upstream/evaluate.py` (score line 92) · `upstream/frame_utils.py` (TensorVideoDataset `.raw` contract).
- Authority discipline: CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA", "MPS auth eval is NOISE", "Frontier scores are pointer-only", deterministic-reproducibility + rule-118.
