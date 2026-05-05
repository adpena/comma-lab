---
name: Vast.ai NVDEC exposure varies by physical host — CUDA visible ≠ DALI video works
description: Two RTX 4090 instances on the same Vast.ai image, same driver version, same CUDA version. One (Oregon) ran upstream/evaluate.py via DALI fine. The other (California) failed DALI's video MIXED operator with "CUDA_ERROR_NO_DEVICE (100): no CUDA-capable device is detected" — even though /opt/conda python could see the GPU. The blocker is host-side NVDEC exposure, not CUDA compute.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified 2026-04-27 across two RTX 4090 instances on the same `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` image, both with driver 580.95.05+ and CUDA 13 reported by `nvidia-smi`:**

- Instance 35665497 (Oregon, ssh9.vast.ai): DALI `fn.experimental.inputs.video` initialized cleanly. Full upstream/evaluate.py ran to completion (53.60 score landed). Worked.
- Instance 35666793 (California, ssh5.vast.ai): same image, same image_pull, same nvidia-dali-cuda120 install. DALI's video MIXED operator failed with `CUDA driver API error CUDA_ERROR_NO_DEVICE (100): no CUDA-capable device is detected`. Even after switching to `nvidia-dali-cuda130` and forcing `CUDA_VISIBLE_DEVICES=0`, identical error.
- Confirming basic CUDA still worked on the failing host: `python -c "import torch; print(torch.cuda.is_available())"` → True. `inflate_renderer.py` (pure torch on cuda) ran successfully and produced a correct 3.66 GB .raw file. So compute CUDA is fine — what's missing is NVDEC.

**Root cause:** DALI's video decoder uses NVDEC (NVIDIA's hardware video decoder, a separate hardware unit from the compute cores). Vast.ai docker hosts can pass through compute CUDA without exposing NVDEC if the container runtime config or driver capabilities don't include `video` in `NVIDIA_DRIVER_CAPABILITIES`.

**How to apply:**
1. **Before any GPU spend on a Vast.ai instance that needs upstream/evaluate.py with --device cuda**, run a 5-second NVDEC probe:
   ```bash
   ssh ... "python -c 'from nvidia.dali import pipeline_def, fn; from nvidia.dali.types import DALIImageType
   @pipeline_def(batch_size=1, num_threads=1, device_id=0)
   def p():
       return fn.experimental.inputs.video(name=\"x\", sequence_length=2, device=\"mixed\")
   pipe = p(); pipe.build(); print(\"NVDEC OK\")'"
   ```
   If this fails, destroy the instance and pick a different host. Don't burn 30 min of setup on a host that will fail at the eval stage.
2. **Filter Vast.ai offers by `nvidia_driver_capabilities=video,compute,utility`** if the API exposes it. Otherwise, prefer hosts that have run successful DALI workloads recently (the `inet_down` field is a weak proxy for "modern host config").
3. **Track which providers reliably expose NVDEC.** This session: Oregon (id=27694297-ish) worked; California (35563940 / 35666793) didn't. Log the geo + machine_id of working instances over time.
4. **Fallback path** if a session must use a no-NVDEC host: pre-decode the GT video to a tensor on a separate machine and ship the tensor instead of running DALI. This breaks "1:1 contest compliance" so it's last-resort only.

**Cost of this trap:** $0.20 in setup + 30 min wall time on the bad California instance before discovering DALI failure at the evaluate.py stage. The probe-first protocol above prevents the second occurrence.

**Repeat occurrence 2026-04-27 (Texas inst 35691284, ssh6.vast.ai port 11284):** Lane A pose TTO ran cleanly for 3.4h (12,323s, 600/600 pairs, optimized_poses.{pt,bin} saved). The downstream auth eval invoked `contest_auth_eval.py → upstream/evaluate.py` and crashed with the SAME `CUDA_ERROR_NO_DEVICE (100)` from `nvidia.dali.fn.experimental.inputs.video`. NVDEC was missing on this host but compute CUDA was fine — the entire pose TTO ran on torch.cuda without ever needing DALI. Cost: $0.85 of GPU time, 3.4h wall, no auth-eval result on this run (artifacts pulled and reusable). Mitigation worked partially: the bootstrap script's NVDEC probe stage (`scripts/remote_lane_a_pose_tto.sh`) MUST run BEFORE pose TTO start — currently the probe is structured to gate setup but didn't fire on this host. Audit `scripts/remote_setup_full.sh` Stage 0 NVDEC probe to confirm it's being invoked.

**Permanent fix wanted:** add a `check_vastai_nvdec_probe_runs_first` preflight gate that scans every `scripts/remote_*.sh` and verifies the NVDEC probe is in Stage 0 (BEFORE pose TTO / training launches). The probe is cheap; the consequence of a missing probe is hours of wasted GPU. This is a meta-bug worthy of a preflight check — add to the meta-bug scanner queue.

**Permanent fix LANDED 2026-04-27 (commit eef64293 / 9c0fbe36 / 4f8a9667):**
- `scripts/probe_nvdec.sh` — single canonical probe with --ensure-dali flag.
- All 4 lane scripts + 3 bootstrap scripts now invoke the probe at Stage 0.
- `check_remote_scripts_have_nvdec_probe` (preflight strict 7f2740e4) catches any new remote_*.sh that omits the probe.

**Repeat occurrence 3 — 2026-04-27 Lane A relaunch (Hungary inst 31757063):**
The strengthened-via-Stage-0 probe PASSED on this host but the actual contest_auth_eval still failed at runtime with `CUDA_ERROR_NO_DEVICE` — at `pipe.share_outputs()` not `pipe.build()`. The previous probe only validated handle allocation; NVDEC hardware decode was never actually exercised. Wasted 22 min on the host before falling back.

**Probe strengthened 2026-04-27 (commit pending):** the probe now synthesizes a tiny in-memory MP4 via PyAV, feeds it through `fn.experimental.inputs.video`, runs `schedule_run` + `share_outputs`, and asserts the decoded output shape. If NVDEC is missing, share_outputs() raises here in 5 seconds — not 22 min later in auth_eval. **This closes the third occurrence pattern; total cost over the 3 incidents: ~$1.30 + ~6h wall.**

**Lane A actual auth-eval result:** 1.15 [contest-CUDA] vs baseline 2.29 (49.8% reduction; just outside predicted 0.85-1.10 by 0.05). PoseNet got the predicted 23× drop (0.247 → 0.005); rate inflated 2× (337KB → 694KB) because the new optimized_poses + masks added bytes. Lane B-alt brotli should reclaim ~0.023.

**Probe FALSE-NEGATIVE class — 2026-04-27 Lane G launch (Washington inst 35708461):**
The 64-bit-base64 16x16 fixture in scripts/probe_nvdec.sh (commit 910ea515) is BELOW the NVDEC `nMinWidth=48` threshold on driver 580.126.09. Result: hosts that DO have NVDEC reject the 16x16 fixture and the probe exits 2 ("destroy host"). Lane G subagent had to write a workaround `probe_nvdec_override.sh` that decodes the actual `upstream/videos/0.mkv` (1164×874) instead.

**Permanent fix (queued for follow-up commit after codex-fix subagent lands):**
- Replace 16x16 fixture with 64x64 (above the nMinWidth=48 threshold).
- 64x64 base64 fixture saved at `.omx/state/nvdec_probe_fixture_64x64.b64` (2173 bytes, 29 lines).
- Generated via: `ffmpeg -y -f lavfi -i color=black:s=64x64:d=2:r=1 -pix_fmt yuv420p -c:v libx264 -t 2 tiny.mp4 && base64 -i tiny.mp4 | fold -w 76`.
- Codex round 6 Finding #4 fix (in flight) is also adding error-text classification (exit codes 1-5 distinguish DALI-missing / NVDEC-missing / fixture-corrupt). The two fixes compose: classified errors AND a fixture that doesn't trigger nMinWidth.

**Lane G result (incomplete — destroyed at $0.20 spend):** KL distill weight=1.0 was 14000× too large vs scorer hinge — KL dominated the loss. Re-launch needed with weight ≈ 0.01 OR 0.001. Plus ETA was 11h vs $1.50 cap — bp=4 doubled the wall-time vs Lane A's bp=8 due to KL adding a second SegNet forward+backward.
