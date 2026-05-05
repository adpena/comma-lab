---
name: 2026-04-30 RECOVERY-AGENT-4 Phase B — IMP dispatch ABORTED, pre-existing script gap discovered
description: Lightning Studio L40S (48GB) attached + pact bootstrapped successfully. IMP script dispatched but ABORTED after cycle 0 — discovered pre-existing gap: `train_imp_cycle.py` runs only a 3.3s "lightweight stub" loop (no real training), and Stage 1.5 per-cycle auth eval crashes with pose_dim mismatch (anchor renderer.bin pose_dim=6 vs auth eval builds with pose_dim=2). This is NOT a Lightning-introduced bug; the same broken IMP would have run on lost Vast.ai instance 35899275. Lane 17 IMP needs council redesign before another dispatch — not just re-dispatch.
type: project
originSessionId: recovery_agent_4_session
---

## Phase A: Lightning Studio bootstrap COMPLETE

- Studio: `lossy-compression-challenge` in `comma-lab` teamspace (project_id `01knw7m9kx9h8rt5rz5csmrdan`)
- Account: `adpena` (adpena@gmail.com), Lightning Pro, balance $47.38 of $240 annual credits
- SSH: `ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai` confirmed working
- GPU: **NVIDIA L40S 48GB** attached (H100 NOT available on this AWS cluster — `lit-h100-1 not found for this AWS cluster` error). L40S accepted via `lightning_sdk`.
- CUDA verified: torch 2.11.0+cu130, cuda_available=True, NVIDIA L40S 47.7GB
- `/home/zeus/pact` bootstrapped:
  - tac 1.0.5 installed via `uv pip install -e .`
  - venv at `/home/zeus/pact/.venv`
  - Anchors: `experiments/results/lane_g_v3_landed/iter_0/{renderer.bin, masks.mkv, optimized_poses.pt}` synced
  - Anchors: `experiments/results/lane_pfp16_stack_landed/archive_lane_g_v3_pfp16.zip` synced
  - Upstream: `upstream/videos/0.mkv` + `upstream/models/{posenet,segnet}.safetensors` synced
- Tarballs uploaded:
  - `/home/zeus/pact_for_lightning.tar.gz` (8.5G — main repo + `experiments/results/`)
  - `/home/zeus/pact_anchors_for_lightning.tar.gz` (2.0M)
  - `/home/zeus/pact_upstream_videos.tar.gz` (35.8M)

## Phase B: Lane 17 IMP dispatch ABORTED

### What I did
1. Created `scripts/lightning_lane_j_imp_iterative_magnitude_pruning.sh` adapted from Vast.ai variant:
   - WORKSPACE=/home/zeus/pact (not /workspace/pact)
   - PYBIN=/home/zeus/pact/.venv/bin/python (uv venv, not conda)
   - Stage 0 NVDEC probe replaced with simple GPU-presence check (Lightning L40S is known-good NVDEC)
   - Stage 4 auth eval will install `nvidia-dali-cuda130` lazily before invocation
   - Self-references in script renamed (lane_script provenance + dead-flag scan path)
2. SCP'd to Studio + chmod +x
3. Pre-flight passed (Stage 0 GPU check OK, Stage 0b parity OK, argparse dead-flag scan OK)
4. Dispatched via Pattern A nohup, PID 3273 on Studio
5. Cycle 0 finished in **3.3 seconds** with this log:
   ```
   [lane-j-imp] fine-tune: 200 epochs @ lr=0.0001 (in-script lightweight loop; deploy script swaps in train_distill)
   ```

### What broke
**Pre-existing bug class — NOT Lightning-specific:**

1. **`experiments/train_imp_cycle.py` is a STUB**, not real training. From source:
   ```python
   # The deploy script ``scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh``
   # OVERRIDES this stub by calling ``train_distill.py`` with a mask-applied ...
   ```
   But the deploy script does NOT actually swap in `train_distill.py`. Both Vast.ai script (`scripts/remote_lane_j_imp_*`) and my Lightning variant call `train_imp_cycle.py` directly, which produces a 3.3s stub-trained checkpoint (no real fine-tune).

2. **Stage 1.5 per-cycle auth eval crashes immediately on cycle 0** with shape mismatch:
   ```
   size mismatch for motion.head.weight: copying a param with shape torch.Size([6, 32, 3, 3])
       from checkpoint, the shape in current model is torch.Size([2, 32, 3, 3])
   ```
   The Lane G v3 anchor renderer has `pose_dim=6` but the cycle 0 model is built with `pose_dim=2` (default in cycle 0 training command). The cycle 0 training command line in the script DOES pass `--pose-dim 6`, so this is happening in the auth-eval-on-cycle subprocess (Stage 1.5 path) — which is built somewhere else.

### Original Vast.ai dispatch was likely identical-broken
Lost instance 35899275 was probably running the same broken script. We don't know how far it got before disappearing (lost cycle 0 ckpt has nothing to harvest anyway).

### Action taken
- Killed IMP process on Studio (`pkill -f lightning_lane_j_imp`)
- Studio still has L40S attached + pact bootstrapped — ready for Phase C (PFP16) and any future dispatch
- Lane 17 IMP needs council redesign before re-dispatch:
  1. Wire `train_distill.py` (or actual training loop) into the cycle loop
  2. Fix Stage 1.5 auth eval pose_dim plumbing
  3. Verify cycle artifacts have non-trivial training delta vs anchor

### What's still on Studio for Phase C
- L40S 48GB attached (~$1.80/hr — costs accruing while Studio is "Running")
- pact installed at `/home/zeus/pact` with venv + Lane G v3 anchors + PFP16 archive ready
- Phase C (PFP16 contest-CUDA eval) can run inline without re-dispatch

## Cost so far this session

- ~5 min Studio L40S boot/test = ~$0.15 of credits
- ~2 min IMP attempted run = ~$0.06 of credits
- Total: ~$0.20 of credits (negligible vs $47.38 balance)

## Outstanding work for next agent

- **Lane 17 IMP**: council redesign required (not just re-dispatch). Files affected:
  - `experiments/train_imp_cycle.py` — needs real-training mode, not stub
  - `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` Stage 1.5 — pose_dim plumbing for auth eval
  - Same fixes needed for Lightning variant
- Update `.omx/state/active_dispatches.md`: mark `lane_17_imp_10cycle_lightning_l40s` row as ABORTED:PRE_EXISTING_SCRIPT_BUG
- Phase C PFP16 dispatch — proceed in this session (inline on Studio L40S)

## Key non-monetary lesson

**Quota incident #4's "fresh dispatch" plan was based on the assumption that the script was sound and just lost an instance.** It wasn't. Lost instance 35899275 was running the same broken stub. **Lesson: before re-dispatching after instance loss, sanity-check the script ran ANY real work on the lost instance. Vast.ai usage history would have shown 0% GPU util during cycle 0 → "this script doesn't actually train" earlier.**

## Cross-refs

- /tmp/imp_migration_pivot_for_311.md — pivot doc that triggered this attempt
- project_quota_incident_4_recovery_state_20260430_1530.md
- feedback_lightning_ai_ssh_credentials_20260430.md
