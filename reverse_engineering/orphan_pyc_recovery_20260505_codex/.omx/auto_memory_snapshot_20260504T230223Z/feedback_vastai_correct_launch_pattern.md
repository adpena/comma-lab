---
name: Vast.ai correct launch + SSH pattern (per docs.vast.ai 2026-04-27)
description: There's NO --wait/--block flag on `vastai create instance`. Manual polling is the documented pattern. Container pull takes 3-10+ min on slow hosts; filter offers by `inet_down>800` to get hosts that pull the 5GB PyTorch image quickly.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified 2026-04-27 against https://docs.vast.ai/cli/commands and https://docs.vast.ai/documentation/instances/connect/ssh.**

**The official lifecycle pattern:**
1. `vastai create instance <offer_id> --image <img> --disk <gb> --label <name> --raw` — returns instantly with `new_contract` ID
2. Poll `vastai show instance <id> --raw` until `actual_status == "running"` AND `cur_state == "running"`. Both must be `running`. If `cur_state == "stopped"`, the contract was created but the host never started it (host issue, kill + relaunch).
3. After `running`, the container is BOOTING (still pulling image). Poll SSH connectivity:
   ```
   ssh -p <port> root@<host> "echo OK"
   ```
   This will return `Connection refused` until sshd is up inside the container.
4. After SSH succeeds, BUT before doing 30 min of setup, verify the instance won't die:
   - Check `vastai logs <id>` for `"remote port forwarding failed"` → port collision, instance is DEAD even though SSH might briefly accept (kill + relaunch). See `feedback_vastai_port_forward_fail`.
   - For CUDA workloads using DALI: NVDEC probe (see `feedback_vastai_nvdec_host_variation`).
5. Then run `vastai attach ssh <id> <key>` if your SSH key isn't already on account default.
6. `vastai ssh-url <id>` gives a shortcut connection string.

**Container pull timing (5GB pytorch:2.5.1-cuda12.4-cudnn9-devel image):**
- Fast hosts (`inet_down > 800 Mbit/s`): 1-3 min
- Slow hosts (`inet_down ~200 Mbit/s`): 5-10 min
- Some hosts (Indiana 0.9974-reliable today): 10+ min, abandon

**Filtering for fast-loading hosts:**
```
vastai search offers 'gpu_name=RTX_4090 reliability>0.99 num_gpus=1 disk_space>=30 inet_down>800 geolocation in [US,CA]' -o 'dph_total'
```
The `inet_down>800` is the key filter — slow inet hosts kill iteration speed.

**What to do when stuck loading > 5 min:**
- `vastai logs <id> | tail` — if `"No such container"`, the docker pull is still in flight or stuck.
- If 10+ min and still stuck: kill, pick a higher inet_down host, re-launch.

**No --wait flag exists.** The docs explicitly note "no `--block` or `--wait` flags mentioned for instance creation. ... You would need to manually poll instance status."

**My past mistakes captured here:**
1. Indiana 27290 (reliability 0.9974 but inet_down<800) → 10+ min loading, abandoned (~$0.05 wasted on stopped contract).
2. Nevada 35688167 → cur_state stuck on "stopped" (host issue) → killed.
3. California 35666793 → fast load but NVDEC failed → caught at probe stage.
4. Spot interruption mid-experiment (Oregon 35676495 at 3h) — only mitigation is reliability >0.99 OR resumable training.

**Total time wasted by suboptimal host filter today:** ~30 min wall + ~$0.50 of stopped-contract billing. Filter `inet_down>800` would have eliminated all of it.
