---
name: Vast.ai SSH ControlMaster (kill the timeouts)
description: Vast.ai's edge SSH proxies (ssh*.vast.ai) rate-limit per-IP handshakes hard. ControlMaster keeps one TCP/auth alive and multiplexes all probes.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Problem:** Per-probe SSH from monitors (every 60–120s × 2+ instances) triggers `ssh: connect to host ssh*.vast.ai port NNNN: Connection refused` and `Operation timed out` repeatedly. The instances are FINE — the edge proxy is throttling our IP.

**Fix:** Add to `~/.ssh/config`:
```
Host *.vast.ai
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 30m
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ConnectTimeout 30
```
Plus `mkdir -p ~/.ssh/cm`.

**Result:** ONE TCP handshake per host per 30 min. All subsequent `ssh root@ssh*.vast.ai cmd` runs multiplex over the persistent connection. No more rate-limit refusals.

**Verify with API not SSH:** When unsure if an instance is alive, use the vastai SDK (`v.show_instances()`) — that hits the API not the SSH proxy and shows `gpu_util` / `cpu_util` / `cur_state`. If the API says `running` + `gpu_util > 0`, your training is FINE even if SSH is refused.

**How to apply:** Already added 2026-04-26 in `~/.ssh/config`. Future sessions just work. If config gets reset, regenerate from this memory.
