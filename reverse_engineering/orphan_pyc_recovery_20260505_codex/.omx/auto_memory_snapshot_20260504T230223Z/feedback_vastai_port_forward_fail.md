---
name: Vast.ai SSH "remote port forwarding failed" — instance is dead, destroy + relaunch
description: When Vast.ai's SSH proxy port collides with another tenant, you get "Error: remote port forwarding failed for listen port NNNNN" in instance logs and SSH connections close immediately. The instance shows status=running but is unreachable. Detect via vastai logs <id> and auto-destroy.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified 2026-04-27 on instance 35676150 (Oregon machine_id=18857).**

Symptoms:
- `vastai show instance` reports `actual_status=running, cur_state=running` — looks healthy.
- `ssh root@ssh4.vast.ai -p 36150` returns `Connection closed by 3.80.254.151 port 36150` immediately. No banner, no auth attempt.
- `vastai logs 35676150` shows repeated:
  ```
  Error: remote port forwarding failed for listen port 36150
  Mon Apr 27 09:40:20 UTC 2026
  Warning: Permanently added 'ssh4.vast.ai' (ED25519) to the list of known hosts.
  Error: remote port forwarding failed for listen port 36150
  ```
  every 2 seconds.

Root cause: Vast.ai's SSH proxy (the `ssh4.vast.ai` host) cannot bind port 36150 because another tenant or stale forward already has it. The container is up but unreachable from outside. Vast.ai's port pool collision — random luck which port you get assigned.

**How to apply (preflight protocol for any Vast.ai instance):**

After `vastai create instance`, BEFORE running setup:
1. Wait for `actual_status=running` (existing).
2. Wait 30s for sshd to come up.
3. **Run `vastai logs <id> | tail -20` and grep for `"remote port forwarding failed"` or `"Connection closed by"`.** If found:
   - The instance is DEAD. Destroy immediately. Re-launch on a different offer.
   - Do NOT spend setup time on it.
4. Try one `ssh ... echo OK` with a 10s timeout. If it fails AND logs show port-forward errors, destroy.
5. If logs are clean and SSH responds, proceed with setup.

This protocol turns a 30-min wasted setup ($0.30) into a 60-second probe ($0.005).

**Cost of this trap:** $0.05 + 6 min on instance 35676150 today. Could have been catastrophic if I'd kicked off the experiment script blind.
