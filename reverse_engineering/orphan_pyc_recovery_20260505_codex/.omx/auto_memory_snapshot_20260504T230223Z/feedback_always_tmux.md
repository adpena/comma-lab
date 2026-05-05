---
name: ALWAYS use tmux for remote processes
description: BINDING — never use nohup + & for long-running remote processes. Use tmux. nohup dies with shell sessions and SSH disconnects.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
ALWAYS use `tmux` (or `screen`) for ANY long-running process on a remote machine. NEVER `nohup ... &` alone.

**Why:** On 2026-04-25, deployed 3 A100 post-training watchers using `nohup bash -c '...' &`. All 3 watchers died (probably when the launching SSH session ended, or due to SIGHUP). The training process kept running (managed by separate nohup) but the WATCHER that was supposed to fire QAT/TTO/eval after training exited was dead. Result: when training finishes, NOTHING auto-triggers the post-pipeline. Wasted hours of A100 time, broke the user's expectation of "auto-chained pipeline."

**The user explicitly demanded:** "always use tmux" and "i told you never ad hoc."

**How to apply:**
- For ANY long-running remote command: `tmux new -d -s session_name 'command'`
- For pipeline watchers: `tmux new -d -s pipeline_watch 'while pgrep -f train; do sleep 60; done; bash /path/run_pipeline.sh'`
- Verify with `tmux ls` before disconnecting
- For the WATCHER specifically, even better: use `systemd --user` units for self-healing
- nohup + & is FORBIDDEN for anything that needs to survive past the SSH session
