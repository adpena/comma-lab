---
name: tmux and mosh Available for Long-Running Processes
description: Use tmux for processes that exceed tool timeouts. mosh also available for persistent connections.
type: reference
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
tmux 3.6a is installed at /opt/homebrew/bin/tmux. Use it for ANY process that takes more than 10 minutes (the tool timeout limit).

**Why:** The scorer takes 30+ minutes on CPU. Tool timeouts kill processes at 10 minutes. nohup processes die silently. tmux sessions persist independently.

**How to apply:**
- `tmux new-session -d -s eval 'command here'` — start detached session
- `tmux capture-pane -t eval -p | tail -5` — check progress without attaching
- `tmux kill-session -t eval` — clean up when done
- Use for: scorer eval, long training runs, CRF sweep scoring, any process > 10 min
- mosh is also available for persistent SSH connections to Lightning/remote machines
