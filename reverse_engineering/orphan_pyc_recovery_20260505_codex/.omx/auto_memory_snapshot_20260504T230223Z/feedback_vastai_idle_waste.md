---
name: Vast.ai instance 35323868 was running idle — caught and destroyed but too late
description: An old Vast.ai instance at $0.302/hr was running idle while we created a new one. Burned unknown hours. ALWAYS check for existing instances before creating new ones.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
On 2026-04-21, we created a new Vast.ai instance (35404589) without checking
for existing running instances. Instance 35323868 was still running at $0.302/hr
from a previous session. Discovered only when checking `vastai show instances`.

**Why:** We didn't run `vastai show instances` before `vastai create instance`.

**How to apply:**
- ALWAYS run `.venv/bin/vastai show instances` BEFORE creating new instances
- ALWAYS destroy instances IMMEDIATELY after downloading results
- Add instance count check to the deployment scripts
- Track cumulative spend and compare to budget cap ($24)
