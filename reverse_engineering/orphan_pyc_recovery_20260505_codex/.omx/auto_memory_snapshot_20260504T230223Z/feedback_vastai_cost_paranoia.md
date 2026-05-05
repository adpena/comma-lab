---
name: Vast.ai Cost Paranoia — Destroy Idle Instances Immediately
description: HARD RULE — never leave Vast.ai instances running without active work. Check and destroy immediately when done.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Vast.ai instances cost money EVERY SECOND they exist, whether or not they're doing useful work. At $0.25-0.30/hr, a forgotten instance wastes $6-7/day.

**Why:** Budget is $24 hard cap. Every wasted dollar is a lost experiment. An idle instance for 1 hour = one fewer TTO step curve experiment.

**How to apply:**
- BEFORE launching any experiment: plan exactly what will run and how long
- AFTER every experiment completes: immediately check if more work should run on that instance
- If no more work: `echo "y" | vastai destroy instance <id>` IMMEDIATELY
- When monitoring: always check `vastai show instances` and flag any idle instances
- Set a mental timer: if an instance has been idle for 5+ minutes, something is wrong
- At the end of every session: verify ALL instances are destroyed
- Track cumulative spend: `total_hours × $/hr` for each instance
- Never assume "I'll use it later" — destroy now, create fresh when needed
- The cosine step curve takes ~7 min. Destroy the instance within 10 min of completion.
