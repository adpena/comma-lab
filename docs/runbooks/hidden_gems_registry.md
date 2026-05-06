# Hidden-Gem Registry

The hidden-gem registry is a static planning surface for partially built
techniques that deserve follow-up. It does not launch jobs, read provider
state, or make score claims.

Use it when choosing the next local patch:

```bash
python tools/list_hidden_gems.py --format markdown
python tools/list_hidden_gems.py --format json
python tools/list_hidden_gems.py --format markdown --status ready_for_patch
```

Each entry records:

- category
- status
- evidence paths
- integration targets
- next patch
- contest-compliance notes

Registry entries must stay deterministic and repo-relative. Do not put
`.omx/state`, provider logs, raw experiment-result trees, secrets, live job
IDs, or concrete score claims in this surface. If an entry becomes a candidate
archive, move the evidence into the normal archive custody and exact-eval
workflow before dispatch.
