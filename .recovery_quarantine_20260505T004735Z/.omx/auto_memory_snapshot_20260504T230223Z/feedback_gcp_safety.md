---
name: GCP Safety Rules
description: GCP has NO free credits — billing must stay unlinked. Any usage costs real money.
type: feedback
---

GCP free trial credits are EXHAUSTED. No credits remain.
Billing is UNLINKED. Any re-linking charges real money.

**Why:** User confirmed "No credits to display" on 2026-04-10. Free trial was used on prior projects.

**How to apply:**
- NEVER re-enable billing without explicit user approval and budget discussion
- GCP is configured and ready (quota approved, images selected, deploy scripts work)
- Re-enable command: `gcloud billing projects link personal-mbp-2026 --billing-account=019408-1D0332-BB5DBF`
- If user wants to spend real money on GCP later, discuss cost first
- Budget alert exists at $250 on the billing account
- Project: personal-mbp-2026
