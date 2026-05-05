---
name: Modal Billing — Turn Off Auto-Recharge
description: Modal auto-recharges by default. User wants to spend existing credits only, no new charges without approval.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Modal auto-recharges credits by default. User wants this OFF.

**Action needed:** Go to https://modal.com/settings/billing and disable auto-recharge. 
The Modal CLI does not expose billing settings — must be done in the web dashboard.

**Current state:** $30 free credits consumed. ~$20 charged. Use remaining credits for T4 jobs only (cheapest GPU). No new charges without explicit approval.

**How to apply:**
- Use T4 on Modal ($0.59/hr), not A10G ($1.10/hr) or H100 ($3.50/hr)
- Estimate cost BEFORE deploying: `hours * $/hr`
- Log all Modal jobs with cost estimates in experiment records
