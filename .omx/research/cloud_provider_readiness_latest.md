# Cloud provider readiness

Generated: `2026-05-10T07:55:53Z`

This is a read-only provider inventory. It is not a dispatch, score claim, or promotion artifact.

| provider | status | exact CUDA allowed now | proxy only | blockers | next action |
|---|---|---:|---:|---|---|
| modal | ready | yes | no | - | Harvest active Modal A1 calls before refiring. |
| kaggle | ready_proxy | no | yes | - | Use Kaggle for Optuna/CMA-ES/proxy curves only. |
| aws | blocked_auth | no | no | aws_session_expired | Run `aws login`, then re-run this readiness tool. |
| gcp | blocked_billing | no | no | gcp_billing_not_enabled_or_not_readable, gpu_quota_not_checked | Enable billing on the selected GCP project or switch to a billed project. |
| azure | blocked_auth | no | no | azure_not_logged_in | Run `az login`, then re-run this readiness tool. |
