# Appended notes — do not rewrite the prose above/elsewhere in this directory

## 2026-08-20 · ddm_oc2 · either repo may be linked in the PR body

Operator directive 2026-08-20, verbatim intent: the consolidation may use **either** repo in the PR
body as appropriate. Recorded here so the PR author does not re-litigate it.

The practical consequence is smaller than it sounds, because there is only one repo:
`origin` for the pact research repo **is** `git@github.com:adpena/comma-lab.git`, and that repo is
**PUBLIC** (`gh repo view adpena/comma-lab` → `"visibility": "PUBLIC"`). The research repo and the
public comma-lab repo are the same object. So "either repo" resolves to: link the comma-lab repo,
optionally deep-linking a path inside it (for example the packet staging directory or the jg5
runtime custody tree) rather than only the repository root.

This does **not** authorize the contest PR. That still gates on the operator's one-line confirm
(#1111). The 2026-08-20 authorization covers pushes to our own repos only.

Receipt: `.omx/research/ddm_oc2_origin_consolidation_20260820/PUSH_READY_COMMA_LAB.md`.
