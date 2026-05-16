# Rudin-Daubechies panel-axis hardening - 2026-05-16

Scope: Cathedral autopilot Rudin-Daubechies SLIM/Rashomon reranking.

Finding: pre-dispatch reranking built every `ProxyPanel` with
`panel_axis="macos_cpu_advisory"`, even when the caller was ranking
contest-target dispatch candidates. The update path already required an
explicit axis, so rerank and update had asymmetric evidence-axis discipline.

Change: `rerank_candidates_via_rudin_daubechies(...)` now accepts an explicit
`panel_axis` and defaults the dispatch-facing surface to `contest_cuda`. The
axis is validated against `contest_cuda`, `contest_cpu`, and
`macos_cpu_advisory`, and the operator explanation includes the chosen axis.

Status: no score claim. This prevents advisory predictions from silently being
read as contest-axis evidence and keeps CPU/CUDA/advisory semantics visible.
