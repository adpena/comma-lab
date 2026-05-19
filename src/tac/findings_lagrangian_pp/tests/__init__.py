# SPDX-License-Identifier: MIT
"""Smoke tests for `tac.findings_lagrangian_pp` (TRACK B; NumPyro hierarchical posterior).

Per Carmack ULTRA-MVP discipline (slot 20-second-supplemental Q9 AMEND):
TRACK B smoke is minimal — just verify the package imports cleanly + the
optional NumPyro dependency contract is honored. Per the operator-frontier-
override 2026-05-19 verbatim *"we shoud pursue PP in parallel"*: TRACK B is
operator-mandated PARALLEL TRACK to TRACK A, but the heavy NumPyro tests
(MCMC convergence / hierarchical model validation) are deferred to Phase
1.B since NumPyro is an opt-in dependency requiring `pip install` operator
decision per CLAUDE.md "Deployment version checklist" non-negotiable.
"""
