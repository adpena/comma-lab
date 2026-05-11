"""Retired Modal deployment archive namespace.

Do not add executable score-lowering dispatch scripts here. Provider launch
logic belongs in the canonical deploy surfaces under :mod:`tac.deploy` and
must use the shared claim, custody, and provider-contract helpers.

The stale ``train_tac.py`` archive launchers were replaced with fail-closed
stubs on 2026-05-11 after the score path moved to canonical provider bundles
and Modal-specific T1 actuators. Their reusable lessons are preserved in the
corresponding dated ``.omx/research`` ledger, not as runnable legacy code.
"""
