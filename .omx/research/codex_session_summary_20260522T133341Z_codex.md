# Codex Session Summary

Timestamp: 2026-05-22T13:33:41Z

## Landed

- Built the decoder-q selective-window bridge module and CLI.
- Generated a real top-32 bridge plan from the strict MLX selector and existing
  `d1f1e56e042692f2` materialized candidate.
- Preserved the bridge artifact signal in
  `codex_findings_decoder_q_selective_window_bridge_20260522T133341Z_codex.md`
  with exact paths and SHA-256 hashes instead of forcing ignored experiment
  outputs into git.

## Current State

- MLX remains `[macOS-MLX research-signal]` only.
- The strict top-32 windows are work-order inputs, not score/rank/promotion or
  dispatch authority.
- The bridge is intentionally blocked with
  `blocked_missing_decoder_q_selective_runtime_grammar`; the existing PR101
  FES/FEC selector packet grammar cannot encode decoder-q tensor mutation arms
  per selected window.
- Existing dirty waterbucket WIP remains quarantined and unstaged. It is useful
  later for dedupe/false-authority hygiene but is not the selective bridge.
- Existing dirty HFV sparse-sidecar WIP remains untouched.

## Verification

```bash
ruff check src/tac/optimization/decoder_q_selective_window_bridge.py \
  tools/build_decoder_q_selective_window_bridge_plan.py \
  src/tac/tests/test_decoder_q_selective_window_bridge.py
```

Result: `All checks passed!`

```bash
.venv/bin/python -m pytest src/tac/tests/test_decoder_q_selective_window_bridge.py -q
```

Result: `4 passed`.

## Recommended Next Step

Implement the byte-closed selective decoder-q runtime grammar, then generate
singleton and small-run archives from the bridge work units and gate them with
official inflate/raw-output controls before any claimed exact CPU/CUDA auth
eval dispatch.
