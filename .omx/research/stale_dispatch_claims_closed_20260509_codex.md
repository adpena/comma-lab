# Stale dispatch claims closed (2026-05-09)

## Summary

Closed three stale preauthorization placeholder rows with terminal
`stale_superseded_no_active_dispatch` statuses. No remote/GPU/eval dispatch was
launched.

## Terminal rows appended

Command surface inspected first:

```zsh
tools/claim_lane_dispatch.py claim --help
```

Rows closed:

| lane_id | instance/job_id | previous class | terminal status |
|---|---|---|---|
| `apogee_int6_contest_cuda_anchor` | `PRESTAGE:apogee-int6-cuda-anchor-20260508-PLACEHOLDER` | stale preauthorization placeholder | `stale_superseded_no_active_dispatch` |
| `pr101_admm_step6_no_dead_k` | `PRESTAGE:admm-no-dead-k-20260508-PLACEHOLDER` | stale preauthorization placeholder | `stale_superseded_no_active_dispatch` |
| `pr107_apogee_cpu_auth_eval_linux_x86_64` | `PRESTAGE:pr107-cpu-eval-lightning-20260508-PLACEHOLDER` | stale preauthorization placeholder | `stale_superseded_no_active_dispatch` |

Post-closure summary:

```text
active_count = 1
stale_nonterminal_count = 0
active lane = lane_avvideodataset_cuda_path_mechanism_discriminator
active job = discriminator-sweep-20260509T110211Z
```

## Classification

These rows were not score results and not method negatives. They were stale
coordination state that could misroute future dispatch decisions.

## Reactivation criteria

Any of the closed lanes may be reopened only with:

- fresh `tools/claim_lane_dispatch.py claim ...` row;
- current archive/runtime custody;
- exact blocker review for the prior failure class;
- operator dispatch decision where required;
- no active same-lane conflict in the 24h claim window.
