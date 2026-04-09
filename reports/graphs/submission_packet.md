# submission packet

## notebook surface

- interactive lab notebook: `reports/graphs/lab_notebook.md`
- methodology: `docs/lab_methodology.md`
- glossary: `reports/graphs/glossary.md`

## score target

- Best honest Track B **current_workflow** score: **`1.73`**
- Best honest Track B **current_workflow** bytes: `864,167`
- Best honest Track B **rule_faithful estimate**: `1.7947470454539947` at `966,071` bytes
- Promoted evidence root: `reports/raw/2026-04-09-long1000-h64-authoritative`

## evidence

- scorer-backed candidate beat the prior `1.84` floor
- local smoke gate passed
- authoritative local CPU scorer confirmed the gain
- written promotion review exists
- current_workflow vs rule_faithful separation is explicit

## path summary

- x265 honest floor reached `3.25`
- repaired AV1 path reached `2.20`
- one-axis AV1 tuning reached `2.18` then `2.12`
- encoder-side `sharpness=1` reached `2.08`
- a tiny learned int8 post-filter reached `2.05`
- longer-horizon QAT+EMA training improved that to `1.99`
- the wider h32 long-500 QAT+EMA branch reached `1.95`
- extending the h16 branch to 1000 epochs established `1.92`
- extending the h32 branch to 1000 epochs established `1.85`
- a bounded ensemble of the `1.85` floor and the best Monte Carlo refinement established `1.84`
- scaling the same long-horizon QAT+EMA recipe to `h64` established the current floor at **`1.73`**

## active follow-on

- promoted floor is now `long1000_h64`
- main non-promoted training lane is the bat00 WSL quantization-parity rerun
- main non-promoted proxy lane is the official-path h64 repo-side proxy readout
