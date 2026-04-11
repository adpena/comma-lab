# 2026-04-05 codec shortlist summary

## BAT00 shortlist

Top ranked candidates from the BAT00 codec-only surrogate:

[
  {
    "predicted_score": 1.6070749759674072,
    "run_id": "cand-432x324-crf23-g48-b3-r4"
  },
  {
    "predicted_score": 1.8104490041732788,
    "run_id": "cand-432x324-crf23-g64-b4-r4"
  },
  {
    "predicted_score": 1.8169111013412476,
    "run_id": "cand-424x318-crf23-g48-b4-r4"
  },
  {
    "predicted_score": 1.8240996599197388,
    "run_id": "cand-432x324-crf23-g48-b4-r5"
  },
  {
    "predicted_score": 1.8952363729476929,
    "run_id": "cand-416x312-crf23-g48-b4-r4"
  }
]

## measured local CPU results

1. `432x324 / medium / 23 / keyint48 / bframes3 / ref4`
   - score: **`3.43`**
   - bytes: `1898751`
   - verdict: reject

2. `432x324 / medium / 23 / keyint64 / bframes4 / ref4`
   - score: **`3.38`**
   - bytes: `1753611`
   - verdict: reject

## interpretation

- The BAT00 codec-only surrogate was useful enough to prioritize the next two local runs.
- Both top candidates still lost to the promoted `3.33` floor.
- `keyint64` at `432x324` is closer than the `bframes3` follow-up, but still not a promotion.


3. `424x318 / medium / 23 / keyint48 / bframes4 / ref4`
   - score: **`3.25`**
   - bytes: `1669984`
   - verdict: **promote**

## updated interpretation

- The BAT00 codec-only surrogate did not rank the winner first, but it still helped keep the search focused in a productive region.
- The third shortlisted candidate produced a real new floor at `3.25`.
