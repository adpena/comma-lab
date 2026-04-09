# best write-up outline

## 1. premise

- Track A is the only intentionally non-rule-faithful lane.
- Track B is the honest scorer-backed lane.
- The writeup should foreground rigor, not just cleverness.

## 2. key result

- Best honest Track B result: **`1.95`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / sharpness=1 / long500 QAT+EMA learned int8 post-filter (h=32)`

## 3. main thesis

- strong standard-codec AV1 plus tiny task-aware decode correction is the main frontier
- evaluator-boundary bugs can completely mask achievements
- disciplined measurement, long-horizon training, and explicit state-keeping are the right operating model

## 4. strongest visual story beats

1. honest baseline at `4.06`
2. x265 ladder down to `3.25`
3. AV1 bug at `97.45`
4. AV1 repair to `2.20`
5. honest AV1 tuning down to `2.08`
6. tiny learned post-filter to `2.05`
7. long-horizon QAT+EMA h16 to `1.99`
8. long-horizon QAT+EMA h32 to **`1.95`**
