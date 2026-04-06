# best write-up outline

## 1. premise

- Track A is the only intentionally non-rule-faithful lane.
- Track B is the honest scorer-backed lane.
- The writeup should foreground rigor, not just cleverness.

## 2. key result

- Best honest Track B result: **`2.19`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / bicubic / unsharp`

## 3. main thesis

- strong standard-codec AV1 is the main frontier
- evaluator-boundary bugs can completely mask achievements
- disciplined one-axis experiments plus explicit reflection are the right operating model

## 4. strongest visual story beats

1. honest baseline at `4.06`
2. x265 ladder down to `3.25`
3. AV1 bug at `97.45`
4. AV1 repair to `2.20`
5. one-step AV1 neighborhood improvement to **`2.19`**
6. promotion review and bug-audit standard becoming explicit repo policy
