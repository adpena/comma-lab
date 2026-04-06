# judges one-pager

## current result

- Best honest Track B **current_workflow** score: **`2.12`**
- Best honest Track B **current_workflow** bytes: `864,486`
- Best honest Track B **rule_faithful estimate**: `2.142` at `897,745` bytes
- Best config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp=0.35 / explicit bt709-tv encode / explicit rgb24(pc) decode`

## latest winning change

- prior floor: `2.18`
- change: explicit `tv/bt709` encode tags + explicit `rgb24(pc)` decode
- estimate before run: lower evaluator mismatch at essentially the same byte budget
- result: **`2.12`**
- reflection: bytes barely moved, SegNet worsened slightly, but PoseNet improved enough to win materially
