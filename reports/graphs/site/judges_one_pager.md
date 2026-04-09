# judges one-pager

## headline

- Best honest Track B **current_workflow** score: **`1.73`**
- Best honest Track B **current_workflow** bytes: `864,167`
- Best honest Track B **rule_faithful estimate**: `1.795` at `966,071` bytes
- Best config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / long1000 QAT+EMA learned int8 post-filter (alpha=20, h=64)`

## why it won

- prior floor: `1.84`
- change: keep the long-horizon QAT+EMA recipe, scale width to `h64`, and let the saved-best int8 checkpoint carry the deployment path
- result: **`1.73`** at `864,167` bytes, with a local **rule_faithful** estimate of `1.795` at `966,071` bytes
- reflection: the next real gain came from width scaling, not more ensemble tuning

## proof points

- local smoke passed before the promoted scorer run
- local CPU scorer run established the promoted floor
- written promotion review exists for the promoted run
- at the promoted `1.73` operating point, the score is still much more sensitive to SegNet than PoseNet, though the ratio has narrowed to about **11.5x**
