# PR85 QH0 Record Anatomy

- planning_only: `True`
- score_claim: `False`
- renderer bytes: `61590`
- record count: `111`

## Top Records By Bytes

- `frame1_head.block1.film_proj.weight` `int8_row_scale_unsplit`: `5601` bytes, best probe delta `-68`
- `pose_mlp.2.weight` `fp16_unsplit`: `4609` bytes, best probe delta `-731`
- `shared_trunk.fuse.pw.weight` `fp4_hilo`: `3529` bytes, best probe delta `-132`
- `shared_trunk.down_block.conv1.pw.weight` `fp4_hilo`: `2305` bytes, best probe delta `-38`
- `shared_trunk.down_block.conv2.pw.weight` `fp4_hilo`: `2305` bytes, best probe delta `-23`
- `shared_trunk.down_conv.pw.weight` `fp4_hilo`: `2017` bytes, best probe delta `-17`
- `shared_trunk.up.1.pw.weight` `fp4_hilo`: `2017` bytes, best probe delta `-16`
- `shared_trunk.stem_block.conv1.pw.weight` `fp4_hilo`: `1765` bytes, best probe delta `-42`
- `shared_trunk.stem_block.conv2.pw.weight` `fp4_hilo`: `1765` bytes, best probe delta `-15`
- `shared_trunk.fuse_block.conv1.pw.weight` `fp4_hilo`: `1765` bytes, best probe delta `-28`
- `shared_trunk.fuse_block.conv2.pw.weight` `fp4_hilo`: `1765` bytes, best probe delta `0`
- `frame1_head.block1.conv1.pw.weight` `fp4_hilo`: `1765` bytes, best probe delta `-32`

## Top Recompressible Records

- `pose_mlp.2.weight`: `4609` bytes, best probe delta `-731`
- `pose_mlp.0.weight`: `577` bytes, best probe delta `-316`
- `shared_trunk.fuse.pw.weight`: `3529` bytes, best probe delta `-132`
- `frame1_head.block1.film_proj.weight`: `5601` bytes, best probe delta `-68`
- `shared_trunk.stem_block.conv1.pw.weight`: `1765` bytes, best probe delta `-42`
- `shared_trunk.down_block.conv1.pw.weight`: `2305` bytes, best probe delta `-38`
- `shared_trunk.down_block.conv1.norm.weight`: `129` bytes, best probe delta `-36`
- `frame1_head.block1.conv2.pw.weight`: `1765` bytes, best probe delta `-34`
- `frame1_head.pre.pw.weight`: `1639` bytes, best probe delta `-34`
- `shared_trunk.up.1.norm.weight`: `113` bytes, best probe delta `-33`
- `frame1_head.block1.conv1.pw.weight`: `1765` bytes, best probe delta `-32`
- `frame1_head.block2.conv2.pw.weight`: `1765` bytes, best probe delta `-32`
