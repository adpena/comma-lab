# QS2 R1 fire-request tombstone

`SEALED_REQUEST.json` (SHA-256
`1975c3b4e400895d547dc6e58ae6c71a2d15ea75a6ec7e23f8cb6ef4af369e68`) is
superseded and MUST NOT be fired. It used non-placeholder `local_pose_delta`
and `pose_unmeasured` values that the unchanged RE1T worker rejects.

The only live request is `SEALED_REQUEST_r2.json`; follow the top-level
`SEALED_FIRE_ORDER.json` and verify its recorded request SHA before dispatch.
