# Cathedral probe dispatch-attempted bool hardening - 2026-05-16

Scope: Cathedral autopilot probe-disambiguator read-only consumer.

Finding: `dispatch_attempted` is an authority-bearing field. The top-level
probe payload and each `autopilot_rows[]` entry rejected literal JSON `true`,
but used Python truthiness and therefore accepted malformed non-booleans such as
`0` or `""`.

Change: both reads now route through `_json_bool_field(...)`, matching the
existing score/promotion/exact-ready guards. Focused tests cover malformed
top-level and row-level values (`"false"`, `""`, `0`, `1`).

Status: no dispatch, no score claim. This hardens the read-only planning surface
without changing valid probe rows.
