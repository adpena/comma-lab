# G77 — additive G49 V15 role-aware selected-preimage factor V2

Date: 2026-07-27  
Lane: `lane_g77_g49_v15_roleaware_selected_preimage_factor_v2_20260727`  
Status: local research receiver closure; no public inflate, candidate, score, Pose, or pointer claim

## Result

G77 lands the smallest real additive path from a counted role-aware analytic
operand to the current CarrierCompose V15 semantic program:

```text
exact fresh V15 semantic P
  + one population-global counted TSPPV2/G74RA1 factor
  -> all atoms aggregated before one G74 native render_camera_pairs call
  -> role-specific donor-tap support replacement
  -> camera pairs plus strict execution receipt
```

`TSPPV1` and the legacy G17 P/G/A/E envelope remain byte-for-byte frozen.
G77 neither reinterprets the role-stripped G49 factor nor reuses an old selected
payload. The bounded fixture uses two newly serialized role-aware atoms solely
to prove execution mechanics; it does not claim a real n600 factor population
or score.

## Triality

DSL:

```text
TSPPV2(P_id, target_id, runtime_id, window, G74RA1(selector, atoms))
```

DAG:

```text
flat extracted payloads -> exact P -> strict TSPPV2 -> one G74 native render
  -> selected Y1 pair state -> future conditional Y0|Y1 -> public output
```

Equations:

```text
A = aggregate_role_aware_atoms(G74RA1)
Y1_i = D_G74(P, A, i)
D_G74(P, A, i) == D_G74(P, A, i)  byte-for-byte on repeated decode
B* = min_{m in {STORE,DEFLATE}^6} bytes(zip_m(P_members, TSPPV2))
```

## ABI arbitration

Three shapes were considered.

1. Widen TSPPV1's closed factor enum. Rejected. V1's analytic factor strips
   `BoundaryShearletAtomV1.role` and post-paints scorer-grid RGB. Calling those
   bytes V15 role-aware native prepaint would change their meaning.
2. Widen G17 V1 / `G17ActuatorIRV1`. Rejected. The G17 actuator IR is sealed to
   ep725 `TaskspacePredictorStateV2 + NoTransportV2`; it does not consume a
   CarrierCompose V15 P. G17 also intentionally rejects a complete ZIP in its
   monolithic P section.
3. Add TSPPV2 plus a flat top-level product/demux. Chosen. The five exact P
   member payloads occur once, followed by one counted TSPPV2 member. The
   decoder reconstructs exact logical P and then opens G74.

This is a G77 one-actuator research product, not the terminal public ABI.
The intended composition is ordered `Y1` semantic state followed by a future
conditional `Y0|Y1` enhancement. G77 therefore exposes its current typed
actuator slot and explicitly leaves the final ordered multi-actuator state
machine open.

## TSPPV2 counted factor

Wire magic: `TSPPV2\0\0`  
Schema: `tac.taskspace_selected_preimage_program.v2`

The packet admits exactly one factor:

- role: `ANALYTIC_ROLE_AWARE_PREPAINT`
- mode: `V15_ROLE_AWARE_PREPAINT_G74RA1`
- body: one exact canonical `G74RA1` operand
- scope: one global source pair window
- selector: the single selector already carried by G74RA1
- atoms: one canonical role-aware bank, sorted by the donor wire contract
- execution: the whole bank is passed once to G74; sequential overlays and
  legacy/new mixing are unrepresentable

The counted header binds exact semantic-P length/SHA, target custody identity,
source window, operand length/SHA, selector, source receipt, decoder source
hashes, G74 contract/replacement policy, and Torch runtime. Strict parsing
rejects duplicate/noncanonical JSON, unknown fields/enums, noncontiguous body,
EOF drift, pair-window drift, operand drift, and parse/re-emit drift.

For the retained two-atom mechanics fixture:

| byte home | bytes |
|---|---:|
| counted TSPPV2 framing/custody manifest | 3,541 |
| counted G74RA1 operand | 52 |
| total packet | 3,593 |

This is a custody-rich research IR, not a rate-optimal final wire. Most header
fields can move to the durable external receipt once a compact typed product
wire binds the operand. `COMPACT_BINARY_TSPPV2_RECEIVER_PACKET_OWED` is
therefore load-bearing; the current 3,541-byte framing cannot be called optimal.

## Flat product and exact compression race

Physical top-level member order:

```text
manifest.json
predictor.zip
predict/movable_polygon_worldsheet.g1s
render/receiver_realization.ddrp
render/scorer_solved_templates.ddst
taskspace/selected_preimage_program.tsppv2
```

`predictor.zip` remains one opaque exact semantic member. Recursive flattening
is deliberately outside the contract because the measured G79 falsifier found
it larger.

The builder exhaustively evaluates all `2^6 = 64` STORE/DEFLATE profiles at
DEFLATE level 9, compares exact whole-product bytes, and uses lexical method
bits as the deterministic tie-break. On zlib runtime 1.2.12, the winner is:

```text
complete bits: 110011
semantic bits: 11001
TSPPV2 bit:    1
```

`1` means DEFLATE. The selected six-member product is 131,906 bytes,
SHA-256 `6e67c3bf6b2908dc31e05fad3bbb06e4ccde6598859f79542bc4cef1a32be8c7`.
The five semantic members alone at `11001` are 129,800 bytes. Relative to an
all-STORE semantic product plus the same DEFLATEd TSPPV2, the complete race
saves 4,141 bytes. Relative to original P alone, the six-member research
product is 2,035 bytes smaller despite carrying TSPPV2.

Those byte facts are not a score claim: the packet carries only a bounded
mechanics operand and no full n600 actuator population.

## Public-unzip boundary proof

The evaluator extracts `archive.zip` before invoking the decoder, so product
`ZipInfo` metadata cannot be receiver authority. G77 has a separate typed
extracted-directory entrypoint:

```text
/usr/bin/unzip archive.zip -d extracted/
  -> exact closed file-set/symlink/size checks
  -> read only extracted member payload bytes
  -> rebuild canonical all-STORE P from fixed generic constants/order
  -> require P length/SHA from TSPPV2
  -> retain and strictly parse counted TSPPV2
  -> open G74 and execute the counted operand
```

The test reconstructs byte-identical P:

```text
bytes   133941
sha256  759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df
```

It then performs the real G74 path for pair 0. Both selected frames match the
native mutated numerators and Torch bilinear realization; the realization
profile is consumed; legacy `render_pairs` is not used; inner and outer double
decode pass; scorer and Pose are not invoked.

This closes same-host public-`unzip` demux mechanics, not public submission
closure. Canonical P rebuilding still uses Python `zipfile`; cross-host
byte-identity or a manual fixed ZIP32 writer remains owed.

## Compatibility receipts

The focused test re-derives the legacy fixture bytes:

| object | bytes | SHA-256 |
|---|---:|---|
| frozen TSPPV1 | 3,616 | `8a898992d7bbd84a544f5a9918244c9a0155b6f4bcbd717072dcf52801830e68` |
| frozen G17 G | 142 | `b1ac2bbacefbc049be31a0d9f496246f3201d4b1fa184fd44c69836fc536a6d1` |
| frozen G17 A | 3,758 | `44435df8e5a2c6385075b76a1640acc21bffa267b32f7bda2241414dee943b2a` |
| frozen G17 E | 151 | `650003635e004a6336bdf541a59387fe98f0587ba01342ab293d66ad589cf08d` |

No legacy source file was edited. TSPPV2 rejects legacy/new mixtures, and G17's
closed enum/container behavior remains unchanged.

## Verification

```text
pytest focused G77/TSPPV2: 8 passed
pytest adjacent G49/G17/G74/G76: 27 passed
ruff check: passed
py_compile: passed
```

The native render test is bounded pair-level receiver evidence. It is not
n600 evidence and does not move the exact pointer.

## Open blockers

1. `PUBLIC_INFLATE_FLAT_V15_TSPPV2_RUNTIME_INTEGRATION_OWED` — no shipped
   `inflate.py`/`inflate.sh`, streamed 600-pair output, two clean inflations, or
   recursive public runtime closure.
2. `COMPACT_BINARY_TSPPV2_RECEIVER_PACKET_OWED` — the custody-rich counted
   framing is materially nonminimal.
3. `CROSS_HOST_TORCH_FLOAT32_DETERMINISM_OR_FIXED_CAMERA_BYTES_OWED` — current
   native Torch and Python ZIP reconstruction are not cross-host authority.
4. `FINAL_ORDERED_Y1_THEN_Y0_MULTI_ACTUATOR_DEMUX_ABI_OWED` — G77 does not yet
   define the conditional `Y0|Y1` successor/state transition.
5. Fresh n600 role-aware factor materialization, whole-score admission,
   Pose-safe composition, exact public CPU/CUDA evaluation, and pointer
   promotion remain owed.

## Pointer-delta honesty

G77 closes a real local counted-A-to-V15 receiver seam and a same-host
public-unzip demux seam. It also lands an exact whole-product compression race.
It does not produce a score-bearing archive. The canonical exact pointer is
unchanged.
