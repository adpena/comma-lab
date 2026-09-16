# Artifact custody and retention

The public source copy and all retained proof payloads live under
`/Volumes/APDataStore/pact/ddm_mrs4/`. The six-file delivery copy is
`submissions/mrs4/`. Native libraries exist only in external proof runtimes;
they are reproducible build products and are excluded from the public source
and delivery tar. All archives and every sampled candidate token, raw and
differential probability payload are retained with byte counts and SHA-256.

The full cold raw was certified against the move-53 full digest and all 600
pair digests before the existing automatic scratch cleanup removed it.
`public_native/IDENTITY.json` retains its original path, bytes, digest,
source/archive/runtime bindings, command, environment and false-authority flags.
The unchanged counted archive remains present. The complete raw is reproducible
from that source and archive; this is certified duplicate scratch cleanup,
not a scalar-only payload measurement.

The bare venv uses the charter-required APFS path
`.omx/tmp/ddm_mrs4_bare_venv`, never the ExFAT volume. Its installed-file
inventory, dependency wheels and rebuild commands are retained before cleanup.
Launcher completion receipts are copied from the prescribed local done store
to the SSD delivery store. No operator evidence artifact is placed in a global
temporary directory.

The final retention manifest states logical bytes and allocated blocks,
includes AppleDouble metadata in the byte census, and records any metadata
self-reference exclusions explicitly. The public six-file budget excludes
only filesystem metadata, not source or executable payload.
