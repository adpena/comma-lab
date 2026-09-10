# Recovered SSD source blobs

These files are inert, content-addressed custody copies of historical authored
source discovered by `tools/audit_ssd_authored_signal.py`. Their original
extension, path, owner, exact SHA-1/SHA-256, review results, and disposition are
recorded append-only in
`.omx/research/ddm_cs1_ssd_code_certify_20260909.jsonl`.

Git object insertion was denied in the originating sandbox. Nine bounded
serializer fallback bundles and format patches preserve these 223 exact blobs;
the top-level `../landing_manifest.json` records their hashes and apply order.
The blobs are not counted as Git-reachable until MAIN lands those patches.

The `.blob` suffix is deliberate: these are preserved source evidence, not live
entrypoints. Do not execute one from this directory. A source owner may promote
a reviewed blob to an active repository path in a later, separately tested
landing.
