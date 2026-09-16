# Rate lever found by the second fresh reader (2026-09-16): 16 unread bytes in the carrier section, and one unused header flag bit

MEASURED by code reading (reader, then MAIN to verify at charter time): `decode_carrier` (submissions/mrs6/inflate.py ~1941–1966)
reads carrier metadata at offsets 0–6, 6–54, 54–102, 102, 103–114, 114–123, 139, 140–142; bytes 123–138 (16 B) are never read;
header flag bit 32 is set but unread. Both are inside the 179,186 B charged member. Removing them is a −16 B (≈ 0.64 bar,
−1.07e-5 S) archive change with identical decode — a NEW candidate (new archive bytes ⇒ normal seal path, CUDA fire, packet).
Also check the sealed tree's own receiver for the same dead bytes (they came from the sealed format). Owner: the frontier
line (pd-family) — fold into the next pointer move's byte close rather than a standalone fire.
verdict_scope: n/a (a lever, not a negative).
<!-- # FORMALIZATION_PENDING: reader finding; measured at the next byte close -->
