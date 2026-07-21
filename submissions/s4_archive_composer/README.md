# S4 archive composer receiver

This is the self-contained Task #578 S4 receiver. It reads one monolithic
`0.bin`, validates every outer/section/PPCS digest and exact length, replays the
counted predictor/event/component grammars, and atomically emits contest-layout
RGB bytes. Runtime dependencies are limited to NumPy and Brotli; LZMA and zlib
come from the Python standard library.

The versioned section registry admits the current LZMA/Brotli/zlib streams and
the `range_static_v1` terminal. The latter is a bit-exact standalone twin of the
repository #557 static arithmetic decoder; a sibling may hot-swap a section to
that codec by supplying the registered section bytes and matching manifest row,
without changing the composer or receiver.

The current artifact is a `[macOS-CPU advisory]` apparatus row. Its deterministic
receiver closure does not imply semantic-cell or Pose-tube admission, and it is
not promotion eligible.
