# Packet-compiler context-recode lazy export - 2026-05-16

Scope: packet-compiler package initializer and HNeRV low-level packer imports.

Finding: importing `tac.hnerv_lowlevel_packer` in a fresh interpreter could
cycle through `tac.packet_compiler.__init__`: package init imported
`pr106_context_recode`, which imports `hnerv_lowlevel_packer` for source-archive
parsing while that module is still initializing.

Change: package-level PR106 context-recode helpers remain available via
`from tac.packet_compiler import ...`, but are now exposed lazily through
`__getattr__`. The sidecar packet primitives can import without pulling the
context-recode/HNeRV dependency edge at package initialization time.

Verification target: a subprocess import test imports `tac.hnerv_lowlevel_packer`
first, then resolves `emit_pr106_context_source_payload` through the package
surface.
