# PVSA1 public-runtime staging

This directory exercises the contest three-argument `inflate.sh` ABI against
the exact G80 PVSA1 member and the bounded G85 raw writer.

It is deliberately **repository-bound and research-only**. The current G80
receiver recursively imports the research direct-description stack, including
`pydantic`, while the frozen upstream evaluator lock does not install
`pydantic`. Therefore this staging tree is not a self-contained submission
runtime and must not support a score, candidate, or promotion claim.

The next required implementation is a decoder-only tree-shake that retains the
exact PVSA parse, semantic-P inverse, G74 transition, and uint8 camera renderer
without repository or non-upstream package imports.
