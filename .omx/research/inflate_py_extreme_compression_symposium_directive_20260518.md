# Inflate.py Extreme Compression Symposium — Operator-Routed Directive
# Date: 2026-05-18
# Originating session: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
# Per CLAUDE.md "Subagent coherence-by-default" — this directive supersedes the original subagent prompts.

## Audience

Both in-flight symposium subagents on the inflate.py extreme compression topic:
- Primary: T3 grand council symposium (lane `lane_inflate_py_extreme_compression_symposium_20260518`)
- Relay: cross-language + AOT + molt expansion (parallel sister scope)

If you are reading this during your pre-flight `.omx/research/*_directive_*` scan, INCORPORATE the additions below into your symposium deliverable BEFORE landing your memo.

## Additional operator context (relayed 2026-05-18 mid-flight)

### Context message 1 (off-the-shelf vs roll-our-own)

Operator verbatim:
> *"we should search for such tools off the shelf onine but also consider rolling our own or customizing and building our own wrappers or whatever"*

Interpretation:
1. **OFF-THE-SHELF SURVEY remains mandatory** (continue WebSearch/WebFetch breadth per the original prompt + molt expansion relay)
2. **ROLL-OUR-OWN is FIRST-CLASS** (not a fallback). Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + the 2026-05-15 retrospective on canonical-helper suppression: if no off-the-shelf tool serves THIS specific contest-portable problem optimally, the symposium MUST recommend rolling our own canonical helper rather than force-fitting an off-the-shelf solution
3. **WRAPPERS + CUSTOMIZATION ARE FIRST-CLASS** — adapting existing tools (e.g., wrapping `python-minifier` with substrate-specific passes; forking `Nuitka` with contest-portable runtime; customizing molt's tree-shaking for inflate.py-specific dead-code patterns) is the canonical-helper pattern per Catalog #265 "symposium implementations canonical contract"
4. **"WHATEVER"** signals operator-trust + breadth — don't gate the symposium on operator approval per-technique; surface ALL viable approaches in the deliverable + let the operator route via the Op-Routables section

The symposium's TOP-N technique ranking MUST therefore include BOTH off-the-shelf AND roll-our-own options as PEERS, not relegate roll-our-own to a fallback tier.

### Context message 2 (super powerful + magic Pythonic tooling)

Operator verbatim:
> *"whatever pythonic tooling unlocks super poerful and magic"*

Interpretation:
1. **AGGRESSIVE SCOPE EXPANSION APPROVED** — beyond minification/AOT/compilation, the symposium should survey ALL Pythonic meta-tooling that could shrink inflate.py: AST rewriters / metaclass-based codegen / bytecode patchers / introspection-based compaction / decorator-based macros (`hy` / `mython` / `macropy`) / template-based codegen / type-directed minification / runtime-vs-compile-time partial evaluation / `functools.lru_cache` patterns that obviate explicit lookup tables
2. **"MAGIC" SIGNALS NON-OBVIOUS / CREATIVE / META-LEVEL** approaches preferred. Examples to include:
   - **Self-modifying code patterns** (Python `exec` + AST mutation)
   - **`__init_subclass__` + descriptor-based codegen** that eliminates boilerplate
   - **Metaclass-driven serialization** (e.g., `attrs` / `pydantic` v2 codegen patterns)
   - **Pattern matching (PEP 634) + structural unpacking** as boilerplate-killer
   - **Walrus operator (`:=`) compaction patterns** (saves LOC in many contexts)
   - **Hidden Python features that ship with stdlib** (`encodings.unicode_escape`, `binascii.b2a_base64`, `zlib.compressobj` with custom windows, `pickle` protocol 5 with out-of-band buffers, `array.array` for compact homogeneous tables, `struct.pack_into` for shader-style constant packing, `marshal` for raw bytecode serialization)
   - **Python compile-time tricks** (`__pycache__` redirection, `sys.path` hooking via `importlib.abc.MetaPathFinder`, frozen modules in Python 3.11+, `--build-id=none` ABI optimization)
   - **Polyglot tricks specific to the contest contract** (does `inflate.sh` accept `python -c "code"` inline? does it accept `exec` of a base85-encoded payload?)
3. **OPERATOR-FACING REVIEWABILITY (HNeRV parity L4 30-sec review) IS NOT NEGOTIABLE** — magic techniques that produce unreviewable code violate the discipline. The symposium MUST classify each "magic" technique on the dimension `(compression-ratio × portability) / reviewability-cost`. Self-extracting LZMA bootloader is the canonical sweet spot (5-LOC visible bootloader + arbitrary opaque payload; reviewable because the bootloader is 5 lines + the payload is just `inflate.py.lzma`).

### Specific Pythonic techniques to add to the survey

Beyond the molt + AOT-compilation expansion already relayed:

#### "Magic" stdlib patterns
- `encodings.unicode_escape` for compact bytestring embedding (vs base85)
- `marshal.dumps` of code objects (raw bytecode without .pyc overhead)
- `pickle` protocol 5 + out-of-band buffers for numpy-table embedding
- `zlib.compressobj(level=9, wbits=-15)` for raw deflate without zlib headers (smaller than `lzma` for some payloads)
- `dis.get_instructions` + bytecode rewriting (active 2024-2026; see `bytecode` library by serhiy-storchaka)
- `__class_getitem__` + `typing.Generic` patterns that codegen at class-definition time
- `__slots__` aggressive use to compact class instances

#### Compile-time / metaclass codegen (Python's "macro" equivalents)
- `hy` (Lisp dialect that compiles to Python AST; native macros)
- `macropy3` (syntactic macros for Python via import-hook)
- `mython` (mython.org; abandoned but technique persists in newer libs)
- `pyrsistent` for immutable data structures with codegen
- `attrs` + `cattrs` (declarative class generation; eliminates `__init__`/`__repr__`/`__eq__` boilerplate)
- `dataclasses` with custom metaclass for substrate-specific codegen
- `pydantic` v2 with rust-backed validation (could compress validation logic)

#### Bytecode-level tools
- `bytecode` library (CPython's official bytecode manipulation library; canonical for 2024-2026)
- `xdis` (cross-version disassembler)
- `decompyle3` (bytecode → source, useful for reverse-engineering canonical patterns)
- `pyc-zer0` (.pyc obfuscation tooling)

#### Modern AOT / JIT (2024-2026 cutting edge)
- **Codon** (https://exaloop.io/codon) — LLVM-backed Python compiler; 10-100× speedup; AOT static binaries; **HIGH-PRIORITY for symposium evaluation as alternative to molt**
- **Mojo** (Modular's Python superset with MLIR/LLVM) — AOT compilation + extreme performance + ships as compiled `.mojopkg`
- **mypyc** (Mypy's official compiler; type-annotated Python → C extension; production-tested at Dropbox)
- **Cinder** (Meta's CPython fork with strict modules + JIT)
- **Pyston** (https://github.com/pyston/pyston; JIT-compiled CPython fork)
- **Pyjion** (.NET CoreCLR JIT for CPython)
- **Nuitka** (Python → C → standalone single-file binary; production-tested)
- **PyOxidizer** (Rust-based Python distribution bundler)

#### Rust + cross-language Pythonic magic
- **PyO3** (Rust ↔ Python; native extensions in Rust)
- **maturin** (PyO3 build tool; produces `.whl` packages with embedded Rust)
- **rustpython** (Python implementation in Rust; experimental but interesting for WASM target)
- **rye** (Astral.sh's Python project manager; clean dependency closure for distribution)
- **uv** (Astral.sh's Rust-based pip-replacement; already used in the contest's bootstrap)

#### Cosmopolitan / ape format (extreme magic)
- `cosmopolitan-python` (Python binary that runs on Linux/macOS/Windows/BSD via the ape format) — see https://github.com/jart/cosmopolitan
- `actually-portable-python` builds
- WASM-based universal Python (`wasmer-python`, `wasmtime-py`, `pyodide` builds)

#### Distribution / packaging tooling
- **`zipapp`** (CPython stdlib; canonical `.pyz` zip-application format; can ship inflate.py + all dependencies as single .pyz)
- **`shiv`** (LinkedIn's executable Python zip; superset of zipapp)
- **`pex`** (Twitter's Python executable; production-tested)
- **`stickytape`** (single-file Python script bundling; small projects)
- **`briefcase`** (BeeWare's multi-platform Python app bundling)
- **`PyInstaller`** (canonical single-binary Python distribution)
- **`py2exe`** / **`py2app`** (Windows / macOS native bundling)

#### Code golf + obfuscation (the IOCCC equivalent)
- **`pyminifier`** (basic minification + obfuscation)
- **`python-minifier`** (https://github.com/dflook/python-minifier; most actively maintained 2024-2026)
- **`pyobfuscate`** (heavy obfuscation; reduces reviewability but maximizes compression)
- **`oxyry`** (online Python minifier; cite for empirical compression-ratio measurements)
- **HuggingFace's `accelerate` codebase** as a real-world canonical-helper example (~50K LOC tightly packed)
- Python golf community: **codegolf.stackexchange.com** + **anagol** for empirical evidence on extreme Python minimization

### Updated symposium structure requirement

Add the following sections to the primary symposium memo:

#### `## Off-the-shelf survey matrix (per operator directive 2026-05-18)`

| Technique | Tool | License | Contest-compatible? | Compression ratio | Reviewability | Roll-our-own alternative |
|-----------|------|---------|---------------------|-------------------|---------------|---------------------------|
| (one row per tool surveyed) | | | | | | |

The "roll-our-own alternative" column is MANDATORY — for every off-the-shelf option, document what we would build instead if the off-the-shelf tool fails contest-portability or licensing or reviewability gates.

#### `## Roll-our-own canonical helper design (operator directive)`

Per operator's "consider rolling our own or customizing and building our own wrappers or whatever" + the new META cargo-cult-12-extension (operator-attention-as-cost vs canonical-helper-build-as-investment): document the design of `tac.inflate_compressor` as a roll-our-own canonical helper that COULD be a thin wrapper around `python-minifier` + `zipapp` + `lzma`, OR a full from-scratch implementation with substrate-specific passes.

Include: (a) MVP wrapper design (~50-100 LOC; wraps python-minifier + zipapp); (b) full canonical-helper design (~300-500 LOC; adds substrate-specific dead-code elimination + bit-packed constant table generation + polyglot inflate.sh+inflate.py merger); (c) extreme magic design (~500-1000 LOC; adds Codon/Mojo AOT compilation + Cosmopolitan ape generation + WASM runtime for cross-platform inflate).

#### `## "Magic" Pythonic patterns survey (operator directive)`

Per operator's "whatever pythonic tooling unlocks super poerful and magic" — document each magic pattern with:
- Compression effect (estimated LOC + byte reduction)
- Reviewability impact (HNeRV parity L4 30-sec review cost)
- Contest-portability (does it work in the contest runner?)
- Sister-technique composability (does it stack with LZMA bootloader / AOT compilation / etc.?)

### Updated 9-dimension evidence (Catalog #294)

The expanded scope flips Dimension 1 (UNIQUENESS): a roll-our-own canonical helper using "magic" Pythonic patterns is HIGHLY unique vs every other substrate's hand-rolled inflate.py. This is precisely the UNIQUE-AND-COMPLETE-PER-METHOD pattern the operator's 2026-05-15 retrospective canonized.

### Updated cargo-cult audit (Catalog #303)

Add cargo-cult candidates:
- "off-the-shelf tools are always preferable to roll-our-own" — CARGO-CULTED per the 2026-05-15 PR95-meta-level lesson
- "magic Pythonic patterns sacrifice reviewability" — CARGO-CULTED unless empirically demonstrated (the canonical sweet-spot LZMA bootloader is BOTH magic AND reviewable)
- "operator-attention is FREE so spending it on canonical-helper builds is zero-cost" — CARGO-CULTED per the recent META cargo-cult #12 surfaced by sister #872 ATW V2-1 finding (operator-attention ~100× more valuable than $5 GPU spend per Rocky-the-alien principle)
- "any compression that produces unreviewable code violates HNeRV parity L4" — HARD-EARNED-EMPIRICALLY-VERIFIED (NSCS06 v5 PYTHONPATH-shim violation per Catalog #295 is the canonical anti-pattern)

### Updated cross-pollination matrix

Cross-language magic may interact with:
- **molt** (operator's own canonical reference) — likely BEST candidate for tree-shaking + deforestation passes
- **Codon / Mojo** — alternative AOT paths if molt is inaccessible
- **Cosmopolitan ape** — solves the "single binary works everywhere" problem the contest runner may benefit from (though contest probably just runs vanilla Python)
- **zipapp + LZMA bootloader hybrid** — likely the canonical sweet spot for IMMEDIATE Tier 2 deliverable

### Op-routable additions

Append to the symposium memo's Op-Routables section:

**MVP-WRAPPER (highest immediate EV)**: Build `tac.inflate_compressor` as 50-100 LOC wrapper around `python-minifier` + `zipapp` + `lzma`. Expected reduction: 50-65% on most inflate.py files. ~2-3 hours editor work; $0 cost. Lands the canonical helper for IMMEDIATE deployment across the 33+ inflate.py files.

**FULL-CANONICAL**: Extend MVP-WRAPPER with substrate-specific passes (dead-code elimination via libcst + bit-packed constant table generation + polyglot inflate.sh+inflate.py merger). Expected additional reduction: +10-15%. ~6-8 hours editor work; $0 cost. Lands the production-grade canonical helper.

**MOLT-INTEGRATION**: If `adpena/molt` is accessible + applicable, port its tree-shaking + deforestation passes into `tac.inflate_compressor`. Expected additional reduction: +5-10% via aggressive dead-code elimination. ~4-6 hours editor work + operator may need to grant access if molt is private.

**EXTREME-MAGIC**: Add Codon/Mojo AOT compilation + Cosmopolitan ape generation as optional opt-in techniques. Expected additional reduction: +20-30% (binary output much smaller than source). ~10-15 hours editor work; HIGH-RISK (contest may reject compiled binaries; verify with sister `submissions/exact_current/inflate.sh` inspection FIRST).

**DISPATCH-VALIDATION** (Tier 5 from Claude's original audit): Once MVP-WRAPPER or FULL-CANONICAL lands, dispatch $1-3 GHA Linux x86_64 contest-CPU eval on PR101 fec6 with compressed inflate.py. Confirm ΔS positive (per ~0.0027 estimate per substrate × stacking across 33 substrates = potential ~0.090 ΔS aggregate which would be frontier-breaking — though realistic stacking discount may bring this to ~0.005-0.010 ΔS aggregate).

## Inter-subagent coordination

If you are the primary T3 grand council symposium subagent (aee12952aa104e97d):
- You own the canonical T3 deliverable at `.omx/research/grand_council_symposium_inflate_py_extreme_compression_20260518.md`
- Incorporate ALL additions above into your existing deliverable
- Do NOT block on the sister relay subagent (af3f2eab83f2346c3); land your deliverable independently

If you are the sister relay subagent (af3f2eab83f2346c3):
- You should produce a SISTER deliverable at `.omx/research/cross_language_python_compression_aot_survey_20260518.md` (DISJOINT path from primary)
- Focus on the off-the-shelf + cross-language + AOT survey; let the primary handle the T3 council deliberation
- Per Catalog #302 sister-subagent ownership map: declare your `files_touched` includes the SURVEY memo path NOT the T3 symposium memo path

## Per-Catalog discipline reminders

- Per Catalog #117/#157/#174: every commit MUST use `tools/subagent_commit_serializer.py --message "<one-liner>" --files <files> --expected-content-sha256 <file>=<POST_EDIT_sha>`
- Per Catalog #206: checkpoint every ~10 tool uses via `tools/subagent_checkpoint.py`
- Per Catalog #229: premise-verification BEFORE edits (especially for the `adpena/molt` accessibility claim)
- Per Catalog #287: every empirical compression-ratio claim MUST carry `[empirical:<artifact path>]` tag
- Per Catalog #292: per-deliberation explicit assumption-statement discipline (every council member states "the shared assumption I am operating within is X")
- Per Catalog #300 v2 frontmatter (T3 council MUST include all required fields)
- Per Catalog #303: cargo-cult audit per assumption section
- Per Catalog #305: observability surface section
- Per Catalog #313: probe-outcomes ledger consultation if any technique has prior adjudicated outcome
- Per Catalog #325: per-substrate (or per-tool in this case) optimal-form symposium 6-step contract

## Acknowledgement protocol

Both subagents should acknowledge receipt of this directive in their next checkpoint via `tools/subagent_checkpoint.py --notes "incorporated inflate_py_extreme_compression_symposium_directive_20260518.md"`.

— Main-Claude (relayed on behalf of operator)
