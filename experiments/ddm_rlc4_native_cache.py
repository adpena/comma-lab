#!/Users/adpena/Projects/pact/.venv/bin/python
"""Compile and retain exact public native builds; replay them on checkpoint resume.

The first request really invokes /usr/bin/cc. A repeat reuses only the same
source and flags, with byte-verified retained output. This is a compiler cache.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def fact(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    root = Path("/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42").resolve()
    cache = Path(os.environ["RLC4_NATIVE_CACHE"]).resolve()
    argv = sys.argv[1:]
    if not cache.is_relative_to(root) or len(argv) < 3 or argv[-2] != "-o":
        raise ValueError("unrecognized owned compile request")
    dest = Path(argv[-1]).resolve()
    if not dest.is_relative_to(Path(os.environ["TMPDIR"]).resolve()):
        raise ValueError("compile destination outside scratch")
    sources = [Path(a).resolve() for a in argv[:-2] if a.endswith(".c")]
    if len(sources) != 1 or not sources[0].is_relative_to(root / "candidate_runtime"):
        raise ValueError("compiler source must be owned candidate C")
    template = [*argv[:-1], "{output}"]
    binding = {"argv_template": template, "source": fact(sources[0]),
               "compiler": fact(Path("/usr/bin/cc").resolve())}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    receipt, library = cache / f"{key}.json", cache / f"{key}.so"
    if receipt.exists():
        row = json.loads(receipt.read_text())
        if row["binding"] != binding or row["library"] != fact(library):
            raise ValueError("native cache custody drift")
        dest.write_bytes(library.read_bytes())
    else:
        if library.exists():
            raise ValueError("unreceipted native build retained; recover before resume")
        subprocess.run(["/usr/bin/cc", *argv], check=True)
        library.write_bytes(dest.read_bytes())
        row = {"binding": binding, "library": fact(library), "original_output": str(dest),
               "command": ["/usr/bin/cc", *argv], "score_claim": False}
        temporary = receipt.with_suffix(".new")
        temporary.write_text(json.dumps(row, indent=2) + "\n")
        temporary.replace(receipt)
    if fact(dest)["sha256"] != row["library"]["sha256"]:
        raise ValueError("compiler output differs from retained bytes")


if __name__ == "__main__":
    main()
