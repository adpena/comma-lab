"""Allow ``python -m tac`` to invoke the CLI."""
import os
import sys

# Force unbuffered output so training logs stream in real-time
# when redirected to files, pipes, or remote monitoring.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from tac.cli import main

result = main()
raise SystemExit(result if isinstance(result, int) else 0)
