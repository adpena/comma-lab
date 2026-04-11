"""Allow ``python -m tac`` to invoke the CLI."""
from tac.cli import main

result = main()
raise SystemExit(result if isinstance(result, int) else 0)
