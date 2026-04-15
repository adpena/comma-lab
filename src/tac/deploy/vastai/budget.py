"""Budget tracking for Vast.ai experiments.

Reads/writes ``experiments/results/vastai/budget.json`` and enforces a hard
spending cap so that runaway instances cannot exceed the allocated budget.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tac.deploy.base import BudgetState, repo_root

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_HARD_CAP_USD: float = 24.0
"""Hard spending cap in USD.  Exceeding this blocks new launches."""

DEFAULT_WARN_THRESHOLD_USD: float = 20.0
"""Cumulative spend that triggers a warning (but does not block)."""

DEFAULT_TOTAL_BUDGET_USD: float = 25.0
"""Informational: total account credit available."""

# ANSI helpers (duplicated intentionally — budget.py must be standalone)
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


class BudgetTracker:
    """Persistent budget tracker backed by a JSON file.

    Parameters
    ----------
    budget_file:
        Path to the JSON ledger.  Created automatically if absent.
    hard_cap:
        Maximum cumulative spend in USD.  ``check_remaining`` returns False
        once this is reached.
    warn_threshold:
        Spend level that emits a warning but does not block.
    """

    def __init__(
        self,
        budget_file: Path | None = None,
        hard_cap: float = DEFAULT_HARD_CAP_USD,
        warn_threshold: float = DEFAULT_WARN_THRESHOLD_USD,
    ) -> None:
        if budget_file is None:
            budget_file = repo_root() / "experiments" / "results" / "vastai" / "budget.json"
        self._path = Path(budget_file)
        self.hard_cap = hard_cap
        self.warn_threshold = warn_threshold
        self._state = self._load()

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def total_spent(self) -> float:
        """Cumulative USD spent so far."""
        return self._state.total_spent

    @property
    def remaining(self) -> float:
        """USD remaining before hitting the hard cap."""
        return max(0.0, self.hard_cap - self._state.total_spent)

    @property
    def sessions(self) -> list[dict]:
        """Raw session history list (read-only view)."""
        return list(self._state.sessions)

    def check_remaining(self, estimated_cost: float) -> bool:
        """Return True if *estimated_cost* fits within the remaining budget.

        Prints warnings/errors to stderr as side effects.
        """
        if self.remaining <= 0:
            print(
                f"{_RED}{_BOLD}BUDGET EXHAUSTED. "
                f"Spent ${self._state.total_spent:.2f} / ${self.hard_cap:.2f}{_RESET}",
                file=sys.stderr,
            )
            return False

        if estimated_cost > self.remaining:
            print(
                f"{_RED}Insufficient budget. "
                f"Need ${estimated_cost:.2f}, have ${self.remaining:.2f} remaining{_RESET}",
                file=sys.stderr,
            )
            return False

        projected = self._state.total_spent + estimated_cost
        if projected > self.warn_threshold:
            print(
                f"{_YELLOW}WARNING: This will bring spend to ${projected:.2f} "
                f"(warn threshold: ${self.warn_threshold:.2f}){_RESET}",
                file=sys.stderr,
            )

        return True

    def record_spend(
        self,
        instance_id: str,
        amount: float,
        description: str,
    ) -> None:
        """Append a spending event and persist immediately."""
        self._state.total_spent += amount
        self._state.sessions.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instance_id": instance_id,
            "amount": amount,
            "description": description,
            "cumulative": self._state.total_spent,
        })
        self._save()

    def print_summary(self) -> None:
        """Print a human-readable budget summary to stdout."""
        pct = (self._state.total_spent / self.hard_cap) * 100 if self.hard_cap > 0 else 0

        if self.remaining <= 0:
            color = _RED
        elif self._state.total_spent > self.warn_threshold:
            color = _YELLOW
        else:
            color = _GREEN

        print(f"\n{_BOLD}Vast.ai Budget Tracker{_RESET}")
        print("=" * 50)
        print(f"  Total spent:  {color}${self._state.total_spent:.2f}{_RESET}")
        print(f"  Hard cap:     ${self.hard_cap:.2f}")
        print(f"  Remaining:    {color}${self.remaining:.2f}{_RESET} ({100 - pct:.0f}%)")
        print(f"  Total budget: ${DEFAULT_TOTAL_BUDGET_USD:.2f}")

        if self._state.sessions:
            print(f"\n  {_BOLD}Session history:{_RESET}")
            recent = self._state.sessions[-10:]
            for session in recent:
                ts = session["timestamp"][:19].replace("T", " ")
                amt = session["amount"]
                desc = session["description"]
                if amt > 0:
                    print(f"    {ts}  ${amt:>7.2f}  {desc}")
                else:
                    print(f"    {_DIM}{ts}  ${amt:>7.2f}  {desc}{_RESET}")
            if len(self._state.sessions) > 10:
                extra = len(self._state.sessions) - 10
                print(f"    {_DIM}... and {extra} earlier entries{_RESET}")

        print()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> BudgetState:
        """Load state from disk, returning a fresh state if file is absent."""
        if self._path.exists():
            data = json.loads(self._path.read_text())
            return BudgetState.from_dict(data)
        return BudgetState()

    def _save(self) -> None:
        """Persist current state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state.to_dict(), indent=2) + "\n")
