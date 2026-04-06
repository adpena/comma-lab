"""Tiny placeholder for a JAX surrogate lane.

This file is intentionally small. The mainline repo should not depend on JAX
until a measured need appears.
"""

from __future__ import annotations


def note() -> str:
    return "JAX lane is intentionally inactive until promoted by evidence."


if __name__ == "__main__":
    print(note())
