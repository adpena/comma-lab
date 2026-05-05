---
name: uv is mandatory
description: Non-negotiable — always use uv for Python package management, never raw pip
type: feedback
---

Always use `uv` for package management. Never use `pip`, `pip3`, or `pip install` directly.

**Why:** User mandate. uv is faster, more reliable, and already installed everywhere.

**How to apply:**
- Install: `uv pip install <pkg>`
- Venvs: `uv venv`
- Run: `.venv/bin/python`
- Remote machines: install uv first, then use it
- Exception: cloud containers (Modal/Kaggle/Colab) where uv isn't available
