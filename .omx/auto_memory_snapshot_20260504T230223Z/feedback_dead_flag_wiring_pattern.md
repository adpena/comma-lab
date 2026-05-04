---
name: NEVER invent CLI flags — always read target tool argparse first
description: The "council fixes can themselves be bugged" meta-pattern: shipped TWO rounds of auth-eval-on-best wiring that referenced flags (--auth-eval-masks) that the target tool (auth_eval_renderer.py) never had. Council R3 caught it; user was rightly furious about repeating the same lesson.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Binding rule:** Before wiring any CLI arg into a `subprocess.run([...])` call, READ the target tool's actual `parser.add_argument(...)` list. Don't invent flag names from intent. Don't trust prior PRs/code that "looked like it worked" — verify against argparse.

**Why this rule exists** (2026-04-26 incident):

R1 wiring of `train_renderer.py --auth-eval-on-best`:
- Added `--auth-eval-masks` arg that I assumed `auth_eval_renderer.py` would consume
- Reality: `auth_eval_renderer.py` has NO `--masks` flag (it recomputes SegNet masks from GT internally)
- Council R1 didn't catch it (focused on rate-math)

R2 "fix":
- Council R2 caught the rate term arithmetic ambiguity, default-False issue
- "Fixed" the default to True
- BUT the dead-flag wiring stayed
- Now every existing chain SILENTLY SKIPPED the eval (because masks weren't passed)
- AND every chain that DID pass masks would have failed argparse on the subprocess
- User: "you keep confusing proxy and local with auth eval over and over and over again despite everything I try"

R3 "fix":
- Council R3 finally read the actual tool: no --masks flag
- Real fix: build archive with `build_submission_archive`, pass `--archive-size-bytes` from built file
- Test: `test_train_renderer_auth_eval_wiring.py` (8 tests) introspects auth_eval_renderer.py argparse and asserts train_renderer doesn't emit any flag not in that set

**How to apply:**

1. **Before wiring `subprocess.run([...args])`**, run `grep "add_argument" path/to/target.py` and list the actual flag names. Compare to your `args` list. Any extra flag = dead code (will be rejected by argparse OR silently ignored if `parse_known_args`).

2. **Add a regression test that introspects the target tool's argparse** and asserts the calling code's flag set is a subset. See `src/tac/tests/test_train_renderer_auth_eval_wiring.py::test_train_renderer_does_not_pass_invented_flags`.

3. **Fail loud, not silent**, when wiring is misconfigured. If `--auth-eval-on-best` is True but the inputs (masks/poses) needed to make it meaningful are missing, raise — don't print a WARN and continue.

4. **Council reviews ARE NOT a substitute for reading the target tool's source.** R1, R2, and R3 each missed pieces because they reasoned about intent rather than verifying against the actual argparse. Adversarial review catches LOTS of issues but a focused argparse-grep would have caught the dead-flag wiring on round 1.

5. **The user has called out this pattern explicitly:** *"it is unacceptable to learn the same lessons over and over again."* The cost of these false-fix cycles is days of wasted GPU + lost confidence. Spend the 2 minutes on the grep.
