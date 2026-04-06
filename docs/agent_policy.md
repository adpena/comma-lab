# agent policy

The agent is not the source of truth.
The score and the snapshot are the source of truth.

## Constraints

- Small diffs
- Short experiment batches
- Measured promotion
- No edits outside the mutation frontier
- No claims without numbers

## Roles

- Director: chooses the next 1 to 3 experiments
- Mutator: edits only allowed files
- Referee: runs smoke, proxy, and full eval checks
- Scout: only wakes up when the frontier stalls or upstream changes
- Reporter: turns structured logs into markdown and plots

## Anti-patterns

- huge rewrites
- vague "research more" loops
- changing many variables at once
- inventing performance numbers
- relying on a proxy when the official evaluator is affordable
