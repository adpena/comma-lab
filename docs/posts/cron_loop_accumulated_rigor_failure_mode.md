# Cron-as-fan-out-cadence: an agent-loop failure mode that compounds your rigor against you

When you wire a Claude/Codex agent to a cron-driven outer loop, the temptation is to use the cron interval as the rate-limiter for your dispatch fan-out. It feels right: a 5-minute cron means the agent considers up to 12 candidate dispatches per hour, well within your GPU budget. The agent ranks. The agent dispatches the top one. The cron sleeps. Five minutes later the agent ranks again, possibly with a new candidate, and the loop continues.

This is wrong. It looks rigorous, and that is exactly what makes it dangerous.

I lost a contest this way. I want to walk through what happened, why the architecture was the bug rather than any individual decision, and what to look for in your own loops.

## The loop that made sense at design time

The setup was a video-compression contest. Submissions were ranked by a single composite score. I had ~10 distinct optimization "lanes" — independent attack vectors against the scorer — and each lane could produce candidate archives at a rate of roughly one every 30 seconds of local CPU work. A real evaluation, the only kind that counted on the leaderboard, took 5-30 minutes on a rented GPU and cost between $0.20 and $1.50 per attempt.

The natural protocol:

1. Generator produces N candidates locally (cheap).
2. Local proxy ranker sorts them (cheap, has a known 2-11x gap to the real scorer).
3. Top-K candidates get dispatched to GPU (expensive, authoritative).
4. Harvest results, feed them back into the ranker as new calibration anchors.
5. Repeat.

I scheduled step (1)+(2)+(3) every 5 minutes via cron, with a top-K of 1.

That last sentence is where the failure lives.

## What the cron-as-cadence pattern actually does

Two patterns superficially look identical:

```python
# Pattern A: rank-then-dispatch
def loop_iteration():
    candidates = generate()      # ~30s
    ranked = rank(candidates)    # ~1s
    dispatch(ranked[0])          # fire-and-forget; harvest async

# Pattern B: rank-then-rank-more
def loop_iteration():
    candidates = generate()      # ~30s
    ranked = rank(candidates)    # ~1s
    ranked = rank_more(ranked)   # cheap re-ranking with newest anchors
    dispatch(top_eligible(ranked))  # may dispatch zero, one, or many
```

Pattern A is what cron-driven loops naturally produce, because the cron *is* the scheduler. The agent only sees its current snapshot, fires, sleeps. Pattern B is what you actually want, because it lets the agent revise its commitment based on incoming evidence before paying for the next dispatch.

If your real-eval results land asynchronously and feed back into the ranker, Pattern A wastes most of the harvest. The ranker gets smarter, but the smarter ranker only sees one new candidate per cron tick — never a fresh look at the whole queue. By the time the agent revisits a candidate it dispatched ten ticks ago, it's already paid the cost.

Worse: every tick of cron is *also* a fresh agent-context boot. There's no continuity. Whatever rigor the previous tick accumulated — the suspicion that lane 4 was over-fitting, the pattern across the last three failed dispatches — has to be re-derived from disk every time. And if your loop logic is "spawn agent, agent re-derives state, agent fires top-1, agent exits", then any heuristic the agent develops mid-session evaporates at exit. You are paying for a 30-second agent-bootup just to make a 1-second decision.

## The contest postmortem in numbers

Here is the LOC table from the postmortem:

| Component                              | LOC   | Purpose                                  | Outcome           |
|----------------------------------------|-------|------------------------------------------|-------------------|
| Cron driver                            | 27    | Trigger agent every 5 min                | Worked as written |
| Agent dispatch logic                   | 312   | Rank, gate, fire one                     | Worked as written |
| Candidate generator (apogee_intN lane) | 184   | Produce 5 candidates per lane            | Worked as written |
| Local proxy ranker                     | 891   | Score candidates without GPU             | Worked as written |
| Real-eval harvester                    | 156   | Pull results back into anchors           | Worked as written |
| Anchor calibration update              | 67    | Refit predictor from new anchors         | Worked as written |
| **Total system LOC**                   | **1637** | (none of which were buggy in isolation) | **Lost the contest** |

Every component was unit-tested and individually correct. The integration was the bug. The cron interval was a fan-out cadence dressed up as a rate-limiter, and because the agent's working window was shorter than the harvest window, every harvest landed in a new agent process that had no memory of what it had been thinking five minutes ago.

By the time the leaderboard moved twice in two hours and a competitor's submission landed at a score 4× better than mine, my agent loop had dispatched 14 candidates from one lane, harvested 11, and adjusted its calibration anchors 11 times — but had never re-considered the other 9 lanes, because each cron tick only produced candidates from one lane at a time and the cron-tick budget was always spent on the new top-1 from that single lane.

## The architectural diagnosis

The diagnosis is in §7.8 of my contest writeup, which I'll quote here because the framing matters:

> A cron-driven agent loop has two clocks: the cron clock (when the agent thinks) and the dispatch clock (when paid work happens). When dispatch_clock_period > cron_clock_period, the cron loop is fanning work out faster than results come back. This is fine for embarrassingly parallel workloads where every dispatch is independent. It is catastrophic for closed-loop optimization, because the ranker never gets a chance to consume the harvest before committing to the next dispatch. The agent appears rigorous (it ranks, gates, dispatches with care) but the rigor is wasted: each cron-tick is a fresh boot with no working memory of what the *previous* cron-tick learned.

The fix is not to change the cron interval. The fix is to change which clock owns the loop. **Make the dispatch clock the outer clock.** The agent should run continuously (or at least across enough ticks that one harvest cycle completes inside one agent session), and dispatch should be paced by harvest-rate, not by wall-clock.

In our project this meant rewriting the loop driver to:

- own its own state machine (not start from disk every tick),
- block until the in-flight dispatch returns OR a wall-clock budget is hit,
- consume the harvest in-process and re-rank the entire queue before dispatching again, and
- treat cron as a heartbeat / liveness check, not a scheduler.

That's `tools/feedback_loop_sweep.py` in this repo, which preserves the rank-then-rank-more pattern in a single long-lived process. Cron exists in our setup, but only to ensure the loop is still running — if the cron job notices no progress in the last 30 minutes, it restarts the loop. It does not own the cadence.

## What to look for in your own agent loops

A checklist, in roughly the order things go wrong:

1. **Two clocks.** Identify the cron clock and the dispatch clock. If you only have one clock, you're fine. If you have both and the cron is faster than dispatch, you're at risk.
2. **Cross-tick memory.** Does anything the agent learned on tick N influence its dispatch decisions on tick N+1, beyond what's serialized to disk? If yes, every cron-respawn is throwing that away.
3. **Top-K = 1 per tick.** The most dangerous variant. The agent looks rigorous because it gates before dispatching, but it can never benefit from re-ranking, because re-ranking 1-of-1 is a no-op.
4. **Harvest latency > cron interval.** If your real-eval takes 15 minutes and your cron runs every 5 minutes, you're firing 3 dispatches before the first one returns. The ranker has zero new information when it picks dispatch #2 and #3.
5. **Per-tick agent re-derivation cost.** If your agent spends >10% of each cron tick re-reading state files, recomputing rankings, or warming up a model just to fire one dispatch, you're paying a fixed cost N times instead of once.
6. **Lane diversity decay.** Within one cron tick, how many distinct optimization directions does the agent consider? If the answer is 1 (because the generator emits one lane's worth of candidates per call), the loop is monotonically narrowing into the lane it considered first. Other lanes get starved without you noticing.
7. **No "rank-more" step.** Look at your dispatch code. Does the rank step happen *only* over freshly-generated candidates, or does it pull in the persistent queue and re-rank it against the latest anchors? The first is rank-then-dispatch. The second is rank-then-rank-more.
8. **Asymmetric costs.** Cheap-to-rank, expensive-to-dispatch is the usual configuration. The ranker should be doing 100x the work of the dispatcher. If your ranker is silent for most of the loop and only runs when fresh candidates arrive, your CPU budget is hiding the architecture problem.

## The general lesson

When you give an agent a cron, you're giving it a clock. Clocks are not free; they impose a worldview. A 5-minute cron tells the agent that 5 minutes is the natural unit of decision-making. When the actual decision unit is "until the next harvest lands", the cron lies to the agent and the agent makes worse decisions for it.

The right way to wire an agent into a long-running optimization loop is to give it ownership of the loop. The agent decides when to dispatch, when to wait, when to re-rank, when to exit. Cron becomes a watchdog, not a scheduler. The agent's working memory persists across the natural decision unit, not across a wall-clock interval that has nothing to do with the work.

This is more expensive in agent-tokens (a long-lived agent burns more context than a short-lived one). It is much, much cheaper in dispatched-but-wasted GPU dollars and in lost contest position. Make the trade.

## Appendix: the canonical fix in 30 lines

```python
def closed_loop(generator, ranker, dispatcher, harvester, *, max_cycles, budget_usd):
    spent = 0.0
    best = float("inf")
    no_improve = 0
    for cycle in range(max_cycles):
        if spent >= budget_usd:
            break
        # 1. RANK over the FULL queue, not just freshly generated candidates
        candidates = generator()           # may include held-over candidates
        ranked = ranker(candidates)         # uses anchors from previous cycles
        eligible = [c for c in ranked if dispatcher.gate_pass(c)]
        if not eligible:
            break
        # 2. DISPATCH a small batch in parallel — the ranker chose them together
        results = dispatcher.fire_batch(eligible[: dispatcher.top_k_for_cycle(cycle)])
        spent += sum(r.cost_usd for r in results)
        # 3. HARVEST in-process; refit anchors before next iteration
        new_anchors = harvester.collect(results)
        ranker.absorb(new_anchors)
        # 4. CONVERGE check: if best score hasn't moved, stop
        cycle_best = min((r.score for r in results if r.score is not None), default=None)
        if cycle_best is not None and cycle_best < best:
            best = cycle_best
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 2:
            break
    return best
```

The key line is `ranker.absorb(new_anchors)`. In the cron-driven version, that line cannot exist, because the agent process exits before the next dispatch is even contemplated. In the long-lived version it costs nothing and recovers most of the rigor that the cron pattern destroyed.
