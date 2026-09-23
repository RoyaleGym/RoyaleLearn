# train-hog26-8: why the timestep limit is 2,700,000 and not a round number

`timestep_limit` is **2,700,000**, which is 493 iterations at this run's measured 5,472 env steps
per iteration. It is not a budget and rounding it up is not a small change.

`ladder.candidate_every_env_steps` is 1,500,000, so:

    2,700,000 steps = 493 iterations = ONE ladder candidate
    4,000,000 steps = 730 iterations = TWO

The first candidate into an empty pool is admitted unconditionally and costs 0.017 s. **The second
is a real gate**: 2,200 evaluation battles with no short-circuit (champion 1,000 + anchors 2x200 +
stratified 8x100), at a measured 8.02 s per network-vs-network battle. That is **4.9 hours**,
longer than the 3.9 hours of training it interrupts.

So the two runs cost about 4.8 and 9.7 hours, and nothing else in the config says so. A run raised
to 4M looks like "a bit longer" and is a run somebody kills at hour six believing it has hung.

If you want a real gate, raise it deliberately and budget for it. If you want a longer TRAINING
run, raise `ladder.candidate_every_env_steps` in step so the candidate count stays at one.

## The numbers above, and where they came from

All of them were measured on one laptop, on the runs this config descends from, so treat them as
the shape of the cost rather than as constants:

    env steps per iteration      5,472   identical on two 134- and 147-iteration runs
    seconds per iteration        27-30   medians 29.3 and 27.5, p90 30.7 and 28.5
    evaluation battle, net v net  8.02 s  median of 3, randomly initialised network on CPU
    evaluation battle, scripted   0.195 s
    peak VRAM                    2,011 MB, with zero worker restarts
    projected system RAM          2,412 MB, and the preflight prints it against what is free

The last one is the one that bites. It is a projection made at start-up, and a run that begins with
headroom and loses it two hours later dies two hours later. Watch free memory for the whole run,
not at the door.

## Is one ladder candidate enough? Yes, because the ladder is not this run's instrument

A reader will see 493 iterations producing exactly one unconditional admission and reasonably ask
whether the ladder did anything. It did not, and that is deliberate.

**The question this run asks is whether the policy keeps becoming less uniform if it is allowed to
run long enough.** Every run before it stopped at 147 iterations or fewer. The instruments for that
question are:

    ppo/logit_std                     the spread of the masked logits over choosing rows
    1 - ppo/entropy_normalised        the entropy deficit, the same quantity seen worse
    ladder/score_vs_random_legal      20 probes, an ABSOLUTE signal against a fixed opponent

None of the three depends on the pool. The probe plays a scripted opponent that never changes, so
it measures the policy against a fixed yardstick rather than against its own history.

**What the ladder would add is a different question** — whether the policy beats its own past
selves — and that one needs a populated pool and real gates, which is a run that costs twice this
one. Asking it before knowing whether the policy improves at all is the wrong order.

So the single admission is a side effect worth having rather than a measurement: it is the first
snapshot this project has ever admitted, and the pool-dependent machinery downstream of it has
never run with a member in place. Getting one in costs nothing here.

**Read the learning curve, not the ladder.** If `logit_std` is still climbing at iteration 493 the
run should be longer; if it has flattened, that is the answer and no gate would have told you.
