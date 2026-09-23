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
