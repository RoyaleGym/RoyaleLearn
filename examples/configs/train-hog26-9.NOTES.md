# train-hog26-9: does the building decline survive an honest tap mask?

**One field differs from `train-hog26-8.json`**: `env.action_parser.kwargs.buildings` is
`taps_where_the_building_stays` instead of the default `any_tap`. Same seed, same `noop_bias`,
same dtype, same everything else, so hog26-8 is its control.

## The question

Under the default arm, hog26-8 drove Cannon's per-opportunity play rate to **0.6%** and Log's to
**1.0%**, while Hog Rider reached **95%** and Musketeer **80%**. Every troop is played; the
building and the spells are refused.

The engine relocates a building whose footprint does not fit, so for a 3x3 the agent **loses the
tile it chose about 15% of the time** (gym, full tile scan over all 11 building cards, build
d872d792711934c2; Tesla, being 2x2, is 10%). A policy graded on a placement it did not make cannot
learn that placement. The honest arm offers only taps the building stays on.

**If the decline is relocation, it should weaken under this arm. If Cannon stays near 0.6%, it is
not relocation** — and the reward becomes the leading candidate, since a potential over committed
elixir counts board entities and a spell never becomes one.

## Three things to hold while reading the result

**Mask density is a competing explanation and it is not small.** The honest arm offers 116 of 240
Cannon taps and drops 124 — of which 88 would still have left the building on the chosen tile.
Tesla drops only 73 of 240. So the arm changes exploration asymmetrically by footprint, and a
change in Cannon's rate could be relocation OR simply fewer ways to express the same landing tile.
This deck has no Tesla, so the within-deck control gym proposed is not available here.

**Never flip this mid-run.** It changes the mask, not the shape of the action space.

**The two arms are not comparable as policies**, only as experiments. `n_actions` is 2305 under
both — measured, so a checkpoint loads across them — but the same logit indexes a different
offered set, so a policy trained under one is not a policy trained under the other.

## What made this runnable without a code change

Gym verified three things, and the second is the one that would have wasted the run:
`buildings` round-trips as a JSON kwarg; `config()` **records** the arm, so
`{"buildings": "taps_where_the_building_stays"}` is distinguishable from `{"buildings":
"any_tap"}` in a resumed run — had it been omitted, a resume would have silently reverted to the
default action space and the failure would have read as "the intervention did not work" rather
than as a config bug; and `n_actions` is identical under both.
