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

## CONDITIONAL: measure before spending the slot on this

**Do not run this arm until the relocation rate has been measured on the POLICY'S OWN TAPS.**

Every relocation figure so far — 51.7% of 3x3 taps moving, 15.0% losing the chosen tile — is a
uniform sweep over legal tiles. This policy is not uniform and is getting less so:

    policy/tile_top1_share        0.137 -> 0.170    one tile is 17% of ALL plays
    policy/card_tile_top10_share  0.234 -> 0.348    top ten (card, tile) pairs are 35%
    policy/tile_entropy            4.52 ->  3.95

So 15% is a property of the board, not of this agent. If its favoured building tiles are ones a
3x3 does not fit, the mechanism is far larger than 15% here; if they are open ground, it is far
smaller and this arm would spend eight hours on a non-effect.

`TraceStep` now carries `landed` — the RESOLVED deploy position, parallel to `commands` — as of
RoyaleGym 92f8a49. So the measurement is: record a trace from hog26-8's final checkpoint, compare
each command against its `landed`, and get the loss rate on the distribution the policy actually
taps. That is minutes against this run's eight hours.

**Run this arm if that rate is materially above zero. If it is near zero, relocation is not what
suppresses Cannon here and the reward becomes the candidate** — a potential over committed elixir
counts board entities, and a spell never becomes one.

(Gym's note on why the card behind a step cannot be reconstructed by hand: frames are recorded
AFTER `engine.step()`, so the frame nearest a step shows the hand the play already cycled — the
replacement card, not the card played. `card_ids` removes that rather than documenting it.)
