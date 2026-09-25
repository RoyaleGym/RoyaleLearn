# extensions.md: adding your own part to a run

You can add to a training run without editing RoyaleLearn. Write your code as a package of its
own, give it a section in the run's config, and install it next to RoyaleLearn. A run whose config
has your section runs your code. A run whose config does not have it never imports your package,
so your package cannot change anything about that run.

This page builds one small made-up extension from start to finish, then lists what an extension
can do and what RoyaleLearn checks for you. Section 19 of [harness-spec.md](harness-spec.md) is
the dense version. The example on this page is run by `tests/test_extension_example.py`, so it
works as printed.

## 1. The example: a confidence cap

Say you want to stop the bot from betting everything on one action early in training. This
extension adds a small penalty to the actor's loss whenever the bot puts more than a set
probability on its top action. It reports how often that happened, and it warns you when it
happens on most choices.

A run turns it on with one section in its config file, next to the other top-level keys such as
`ppo` and `net` (not inside one of them):

```json
{
  "confidence_cap": {"cap": 0.9, "coefficient": 0.01}
}
```

### The code

The whole package is one file, `confidence_cap/__init__.py`:

```python
"""confidence_cap: a penalty on putting too much probability on one action."""

import sys

import msgspec
import torch

from royalelearn.extensions import ExtensionBase, MetricAlarm, MetricSpec, SchemaContribution

__version__ = "0.1.0"

OVER = "confidence_cap/over_cap_frac"
RATIO = "confidence_cap/grad_ratio"


class Alarms(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    over_cap_frac: float = 0.5


class Section(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    cap: float = 0.9
    coefficient: float = 0.01
    alarms: Alarms = Alarms()


class CapTerm:
    """Adds coefficient * (how far each choice's top probability is over the cap)."""

    extension = "confidence_cap"
    name = "penalty"
    format_version = 1
    scaling = "rows"
    measure_grad_ratio = True

    def __init__(self, section):
        self.section = section
        self.over = 0
        self.rows = 0

    def begin(self, env_steps, device):
        self.over = 0
        self.rows = 0

    def loss(self, inputs, *, epoch, measure):
        zeros = torch.zeros_like(inputs.log_probs)
        probs = torch.where(inputs.mask, inputs.log_probs.exp(), zeros)
        top = probs.max(dim=-1).values
        if measure:
            self.over += int((top > self.section.cap).sum())
            self.rows += top.numel()
        return self.section.coefficient, torch.relu(top - self.section.cap).sum()

    def finish(self, *, iteration, actor_trained, explained_variance, grad_ratio):
        if self.rows == 0:
            return {}
        fields = {OVER: self.over / self.rows}
        if grad_ratio is not None:
            fields[RATIO] = grad_ratio
        return fields

    def state(self):
        return {}

    def load_state(self, state):
        pass


class ConfidenceCap(ExtensionBase):
    name = "confidence_cap"
    package = sys.modules[__name__]
    section_type = Section

    def problems(self, section, config):
        found = []
        if not 0.0 < section.cap <= 1.0:
            found.append(f"confidence_cap.cap must be above 0 and at most 1, not {section.cap}")
        if section.coefficient < 0.0:
            found.append(f"confidence_cap.coefficient must be 0 or more, not {section.coefficient}")
        return found

    def actor_terms(self, section, ctx):
        return [CapTerm(section)]

    def alarms(self, section):
        limit = section.alarms.over_cap_frac
        return [
            MetricAlarm(
                "confidence_cap_crowded",
                lambda frac: frac > limit,
                keys=[OVER],
                patience=5,
                meaning=f"more than {limit:.0%} of choices put their top action over the cap",
            )
        ]

    def metric_schema(self, section):
        return SchemaContribution(
            metrics={
                OVER: MetricSpec(
                    unit="ratio",
                    description="Share of choices whose top action was over the cap.",
                    low=0.0,
                    high=1.0,
                ),
                RATIO: MetricSpec(
                    unit="ratio",
                    description="The penalty's gradient size over the policy loss's.",
                ),
            },
            alarm_metrics={"confidence_cap_crowded": (OVER,)},
        )


EXTENSION = ConfidenceCap()
```

Reading it from the bottom up:

- `EXTENSION` is the object RoyaleLearn loads. Its `name` is the config key it owns.
- `section_type` is the shape of the section. It must refuse keys it does not know, which is what
  `forbid_unknown_fields=True` does, so a typo inside the section is an error and not a silent
  default.
- `problems` checks the values before anything is built. Whatever it returns is printed, and the
  run does not start.
- `actor_terms` gives the update one loss term. Section 3 has the details.
- `alarms` adds an alarm to the run's table. Put thresholds under `alarms` in the section. They are
  left out of the run's identity, so making an alarm louder or quieter is not a new experiment.
  Every other value in the section is part of the identity.
- `metric_schema` describes the keys the term reports, so the run's schema knows them like its own.
- `package` is the package itself. RoyaleLearn records its commit in each run's identity.

### Installing it

Keep the package in a git repository of its own, with the package folder at the top:

```text
confidence-cap/
    pyproject.toml
    confidence_cap/
        __init__.py
```

The `pyproject.toml` declares the section under the `royalelearn.extensions` entry-point group.
The name on the left is the config key, and the name on the right is the object to load:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "confidence-cap"
version = "0.1.0"

[project.entry-points."royalelearn.extensions"]
confidence_cap = "confidence_cap:EXTENSION"
```

Install it into the environment RoyaleLearn is in, from inside the `confidence-cap` folder:

```powershell
python -m pip install -e .
```

Commit before you train. Each run records the commit your package was at, next to RoyaleLearn's
own, so that two runs on two versions of your code never look like one run. For the same reason
a package whose commit cannot be found refuses to run. That happens when it was installed from a
wheel, or when it sits inside somebody else's repository. `train` also refuses to start while your
package has uncommitted edits, exactly as it does for RoyaleLearn's own code. `--allow-dirty` runs
anyway.

## 2. What an extension can do

An extension is a subclass of `ExtensionBase`, which gives every hook a default that does
nothing, so you write only the hooks you use. There are no others.

| Hook | When it is called | What it is for |
| --- | --- | --- |
| `problems(section, config)` | when the config is checked, before anything is built | refusing bad values with a message |
| `verify(section)` | at every start, fresh or resumed, before preflight | checking the files the section names, by content |
| `identity_value(section)` | when the run's identity is worked out | saying what part of the section is the experiment. By default, all of it except `alarms` |
| `prepare(section, ctx)` | once, after the model exists | loading what the extension needs. To replace the actor's starting weights here, also return True from `sets_starting_weights(section)` |
| `loaded(section, ctx)` | on a resume, after the checkpoint's weights are loaded | checking what the checkpoint brought back |
| `actor_lr_scale(section)` | when the schedules are built | multiplying the actor's learning rate by a schedule. Zero holds the actor still while the critic learns. At most one section in a run may do this |
| `actor_terms(section, ctx)` | once, before the first update | adding terms to the actor's loss |
| `alarms(section)` | when the alarm table is built | adding alarms |
| `metric_schema(section)` | when the run's schema is built | describing the keys the extension reports |

`ctx` is a `RunContext`. It carries the config, the environment description, the device, the live
model, whether the run is resuming, and helpers for building a spare actor and unpacking rows.

A run that holds the actor still gets two extra alarms, `actor_handoff` and `critic_unready`.
[running.md](running.md) section 3.6 describes them. The extension that sets the schedule adds
them with `freeze_alarms` and chooses their thresholds.

## 3. Loss terms

A term is any object with the attributes and methods `CapTerm` has. The update calls them in this
order:

- `begin(env_steps, device)` at the start of each update.
- `loss(inputs, epoch=..., measure=...)` on each minibatch, every epoch. `inputs.log_probs` and
  `inputs.mask` hold the actor's log-probabilities and legal-action mask for the minibatch's rows
  that had a real choice. `measure` is True in the first epoch, which is a good place to count
  things once. It returns `(coefficient, raw)`.
- `finish(...)` at the end of each update. It returns the keys for this iteration's metrics row.
  Every key must start with the term's `extension` and a slash.

The update adds `coefficient * raw * scale` to the actor's loss. With `scaling = "rows"`, `raw` is
a sum over the rows and `scale` is the one the policy loss uses, so the term's size does not
change when you change the minibatch size. With `scaling = "minibatch"`, `raw` is a mean you
computed yourself and `scale` is the minibatch's share of the batch.

With `measure_grad_ratio = True`, the update measures once per iteration how large the term's
gradient is next to the policy loss's, before your coefficient. It passes that to `finish` as
`grad_ratio`. That number is how you choose a coefficient: a ratio of 0.01 means your term is
barely steering, and a ratio of 10 means it is doing most of the steering.

On an iteration where the actor is held still, `loss` is not called at all and `finish` gets
`actor_trained=False`.

`state()` is what a checkpoint keeps for the term, and `load_state` gets it back on a resume. An
empty dict keeps nothing. Change `format_version` when the shape of the state changes: a checkpoint
saved at another version is refused, not misread.

## 4. What RoyaleLearn checks for you

When the config loads, and all at once, each named by its section:

- A top-level key that no installed package provides. A misspelt section name is refused like a
  misspelt RoyaleLearn key, and so is one set to `null`.
- Two installed packages that both provide one key.
- A section type that would accept a misspelt field.
- An extension written for another version of this API (`EXTENSION_API_VERSION`, 1 today), or
  one whose `name` is not the key it is declared under.
- A package imported from somewhere other than where it was installed, such as a stale copy
  earlier on the path.
- Two sections that both schedule the actor's learning rate.

While the run starts and runs:

- A loss term whose keys do not start with its extension's name, or whose extension is named after
  a built-in metric group such as `ppo`.
- A checkpoint holding state for a term this run does not have, or at another `format_version`.
- A metric key or alarm name your extension adds that RoyaleLearn already has.

## 5. What you may import

Import from `royalelearn.extensions` only. Its `__all__` is the supported surface: the extension
and loss-term types, the alarm and schema types, schedules, and a few helpers for identity,
environment descriptions and saved actors. Anything else in RoyaleLearn can change without
notice. Importing `royalelearn.extensions` is cheap: it loads torch and the rest of RoyaleLearn
only when you use a name that needs them.

## 6. Testing it

`royalelearn.testing` is what RoyaleLearn's own tests use, and it does not need pytest:

- `tiny_config(folder)` is a run small enough for a unit test, on the mock engine.
- `coordinator(config)` is that run, started.
- `use_extensions(monkeypatch, {...})` makes your extension the provider of its key for one test,
  without installing anything.
- `StubTerm` is a loss term that exercises the plumbing, for comparing against yours.

A pytest test of the example, in the package's own repository. An untrained bot spreads its
probability over hundreds of legal actions, with about 0.001 on the top one, so the test sets a
tiny cap to make the penalty bite:

```python
from royalelearn.extensions import with_sections
from royalelearn.testing import coordinator, tiny_config, use_extensions

import confidence_cap


def test_the_cap_reports_how_often_it_bites(tmp_path, monkeypatch):
    use_extensions(monkeypatch, {"confidence_cap": confidence_cap.EXTENSION})
    config = with_sections(tiny_config(tmp_path), confidence_cap={"cap": 0.0005})
    with coordinator(config) as run:
        run.iterate()
    row = run.rows[-1]
    assert 0.0 < row["confidence_cap/over_cap_frac"] <= 1.0
    assert row["confidence_cap/grad_ratio"] > 0.0
```

The test run records your package's commit, like a real run does, so commit the package once
before the first test.
