"""Learning from demonstrations: section 19 of ``docs/harness-spec.md``.

Generic machinery only. A demonstration is any timed log of card plays that can be driven
through the run's own environment; where the logs come from, and the code that turns them into
the driver's input, belong to whoever owns them.

What is here:

- ``artifacts``: the folder formats a cloned actor and a field model are stored in, the digest a
  config names them by, and the probe-logit self-test.
- ``rows``: codec rows packed the way the rollout packs them and decoded the way the update
  decodes them.
"""

from __future__ import annotations
