"""Learning from demonstrations: section 19 of ``docs/harness-spec.md``.

Generic machinery only. A demonstration is any timed log of card plays that can be driven
through the run's own environment; where the logs come from, and the code that turns them into
the driver's input, belong to whoever owns them.

What is here:

- ``artifacts``: the folder format a field model is stored in. Actor artifacts, the folder
  digest and the probe-logit self-test are generic and live in ``royalelearn.artifacts``, and
  codec rows in ``royalelearn.learn.rows``.
"""

from __future__ import annotations
