"""The actor-loss term hook (``api.update.ActorLossTerm``): what the update adds around a term.

A term an extension hands the update must change the run by exactly what the term computes and
nothing else. So a term at coefficient zero is the run without it, bit for bit, over whole
iterations; an update with no term keeps no state for terms and leaves the state digest as it
always was; and a checkpoint whose kept state this run cannot restore is refused rather than
dropped. Every test here uses ``royalelearn.testing.StubTerm``, the term a package built on
RoyaleLearn can use to hold the same contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from royalelearn.errors import CheckpointFormatError
from royalelearn.testing import StubTerm, coordinator, tiny_config

torch = pytest.importorskip("torch")

from royalelearn.learn.ppo import PPOUpdate  # noqa: E402


def _three(config: Any, **kwargs: Any) -> tuple[list[Any], list[str], list[dict[str, Any]]]:
    """Every learner tensor, the state digest after each iteration, and the rows."""
    with coordinator(config, **kwargs) as run:
        digests = []
        for _ in range(3):
            run.iterate()
            digests.append(run.state_digest())
        tensors = [t.detach().clone() for t in run.model.state_dict().values()]
        for optimizer in run.update.optimizers:
            for state in optimizer.state_dict()["state"].values():
                tensors.extend(v.clone() for v in state.values() if torch.is_tensor(v))
        return tensors, digests, list(run.rows)


def _same(one: list[Any], other: list[Any]) -> bool:
    return len(one) == len(other) and all(
        torch.equal(a, b) for a, b in zip(one, other, strict=True)
    )


def test_an_inert_extension_term_is_the_run_without_it(tmp_path: Path) -> None:
    """At coefficient zero a term runs its whole path -- the loss on every minibatch, the
    gradient-ratio pass, its keys -- and the run's weights, moments and state digests are the
    run without it, bit for bit, over three iterations."""
    plain = _three(tiny_config(tmp_path / "plain"))
    inert = _three(tiny_config(tmp_path / "inert"), extra_actor_terms=(StubTerm(0.0),))
    assert _same(plain[0], inert[0])
    assert plain[1] == inert[1]
    assert all(row["stub/inert/calls"] > 0 for row in inert[2])
    assert any(row.get("stub/inert/grad_ratio", 0.0) > 0.0 for row in inert[2])
    assert not any(key.startswith("stub/") for row in plain[2] for key in row)


def test_the_inert_comparison_can_see_a_millionth(tmp_path: Path) -> None:
    """The same comparison, seen failing: a coefficient of 1e-6 moves the weights."""
    plain = _three(tiny_config(tmp_path / "plain"))
    leaking = _three(tiny_config(tmp_path / "leak"), extra_actor_terms=(StubTerm(1e-6),))
    assert not _same(plain[0], leaking[0])
    assert plain[1] != leaking[1]


class _OwnForward(StubTerm):
    """Shaped like demonstration cloning, the term the protocol has to be able to express: it
    runs the LIVE actor forward itself, with the graph attached, on rows it chose, and its raw
    value is a mean of its own, so the update scales it by the minibatch weight."""

    scaling = "minibatch"

    def loss(self, inputs: Any, *, epoch: int, measure: bool) -> tuple[float, Any]:
        self.calls += 1
        distribution = inputs.actor.distribution(inputs.obs)
        return self.coefficient, -distribution.log_probs[:, 0].mean()


def test_a_term_can_train_the_live_actor_through_a_forward_of_its_own(tmp_path: Path) -> None:
    """At zero it is the run without it, bit for bit, though it ran a whole forward per
    minibatch; above zero its gradient reaches the actor, so the weights move."""
    plain = _three(tiny_config(tmp_path / "plain"))
    silent = _three(tiny_config(tmp_path / "silent"), extra_actor_terms=(_OwnForward(0.0),))
    pulling = _three(tiny_config(tmp_path / "pull"), extra_actor_terms=(_OwnForward(0.5),))
    assert _same(plain[0], silent[0])
    assert plain[1] == silent[1]
    assert not _same(plain[0], pulling[0])
    assert all(row["stub/inert/calls"] > 0 for row in silent[2])


def _update_state(checkpoint: Path) -> dict[str, Any]:
    (path,) = list(checkpoint.rglob(PPOUpdate.STATE_FILE))
    return json.loads(path.read_text(encoding="utf-8"))


def test_an_update_without_terms_keeps_no_state_for_them(tmp_path: Path) -> None:
    """No term and no freeze: no 'extensions' or 'freeze' key, and the digest a run without the
    hook always had. Seen failing: an inert term that keeps state leaves the weights alone and
    moves both."""
    with coordinator(tiny_config(tmp_path / "plain")) as run:
        run.iterate()
        bare = _update_state(run.checkpoint())
        plain_digest = run.state_digest()
        assert run.update.extra_state() == []
    assert set(bare) == {"format_version", "cumulative_model_updates"}

    keeper = StubTerm(0.0, keep_state=True)
    with coordinator(tiny_config(tmp_path / "keeper"), extra_actor_terms=(keeper,)) as run:
        run.iterate()
        kept = _update_state(run.checkpoint())
        kept_digest = run.state_digest()
    assert kept["extensions"] == {
        "stub": {"inert": {"format_version": 1, "state": {"calls": keeper.calls}}}
    }
    assert "freeze" not in kept
    assert kept_digest != plain_digest


def test_kept_state_this_run_cannot_restore_is_refused(tmp_path: Path) -> None:
    """A checkpoint written with a term, resumed without it, would lose the term's state
    silently: an anchor's coefficient back at its start. So would a checkpoint whose state is at
    another format, or one written before the terms had their own entry."""
    config = tiny_config(tmp_path / "run")
    with coordinator(config, extra_actor_terms=(StubTerm(0.0, keep_state=True),)) as run:
        run.iterate()
        saved = run.checkpoint()
        calls = run.actor_terms[-1].calls

    with (
        pytest.raises(CheckpointFormatError, match="extensions/stub/inert"),
        coordinator(config, resume=saved),
    ):
        pass

    newer = StubTerm(0.0, keep_state=True)
    newer.format_version = 2
    with (
        pytest.raises(CheckpointFormatError, match="format 1 and this build reads format 2"),
        coordinator(config, resume=saved, extra_actor_terms=(newer,)),
    ):
        pass

    same = StubTerm(0.0, keep_state=True)
    with coordinator(config, resume=saved, extra_actor_terms=(same,)):
        assert same.calls == calls


def test_the_layout_before_terms_had_entries_is_refused(tmp_path: Path) -> None:
    """Update state as a build before the hook wrote it for a run with the imitation block:
    one 'imitation' key holding the coefficients and the freeze. Read by this build it would
    restore nothing; refused, it names the key. (The checkpoint's manifest hashes would refuse a
    hand-edited file first, so this is the update's own reader, handed the folder directly.)"""
    folder = tmp_path / "update"
    folder.mkdir()
    (folder / PPOUpdate.STATE_FILE).write_text(
        json.dumps(
            {
                "format_version": 1,
                "cumulative_model_updates": 3,
                "imitation": {"lambda": {"bc": 2.0}, "unfrozen_at": None, "last_frozen": False},
            }
        ),
        encoding="utf-8",
    )
    with (
        coordinator(tiny_config(tmp_path / "run")) as run,
        pytest.raises(CheckpointFormatError, match="'imitation'"),
    ):
        run.update.load_checkpoint(folder, strict=False)


class _Trespasser(StubTerm):
    """Reports a key in the core's namespace."""

    def finish(self, **kwargs: Any) -> dict[str, float]:
        return {"ppo/kl": 0.0}


def test_a_term_reports_only_under_its_own_extension(tmp_path: Path) -> None:
    """A term that could write ``ppo/kl`` could overwrite the number an alarm halts on."""
    with (
        coordinator(tiny_config(tmp_path / "run"), extra_actor_terms=(_Trespasser(),)) as run,
        pytest.raises(ValueError, match="outside its extension's namespace"),
    ):
        run.iterate()


def test_terms_the_update_could_not_hold_apart_are_refused_before_anything_is_built(
    tmp_path: Path,
) -> None:
    """Refused where the terms are assembled, ahead of the rollout buffer's shared memory and
    the workers, so nothing costly is built for a run that cannot start."""
    with (
        pytest.raises(ValueError, match="share an"),
        coordinator(tiny_config(tmp_path / "twice"), extra_actor_terms=(StubTerm(), StubTerm())),
    ):
        pass
    odd = StubTerm()
    odd.scaling = "episodes"
    with (
        pytest.raises(ValueError, match="'rows' or 'minibatch'"),
        coordinator(tiny_config(tmp_path / "odd"), extra_actor_terms=(odd,)),
    ):
        pass
