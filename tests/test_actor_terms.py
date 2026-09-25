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

import msgspec
import pytest

from royalelearn.errors import CheckpointFormatError, PreflightError
from royalelearn.extensions import with_sections
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


class _Recording(StubTerm):
    """A stub that remembers which epochs it ran in and the minibatch weights it was handed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.epochs: set[int] = set()
        self.measured: set[tuple[int, bool]] = set()
        self.weights: dict[tuple[int, int], float] = {}
        self.iteration = 0

    def begin(self, env_steps: int, device: Any) -> None:
        self.iteration += 1

    def loss(self, inputs: Any, *, epoch: int, measure: bool) -> tuple[float, Any]:
        self.epochs.add(epoch)
        self.measured.add((epoch, measure))
        key = (self.iteration, epoch)
        self.weights[key] = self.weights.get(key, 0.0) + float(inputs.weight)
        return super().loss(inputs, epoch=epoch, measure=measure)


def test_an_inert_extension_term_is_the_run_without_it(tmp_path: Path) -> None:
    """At coefficient zero a term runs its whole path -- the loss in every epoch, the
    gradient-ratio pass, its keys -- and the run's weights, moments and state digests are the
    run without it, bit for bit, over three iterations."""
    plain = _three(tiny_config(tmp_path / "plain"))
    term = _Recording(0.0)
    inert = _three(tiny_config(tmp_path / "inert"), extra_actor_terms=(term,))
    assert _same(plain[0], inert[0])
    assert plain[1] == inert[1]
    assert term.epochs == {0, 1}, "the term did not run in every epoch"
    assert term.measured == {(0, True), (1, False)}, "a term measures in the first epoch only"
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


def _digest_without_the_hook(run: Any) -> str:
    """The state digest by the formula it had before terms existed, from the run's own parts."""
    import hashlib

    from royalelearn.coordinator import _hash_tensors

    digest = hashlib.sha256()
    _hash_tensors(digest, run.model.state_dict())
    for optimizer in run.update.optimizers:
        state = optimizer.state_dict()["state"]
        for key in sorted(state):
            digest.update(str(key).encode("utf-8"))
            _hash_tensors(digest, state[key])
    digest.update(msgspec.json.encode(run.gae.scaler.state()))
    digest.update(msgspec.json.encode(run.schedules.backoff.state()))
    digest.update(str(run.update.model_updates).encode("utf-8"))
    return digest.hexdigest()


def _update_state(checkpoint: Path) -> dict[str, Any]:
    (path,) = list(checkpoint.rglob(PPOUpdate.STATE_FILE))
    return json.loads(path.read_text(encoding="utf-8"))


def test_an_update_without_terms_keeps_no_state_for_them(tmp_path: Path) -> None:
    """No term and no freeze: no 'extensions' or 'freeze' key, the update format of a build that
    had no terms at all, and the same state digest as a run whose term keeps nothing. Seen
    failing: an inert term that keeps state leaves the weights alone and moves both."""
    with coordinator(tiny_config(tmp_path / "plain")) as run:
        run.iterate()
        bare = _update_state(run.checkpoint())
        plain_digest = run.state_digest()
        assert run.update.extra_state() == []
        assert plain_digest == _digest_without_the_hook(run)
    assert bare["format_version"] == 1
    assert set(bare) == {"format_version", "cumulative_model_updates"}
    with coordinator(tiny_config(tmp_path / "quiet"), extra_actor_terms=(StubTerm(0.0),)) as run:
        run.iterate()
        assert set(_update_state(run.checkpoint())) == set(bare)
        assert run.state_digest() == plain_digest

    keeper = StubTerm(0.0, keep_state=True)
    with coordinator(tiny_config(tmp_path / "keeper"), extra_actor_terms=(keeper,)) as run:
        run.iterate()
        kept = _update_state(run.checkpoint())
        kept_digest = run.state_digest()
    assert kept["extensions"] == {
        "stub": {"inert": {"format_version": 1, "state": {"calls": keeper.calls}}}
    }
    assert kept["format_version"] == 2
    assert "freeze" not in kept
    assert kept_digest != plain_digest


def _component_versions(checkpoint: Path) -> dict[str, int]:
    return json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))[
        "component_versions"
    ]


def test_the_manifest_records_the_format_the_update_wrote(tmp_path: Path) -> None:
    """The checkpoint store records each component's format beside its files; for the update it
    is the format it wrote, so a run without terms records what it always did. Seen failing: the
    store reading a class constant raised to 2 for every run."""
    with coordinator(tiny_config(tmp_path / "plain")) as run:
        run.iterate()
        saved = run.checkpoint()
    assert _component_versions(saved)["optimizers"] == 1
    assert _update_state(saved)["format_version"] == 1
    with coordinator(
        tiny_config(tmp_path / "keeper"), extra_actor_terms=(StubTerm(0.0, keep_state=True),)
    ) as run:
        run.iterate()
        saved = run.checkpoint()
    assert _component_versions(saved)["optimizers"] == 2
    assert _update_state(saved)["format_version"] == 2


def test_a_build_that_knows_no_terms_refuses_their_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written as format 2 exactly when a term's state or the freeze is kept, so a build that
    reads only format 1 refuses it instead of resuming without the anchor's coefficient."""
    with coordinator(
        tiny_config(tmp_path / "keeper"), extra_actor_terms=(StubTerm(0.0, keep_state=True),)
    ) as run:
        run.iterate()
        saved = run.checkpoint()
    (state,) = list(saved.rglob(PPOUpdate.STATE_FILE))
    monkeypatch.setattr(PPOUpdate, "READS_FORMAT", 1)
    with (
        coordinator(tiny_config(tmp_path / "old")) as run,
        pytest.raises(CheckpointFormatError, match="update format 2"),
    ):
        run.update.load_checkpoint(state.parent, strict=False)


def _freeze_state(folder: Path, freeze: Any) -> Path:
    folder.mkdir()
    (folder / PPOUpdate.STATE_FILE).write_text(
        json.dumps({"format_version": 2, "cumulative_model_updates": 1, "freeze": freeze}),
        encoding="utf-8",
    )
    return folder


def test_a_freeze_state_is_refused_without_a_freeze_or_at_another_format(tmp_path: Path) -> None:
    """The freeze half of the refusals: a run with no schedule has nothing to restore it into,
    and one at another format would be read as this one."""
    state = {"unfrozen_at": 3, "last_frozen": False}
    stray = _freeze_state(tmp_path / "stray", {"format_version": 1, "state": state})
    with (
        coordinator(tiny_config(tmp_path / "plain")) as run,
        pytest.raises(CheckpointFormatError, match="'freeze'"),
    ):
        run.update.load_checkpoint(stray, strict=False)
    future = _freeze_state(tmp_path / "future", {"format_version": 9, "state": state})
    scheduled = with_sections(
        tiny_config(tmp_path / "frozen"),
        warm_start={"actor_lr_scale": {"kind": "constant", "value": 1.0}},
    )
    with (
        coordinator(scheduled) as run,
        pytest.raises(CheckpointFormatError, match="format 9 and this build reads format 1"),
    ):
        run.update.load_checkpoint(future, strict=False)


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
    """A term that could write ``ppo/kl`` could overwrite the number an alarm halts on: by
    reporting outside its namespace, or by taking a core group as its extension's name."""
    with (
        coordinator(tiny_config(tmp_path / "run"), extra_actor_terms=(_Trespasser(),)) as run,
        pytest.raises(ValueError, match="outside its extension's namespace"),
    ):
        run.iterate()
    for name in ("ppo", "run", "a/b", ""):
        with (
            pytest.raises(ValueError, match="cannot be an extension's name"),
            coordinator(
                tiny_config(tmp_path / f"core-{len(name)}{name.replace('/', '-')}"),
                extra_actor_terms=(StubTerm(extension=name),),
            ),
        ):
            pass


class _Remembering(StubTerm):
    """Keeps the raw value and the scale of its first call, with the graph attached."""

    first: tuple[Any, float] | None = None

    def loss(self, inputs: Any, *, epoch: int, measure: bool) -> tuple[float, Any]:
        coefficient, raw = super().loss(inputs, epoch=epoch, measure=measure)
        if self.first is None:
            self.first = (raw, float(inputs.actor_scale))
        return coefficient, raw


def test_the_gradient_ratio_is_the_raw_value_at_the_policy_terms_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The definition, recomputed: ||grad(raw * actor_scale)|| / ||grad(policy term)|| on the
    first minibatch with a choice row, the coefficient (2.5 here) left out.

    Plants: measure the raw value without its scale, or the contribution with the coefficient
    in it, and the value the update differentiates is no longer ``raw * actor_scale``."""
    term = _Remembering(2.5)
    seen: dict[str, Any] = {}
    original = PPOUpdate._grad_ratio

    def spy(self: Any, policy_loss: Any, scaled: Any) -> None:
        ((_, value),) = scaled
        raw, scale = term.first  # type: ignore[misc]
        params = [p for p in self.actor_params if p.requires_grad]

        def norm(tensor: Any) -> float:
            grads = torch.autograd.grad(tensor, params, retain_graph=True, allow_unused=True)
            return float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)))

        seen["same_value"] = torch.equal(value.detach(), (raw * scale).detach())
        seen["expected"] = norm(raw * scale) / max(norm(policy_loss), 1e-30)
        original(self, policy_loss, scaled)

    monkeypatch.setattr(PPOUpdate, "_grad_ratio", spy)
    with coordinator(tiny_config(tmp_path / "run"), extra_actor_terms=(term,)) as run:
        run.iterate()
        reported = run.rows[-1]["stub/inert/grad_ratio"]
    assert seen["same_value"], "the ratio was not measured on raw * actor_scale"
    assert reported == pytest.approx(seen["expected"], rel=1e-5)
    assert reported > 0.0


class _Detached(StubTerm):
    """A term with nothing to add: a detached zero, as demo_bc with no demonstration rows."""

    def loss(self, inputs: Any, *, epoch: int, measure: bool) -> tuple[float, Any]:
        self.calls += 1
        return self.coefficient, torch.zeros(())


def test_a_term_without_a_gradient_reads_a_ratio_of_zero(tmp_path: Path) -> None:
    with coordinator(tiny_config(tmp_path / "run"), extra_actor_terms=(_Detached(1.0),)) as run:
        run.iterate()
        assert run.rows[-1]["stub/inert/grad_ratio"] == 0.0


class _MinibatchMean(_Recording):
    scaling = "minibatch"


@pytest.mark.parametrize("arm", ["all", "critic_only_choice_mean"])
def test_a_minibatch_term_runs_on_every_minibatch(tmp_path: Path, arm: str) -> None:
    """Its raw value is a mean of its own, weighted by the minibatch's share of the batch, so
    the weights it is handed must sum to the batch count in every epoch whatever the cut. At a
    minibatch of one row most minibatches hold no choice row. Seen failing: run only where a
    choice row is, and the sums fall short."""
    base = tiny_config(tmp_path / arm)
    config = msgspec.structs.replace(
        base, ppo=msgspec.structs.replace(base.ppo, minibatch_size=1, forced_rows=arm)
    )
    term = _MinibatchMean(0.0)
    with coordinator(config, extra_actor_terms=(term,)) as run:
        run.iterate()
        batches = config.ppo.timesteps_per_iteration // config.ppo.batch_size
    assert term.weights, "the term never ran"
    for key, total in term.weights.items():
        assert total == pytest.approx(float(batches)), (key, total)


def test_terms_the_update_could_not_hold_apart_are_refused_before_anything_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused where the terms are assembled, ahead of the rollout buffer's shared memory and
    the workers, so nothing costly is built for a run that cannot start. Seen failing: without
    the early check, the update refuses them after the buffer exists."""

    def built(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the rollout buffer was built for a run that cannot start")

    monkeypatch.setattr("royalelearn.learn.buffer.RectBuffer", built)
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


def test_a_section_s_term_must_report_under_the_section_s_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extension's terms are its own: one reporting under another name would put its keys
    and its checkpoint state where another extension's belong."""
    from royalelearn.testing import StubExtension, use_extensions

    class _Lender(StubExtension):
        def actor_terms(self, section: Any, ctx: Any) -> tuple[StubTerm, ...]:
            return (StubTerm(0.0, extension="imitation"),)

    use_extensions(monkeypatch, {"lender": _Lender("lender")})
    config = with_sections(tiny_config(tmp_path / "run"), lender={})
    with (
        pytest.raises(PreflightError, match="lender: imitation/inert"),
        coordinator(config),
    ):
        pass
