"""Reference policies: fixed policies a run is compared with, never trained (section 19.6).

Two kinds. A ``snapshot`` is an actor artifact on a second copy of the run's own actor, so it has
an opinion about every legal action. A ``field_mlp`` is a small model over named observation
fields that gives the logit of p(play) on a row, and has an opinion about nothing else: it can
anchor the play/wait marginal and cannot say which card or which tile.

Both are evaluated under ``no_grad`` on the rows the actor is trained on, with the observation
the actor sees -- the same decoded batch -- so a difference between a reference and the policy
is never a difference between two views of the state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ..artifacts import (
    check_actor_artifact,
    load_actor_state,
    read_actor_artifact,
    self_test,
)
from ..errors import PreflightError
from ..obs_layout import field_slice
from .artifacts import build_field_mlp, read_field_model

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor

    from ..api.policy import ObsBatch
    from ..api.rollout import EnvSpec
    from ..ladder.snapshots import SnapshotSpec
    from ..learn.rows import RowCodec
    from .config import ImitationSection

__all__ = ["FieldMLPReference", "SnapshotReference", "build_references"]


class SnapshotReference:
    """An actor artifact on a frozen copy of the run's actor: a full policy over the legal set."""

    kind = "snapshot"

    def __init__(self, name: str, actor: Any) -> None:
        self.name = name
        self.actor = actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    def log_probs(self, obs: ObsBatch) -> Tensor:
        """``(B, n_actions)`` float32 masked log-probabilities over each row's own mask."""
        import torch

        with torch.no_grad():
            return self.actor.distribution(obs).log_probs

    def noop_log_probs(self, obs: ObsBatch) -> tuple[Tensor, Tensor]:
        """``(log p(no-op), log p(play))``, each ``(B,)``."""
        from royalegym.action import NOOP

        from .regularisers import log1mexp

        log_noop = self.log_probs(obs)[:, NOOP]
        return log_noop, log1mexp(log_noop)


class FieldMLPReference:
    """p(play) on a row, from named fields of the observation vector."""

    kind = "field_mlp"

    def __init__(
        self,
        name: str,
        module: Any,
        slices: list[slice],
        mean: Tensor,
        std: Tensor,
    ) -> None:
        self.name = name
        self.module = module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.slices = slices
        self.mean = mean
        self.std = std

    def play_logit(self, obs: ObsBatch) -> Tensor:
        """``(B,)`` float32: the logit of p(play)."""
        import torch

        with torch.no_grad():
            inputs = torch.cat([obs.vector[:, part].float() for part in self.slices], dim=-1)
            return self.module((inputs - self.mean) / self.std).squeeze(-1).float()

    def noop_log_probs(self, obs: ObsBatch) -> tuple[Tensor, Tensor]:
        """``(log p(no-op), log p(play))``, each ``(B,)``, computed in log space."""
        import torch.nn.functional as F

        logit = self.play_logit(obs)
        return F.logsigmoid(-logit), F.logsigmoid(logit)


Reference = SnapshotReference | FieldMLPReference


def _field_mlp(name: str, folder: Any, spec: EnvSpec, device: Any) -> FieldMLPReference:
    import torch

    what = f"imitation.references.{name}"
    model_spec, state = read_field_model(folder)
    if model_spec.target != "play":
        raise PreflightError(f"{what}: a field model of target {model_spec.target!r}; only 'play'")
    slices: list[slice] = []
    for field, width in model_spec.fields:
        part = field_slice(spec, field)
        if part.stop - part.start != width:
            raise PreflightError(
                f"{what}: field {field!r} is {part.stop - part.start} wide in this run's "
                f"observation and the model was fitted on {width}"
            )
        slices.append(part)
    total = sum(width for _, width in model_spec.fields)
    if len(model_spec.mean) != total or len(model_spec.std) != total:
        raise PreflightError(
            f"{what}: the normalisation has {len(model_spec.mean)} means and "
            f"{len(model_spec.std)} spreads for {total} inputs"
        )
    if any(not value > 0.0 for value in model_spec.std):
        raise PreflightError(f"{what}: a normalisation spread is not positive")
    module = build_field_mlp(model_spec)
    module.load_state_dict(state, strict=True)
    module.to(device)
    return FieldMLPReference(
        name,
        module,
        slices,
        torch.tensor(model_spec.mean, dtype=torch.float32, device=device),
        torch.tensor(model_spec.std, dtype=torch.float32, device=device),
    )


def build_references(
    imitation: ImitationSection | None,
    *,
    spec: EnvSpec,
    current: SnapshotSpec,
    build_actor: Callable[[Any], Any],
    codec: RowCodec,
    device: Any,
    printer: Callable[[str], None] | None = print,
) -> Mapping[str, Reference]:
    """Every reference the block declares, loaded and checked, by name.

    A snapshot reference is checked the way an init is: its spec against the run's, a strict
    load, and -- when it carries probe rows -- the self-test on them, because a reference that
    does not compute the function it was saved as would anchor the run to something nobody
    trained.
    """
    from .config import FieldMLPReferenceSpec, SnapshotReferenceSpec

    if imitation is None:
        return {}
    out: dict[str, Reference] = {}
    for name, reference in sorted(imitation.references.items()):
        what = f"imitation.references.{name}"
        if isinstance(reference, SnapshotReferenceSpec):
            artifact = read_actor_artifact(reference.path)
            check_actor_artifact(artifact, current, what=what)
            actor = build_actor(device)
            load_actor_state(actor, artifact, what=what)
            loaded = SnapshotReference(name, actor)
            said = "no probe rows"
            if artifact.probe is not None:
                worst = self_test(loaded.actor, codec, artifact.probe, atol=1e-5, what=what)
                said = f"probe self-test {worst:.3g}"
            out[name] = loaded
        elif isinstance(reference, FieldMLPReferenceSpec):
            out[name] = _field_mlp(name, reference.path, spec, device)
            said = "field model"
        else:  # pragma: no cover - the config's tagged union has no third member
            raise PreflightError(f"{what}: unknown reference kind {type(reference).__name__}")
        if printer is not None:
            printer(f"imitation     reference {name}: {reference.path} ({said})")
    return out
