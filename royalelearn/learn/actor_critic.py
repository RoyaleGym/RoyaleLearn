"""The assembled policies: an actor, a critic, and the two ways of pairing them.

``SeparateActorCritic`` is the default and gives the policy and the value function a trunk each.
It costs a second forward at rollout and it buys a property that would otherwise have to be
maintained by hand: the entropy bonus and the mask summary reach the critic's parameters through
no path at all, because there is no shared tensor between them. A shared trunk makes that a matter
of remembering to detach something, and forgetting once produces a value function quietly pulled
towards whatever the exploration term wanted.

``SharedTrunkActorCritic`` ships too. It runs the board through one trunk and both heads, which is
roughly half the rollout inference, and it is the answer if that inference turns out to cost more
than a quarter of an iteration. Under it the trunk is listed among the ACTOR's parameters, because
it carries the policy gradient; the critic's are the value head's alone. The value loss still
flows into the shared trunk -- detaching it would leave the trunk trained by the policy gradient
alone, which is the whole of what the shared variant would have been for.

Both hold the same ``ClashActor``, so a pool snapshot written by one loads into the other.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ..api.policy import Actor, ActorCritic, ActResult, BackpropResult, Critic
from .distribution import MaskedCategorical

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch.nn import Parameter

    from ..api.policy import ObsBatch
    from ..config import ArchSpec
    from .nets import ClashTrunk, PointerPolicyHead, ValueHead

__all__ = [
    "BehaviourSnapshot",
    "ClashActor",
    "ClashCritic",
    "SeparateActorCritic",
    "SharedTrunkActorCritic",
    "autocast_context",
]


def autocast_context(device_type: str, dtype: torch.dtype) -> Any:
    """Autocast for this device and dtype, or nothing at all.

    bf16 rather than fp16 wherever a device offers it: it keeps float32's exponent range, so
    filling a masked logit with ``finfo.min`` and taking a ``log_softmax`` behave, and no gradient
    scaler is needed. A device autocast does not know about runs without one rather than raising,
    because the whole test suite runs on CPU in float32 and a network that refuses to be built
    there would be a network nothing could test.
    """
    if dtype is torch.float32 or device_type not in ("cuda", "cpu", "xpu"):
        return contextlib.nullcontext()
    return torch.autocast(device_type, dtype=dtype)


class _Autocasting(nn.Module):
    """A module that runs its forward under the run's autocast dtype.

    The device type is read off the parameters rather than stored, so that a module moved between
    devices autocasts for the one it is actually on.
    """

    def __init__(self, autocast_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.autocast_dtype = autocast_dtype

    def autocast(self) -> Any:
        return autocast_context(self.device_type, self.autocast_dtype)

    @property
    def device_type(self) -> str:
        for parameter in self.parameters():
            return parameter.device.type
        return "cpu"


class ClashActor(_Autocasting, Actor):
    """A trunk and the pointer head. What a pool snapshot is."""

    def __init__(
        self,
        trunk: ClashTrunk,
        head: PointerPolicyHead,
        autocast_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(autocast_dtype)
        self.trunk = trunk
        self.head = head

    def forward(self, obs: ObsBatch) -> Tensor:
        """``(B, n_actions)`` float32 always, even inside autocast: the cast at the end of the
        head is what makes the masking and the ``log_softmax`` downstream exact."""
        with self.autocast():
            return self.head(self.trunk(obs), obs)

    def logits(self, obs: ObsBatch) -> Tensor:
        return self(obs)

    def distribution(self, obs: ObsBatch) -> MaskedCategorical:
        return MaskedCategorical(self.logits(obs), obs.mask)

    def initialise(self, generator: torch.Generator) -> None:
        self.trunk.initialise(generator)
        self.head.initialise(generator)


class ClashCritic(_Autocasting, Critic):
    """A trunk of its own and the value head."""

    def __init__(
        self,
        trunk: ClashTrunk,
        head: ValueHead,
        autocast_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(autocast_dtype)
        self.trunk = trunk
        self.head = head

    def forward(self, obs: ObsBatch) -> Tensor:
        """``(B,)`` float32."""
        with self.autocast():
            return self.head(self.trunk(obs), obs)

    def value(self, obs: ObsBatch) -> Tensor:
        return self(obs)

    def initialise(self, generator: torch.Generator) -> None:
        self.trunk.initialise(generator)
        self.head.initialise(generator)


class _BaseActorCritic(_Autocasting, ActorCritic):
    """What the two pairings share: the actor, the sampling path and the snapshot format."""

    def __init__(
        self,
        actor: ClashActor,
        arch: ArchSpec,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        super().__init__(autocast_dtype)
        self.actor = actor
        self.actor.autocast_dtype = autocast_dtype
        self.arch = arch
        self.device = device
        #: Filled in by the factory that built this pair; compared on every load.
        self.arch_digest = ""

    def logits(self, obs: ObsBatch) -> Tensor:
        return self.actor.logits(obs)

    def distribution(self, obs: ObsBatch) -> MaskedCategorical:
        return self.actor.distribution(obs)

    def act(self, obs: ObsBatch, uniforms: Tensor) -> ActResult:
        """One rollout forward: sample, and the diagnostics that come free with it.

        Under ``no_grad`` because nothing in an ``ActResult`` is differentiated -- the update
        recomputes everything it needs from the buffer, which is the only way the stored
        log-probability can be the log-probability of the action that was taken. The uniforms
        come from the caller so that the action is a function of the run seed, the cycle and the
        slot rather than of how many rows happened to share this forward.
        """
        with torch.no_grad():
            distribution = self.actor.distribution(obs)
            actions = distribution.sample(uniforms)
            return ActResult(
                actions=actions,
                log_probs=distribution.log_prob(actions),
                entropy=distribution.entropy(),
                p_noop=distribution.p_noop(),
                n_legal=distribution.n_legal(),
            )

    def actor_parameters(self) -> Iterator[Parameter]:
        return self.actor.parameters()

    def actor_state_dict_fp16(self) -> dict[str, Tensor]:
        """The actor's weights, fp16, on the CPU, and nothing else.

        Half precision and actor-only because a pool snapshot is played against, never trained
        from: at about a megabyte apiece a hall of fame of a few hundred fits on disk and a
        handful fit resident in memory. The coordinate planes are not in here -- they are a
        non-persistent buffer -- so what a snapshot carries is weights.
        """
        return {
            name: tensor.detach().to(dtype=torch.float16, device="cpu").contiguous()
            for name, tensor in self.actor.state_dict().items()
        }


class SeparateActorCritic(_BaseActorCritic):
    """The default: a trunk each, so the entropy bonus and the critic share no tensor."""

    def __init__(
        self,
        actor: ClashActor,
        critic: ClashCritic,
        arch: ArchSpec,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        super().__init__(actor, arch, device, autocast_dtype)
        self.critic = critic
        self.critic.autocast_dtype = autocast_dtype

    def value(self, obs: ObsBatch) -> Tensor:
        return self.critic.value(obs)

    def backprop(self, obs: ObsBatch, actions: Tensor) -> BackpropResult:
        """Recompute under the current parameters, under exactly the mask that came out of the
        buffer with the transition. A mask recomputed here would be a second opinion about what
        was legal when the action was taken, and the importance ratio would stop meaning
        anything."""
        distribution = self.actor.distribution(obs)
        return BackpropResult(
            log_probs=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            noop_entropy=distribution.noop_entropy(),
            values=self.critic.value(obs),
            n_legal=distribution.n_legal(),
        )

    def critic_parameters(self) -> Iterator[Parameter]:
        return self.critic.parameters()


class SharedTrunkActorCritic(_BaseActorCritic):
    """One trunk, both heads: half the rollout inference, and the entropy bonus reaches the
    trunk.

    ``critic_parameters`` is the value head alone. The trunk carries the policy gradient and is
    listed with the actor, which keeps the two iterators disjoint and keeps "the entropy term's
    gradient into the critic's parameters is zero" true here as well.
    """

    def __init__(
        self,
        actor: ClashActor,
        value_head: ValueHead,
        arch: ArchSpec,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        super().__init__(actor, arch, device, autocast_dtype)
        self.value_head = value_head

    def value(self, obs: ObsBatch) -> Tensor:
        with self.autocast():
            return self.value_head(self.actor.trunk(obs), obs)

    def backprop(self, obs: ObsBatch, actions: Tensor) -> BackpropResult:
        """One trunk forward, both heads off it. That sharing is the point of this variant."""
        with self.autocast():
            features = self.actor.trunk(obs)
            logits = self.actor.head(features, obs)
            values = self.value_head(features, obs)
        distribution = MaskedCategorical(logits, obs.mask)
        return BackpropResult(
            log_probs=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            noop_entropy=distribution.noop_entropy(),
            values=values,
            n_legal=distribution.n_legal(),
        )

    def critic_parameters(self) -> Iterator[Parameter]:
        return self.value_head.parameters()


class BehaviourSnapshot:
    """The actor as it stood at an iteration boundary.

    With overlapped collection the rollout of iteration ``i`` runs while the update of ``i-1`` is
    still changing the weights. Sampling from the live actor there would produce transitions whose
    stored log-probabilities came from several different parameter vectors, none of them
    recoverable. Sampling from a frozen copy makes the lag exactly one iteration by construction,
    so the importance ratio measures the true off-policyness of the batch and the clip bounds it.
    """

    __slots__ = ("actor",)

    def __init__(self, actor: ClashActor) -> None:
        self.actor = copy.deepcopy(actor).eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def of(cls, model: _BaseActorCritic) -> BehaviourSnapshot:
        """A snapshot of whichever actor this pairing holds."""
        return cls(model.actor)

    def logits(self, obs: ObsBatch) -> Tensor:
        with torch.no_grad():
            return self.actor.logits(obs)

    def distribution(self, obs: ObsBatch) -> MaskedCategorical:
        return MaskedCategorical(self.logits(obs), obs.mask)
