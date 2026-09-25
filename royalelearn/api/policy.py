"""The policy side: what an action distribution is, what an actor-critic must do, and how one
is built from an environment description.

Every ABC here is a plain ``ABC``. The shipped implementations are also ``torch.nn.Module``,
and inheriting from it here would put torch on the import path of ``royalelearn.api``, which is
the one thing this subpackage promises not to do: ``import royalelearn`` and
``royalelearn --help`` work in an environment with no torch installed, and the rollout workers
keep that property at run time. Tensor types appear in annotations only, behind
``TYPE_CHECKING``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from torch import Tensor
    from torch.nn import Parameter

    from ..config import ArchSpec
    from .rollout import EnvSpec

__all__ = [
    "ActResult",
    "ActionDistribution",
    "Actor",
    "ActorCritic",
    "BackpropResult",
    "Critic",
    "NetworkFactory",
    "ObsBatch",
]


class ObsBatch(NamedTuple):
    """One batch of observations on the device, already dequantised.

    Shapes follow from ``EnvSpec`` and the frame stack ``k``: ``spatial`` is
    ``(B, k*S, H, W)``, ``mask_planes`` is ``(B, k*A_s, H, W)`` for the ``A_s`` per-slot action
    planes, ``vector`` is ``(B, V)`` and is the CURRENT frame's only, and ``mask`` is
    ``(B, n_actions)`` bool.
    """

    spatial: Tensor
    mask_planes: Tensor
    vector: Tensor
    mask: Tensor
    #: ``(B, k*2, H, W)`` int64 card ids per tile, own then enemy, per frame: 0 empty, 1 crown
    #: tower, 2 + card id. None unless the observation builder was asked for card identity.
    card_ids: Tensor | None = None


class ActionDistribution(ABC):
    """A categorical distribution over the action space, with the illegal actions removed."""

    @abstractmethod
    def sample(self, uniforms: Tensor) -> Tensor:
        """``(B,)`` int64. Driven by CALLER-SUPPLIED uniforms in [0, 1) so that the sampled
        action is reproducible independently of batch composition and of torch's global RNG."""

    @abstractmethod
    def mode(self) -> Tensor:
        """``(B,)`` int64, mask-respecting argmax."""

    @abstractmethod
    def log_prob(self, actions: Tensor) -> Tensor:
        """``(B,)`` float32."""

    @abstractmethod
    def entropy(self) -> Tensor:
        """``(B,)`` float32, over the whole legal set."""

    @abstractmethod
    def noop_entropy(self) -> Tensor:
        """``(B,)`` float32: the binary entropy of p(no-op) against p(play).

        The leading indicator of no-op collapse, and bounded by 0.693 nats, which is why the
        coefficient that acts on it is not the joint entropy's.
        """

    @abstractmethod
    def n_legal(self) -> Tensor:
        """``(B,)`` int64. Diagnostics: raw entropy falling is ambiguous without it."""


class Actor(ABC):
    """The policy network. Implementations are also ``torch.nn.Module``."""

    @abstractmethod
    def logits(self, obs: ObsBatch) -> Tensor:
        """``(B, spec.n_actions)`` float32 ALWAYS, even inside autocast. Raw logits: no softmax,
        no clamp, no mask."""

    def distribution(self, obs: ObsBatch) -> ActionDistribution:
        """The default masked categorical over this actor's logits.

        The import is here rather than at module scope because the shipped distribution is a
        torch module and this file must import without torch.
        """
        from ..learn.distribution import MaskedCategorical

        return MaskedCategorical(self.logits(obs), obs.mask)


class Critic(ABC):
    """The value network. Implementations are also ``torch.nn.Module``."""

    @abstractmethod
    def value(self, obs: ObsBatch) -> Tensor:
        """``(B,)`` float32."""


class ActorCritic(ABC):
    """The pair, as the learner uses them.

    Implementers may assume obs tensors are on the device and already dequantised, and that
    ``obs.mask[:, 0]`` is True on every row (RoyaleGym sets ``mask[NOOP]=1`` unconditionally,
    ``action.py``, including after game over). They MUST assert it.
    """

    arch: ArchSpec

    @abstractmethod
    def act(self, obs: ObsBatch, uniforms: Tensor) -> ActResult: ...

    @abstractmethod
    def value(self, obs: ObsBatch) -> Tensor: ...

    @abstractmethod
    def backprop(self, obs: ObsBatch, actions: Tensor) -> BackpropResult:
        """Recompute under the current parameters, applying exactly the mask read from the
        buffer -- never a recomputed one."""

    @abstractmethod
    def actor_parameters(self) -> Iterator[Parameter]: ...

    @abstractmethod
    def critic_parameters(self) -> Iterator[Parameter]: ...

    @abstractmethod
    def actor_state_dict_fp16(self) -> dict[str, Tensor]:
        """What a pool snapshot stores: actor only, fp16, no optimizer."""


class ActResult(msgspec.Struct):
    """What one rollout forward produces. Everything after the first two is diagnostics."""

    actions: Tensor
    log_probs: Tensor
    entropy: Tensor
    p_noop: Tensor
    n_legal: Tensor


class BackpropResult(msgspec.Struct):
    """What one update forward produces, recomputed under the current parameters."""

    log_probs: Tensor
    entropy: Tensor
    noop_entropy: Tensor
    values: Tensor
    n_legal: Tensor
    #: Spread of the logits over each row's legal set. Entropy is a saturating function of this
    #: and reads within 1e-5 of its maximum for the first hundred iterations of a run; this does
    #: not, so it is the one a plot can show.
    logit_std: Tensor
    #: The masked distribution itself, over every row, for a term that needs more of it than
    #: the taken action's log-probability -- the reference KL of section 19.7. None from an
    #: implementation that does not say.
    distribution: Any = None


class NetworkFactory(ABC):
    """How an ``ActorCritic`` is built, and how two of them are told apart."""

    @abstractmethod
    def build(self, spec: EnvSpec, arch: ArchSpec, device: Any, dtype: Any) -> ActorCritic: ...

    @abstractmethod
    def arch_digest(self, spec: EnvSpec, arch: ArchSpec) -> str:
        """sha256 of the canonical JSON of (arch, obs_space, frame_stack, num_cards, n_actions).

        A snapshot or checkpoint built from a different architecture is refused by this digest,
        not shape-errored halfway through a load.
        """
