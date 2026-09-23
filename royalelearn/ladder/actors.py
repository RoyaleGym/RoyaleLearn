"""One evaluation seat's policy, for whoever is playing the battle.

A gate plays thousands of battles and every one of them needs the same three things: a scripted
opponent behind a uniform signature, a frozen snapshot loaded from disk, or the live learner. That
resolution used to live on ``LearningCoordinator`` as three methods, which meant a process that
wanted to play evaluation battles had to build a coordinator -- a rollout farm, a buffer, an
optimizer -- or keep a second copy of the code. Neither is acceptable for a worker whose whole job
is to answer ``play``.

So it is a class, built from the four things it actually needs: the environment's spec, the
network architecture, somewhere to load snapshots from, and the release mode. The coordinator hands
it the live model as well; a worker does not have one and says so when asked for the learner.

THE FRAME HISTORY BELONGS TO A POLICY, NOT TO THE THING THAT BUILT IT. Until 2026-09-23 the
stacked-frame history was one list on the coordinator, shared by both seats of a battle and never
reset between battles. With ``obs.frame_stack`` at 1 -- the shipped default and every run so far --
nothing stacked and nothing showed. Above 1 it meant a policy's "previous frame" was usually its
OPPONENT's, because the two seats act alternately through the same object, and the first frames of
a battle were the tail of the one before. Here each policy closure owns its own list, and
``EnvBattlePlayer`` builds the policies once per battle, so the history is per battle and per seat
by construction rather than by remembering to clear it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.rollout import EnvSpec
    from ..config import NetConfig
    from .snapshots import SnapshotStore

__all__ = ["EvalActors", "scripted_policy"]

#: A seat's policy: observation, one uniform in [0, 1), and the generator the caller is drawing
#: from. It returns the action index. The uniform is passed in rather than drawn here so that a
#: battle replays identically wherever it is played.
Policy = Callable[[Mapping[str, Any], float, Any], int]


def scripted_policy(name: str) -> Policy:
    """A scripted opponent behind the battle player's signature."""
    from ..rollout.scripted import build_opponent

    opponent = build_opponent(name)

    def act(obs: Mapping[str, Any], _uniform: float, rng: Any) -> int:
        return int(opponent.act(obs, obs["action_mask"], rng))

    return act


class EvalActors:
    """Resolves a member id to the policy that plays it.

    ``model`` is the LIVE learner and is optional: the probe asks for ``learner@{step}`` because
    the question it answers is what the policy does NOW, and a worker process has no such thing.
    Asking a resolver without one for the learner is a refusal naming the member rather than a
    quietly substituted snapshot, which would answer a different question and look the same.
    """

    def __init__(
        self,
        spec: EnvSpec,
        net: NetConfig,
        snapshots: SnapshotStore,
        *,
        device: Any = "cpu",
        release_mode: str = "stochastic",
        model: Any = None,
    ) -> None:
        self.spec = spec
        self.net = net
        self.snapshots = snapshots
        self.device = device
        self.release_mode = release_mode
        self.model = model

    def __call__(self, member: str) -> Policy:
        """The policy for one member id."""
        from .pool import is_learner

        if member.startswith("scripted:"):
            return scripted_policy(member.split(":", 1)[1])
        if is_learner(member):
            if self.model is None:
                raise LookupError(
                    f"{member!r} is the LIVE learner and this resolver has no model. A snapshot "
                    "would answer a different question: the probe exists because the most recent "
                    "snapshot lags the live weights by the candidate cadence"
                )
            return self.policy(self.model)
        return self.policy(self.snapshots.get(member, self.device))

    def policy(self, actor: Any) -> Policy:
        """A frozen actor behind the battle player's signature.

        One observation at a time and on the device the run uses. Evaluation is a few hundred
        battles at a gate against an update of tens of thousands of rows, so a batch of one is the
        right trade for a path with no rectangle behind it.
        """
        import torch

        from ..learn.distribution import MaskedCategorical

        # This policy's own frames, not the resolver's. See the module docstring.
        history: list[np.ndarray] = []

        def act(obs: Mapping[str, Any], uniform: float, _rng: Any) -> int:
            batch = self.obs_batch(obs, history)
            with torch.inference_mode():
                distribution = MaskedCategorical(actor.logits(batch).float(), batch.mask)
                if self.release_mode == "argmax":
                    return int(distribution.mode()[0].item())
                draw = torch.tensor([uniform], dtype=torch.float32, device=batch.mask.device)
                return int(distribution.sample(draw)[0].item())

        return act

    def obs_batch(self, obs: Mapping[str, Any], history: list[np.ndarray] | None = None) -> Any:
        """One environment observation as the batch of one the networks take.

        Frames are stacked the way the rectangle stacks them -- the current frame first -- and an
        evaluation battle carries no history, so the older frames are zero on the first decision
        and this policy's previous observations after it. Built here rather than through the codec
        because nothing is being stored: quantising an observation in order to dequantise it again
        would be a round trip for its own sake.

        ``history`` is the caller's list and is mutated in place. Passing None is a single frame's
        worth of context and is what a caller with no stacking wants.
        """
        import torch

        from ..api.policy import ObsBatch

        spatial = np.asarray(obs["spatial"], dtype=np.float32)
        mask = np.asarray(obs["action_mask"]).astype(bool)
        planes = mask[1:].reshape(self.spec.hand_size, *self.spec.tiles).astype(np.float32)
        frames = self.spec.frame_stack
        if frames > 1 and history is not None:
            if not history or history[0].shape != spatial.shape:
                history[:] = [np.zeros_like(spatial) for _ in range(frames - 1)]
            spatial_stack = np.concatenate([spatial, *history], axis=0)
            plane_stack = np.concatenate(
                [planes, *[np.zeros_like(planes) for _ in history]], axis=0
            )
            history.insert(0, spatial)
            del history[frames - 1 :]
        else:
            spatial_stack, plane_stack = spatial, planes

        def tensor(array: np.ndarray, dtype: Any) -> Any:
            return torch.from_numpy(np.ascontiguousarray(array)).to(
                device=self.device, dtype=dtype
            )[None]

        return ObsBatch(
            spatial=tensor(spatial_stack, torch.float32),
            mask_planes=tensor(plane_stack, torch.float32),
            vector=tensor(np.asarray(obs["vector"], dtype=np.float32), torch.float32),
            mask=tensor(mask, torch.bool),
        )
