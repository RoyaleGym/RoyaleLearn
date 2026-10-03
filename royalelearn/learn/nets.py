"""The trunk, the two heads, and the factory that builds a pair of them.

Not one width in this file is written down. Every channel count, plane count, field offset and
output width is computed from the ``EnvSpec`` the environment reported at preflight, so the same
code builds a network against MockEngine's sixteen-card catalogue and against the full one, and a
plane added upstream widens the stem without anything here being edited.

Three choices are load-bearing and each has a failure it avoids:

*The mask planes are trunk input.* Each is one hand slot's deploy legality over the whole board --
"where could I place this card" -- which is a state feature the policy would otherwise have to
infer from elixir, the arena and the enemy's towers. It is already computed for the action mask,
so it costs input channels and nothing else.

*The policy head is a pointer, not a flat projection.* ``GridActionParser.encode`` lays the
non-no-op actions out as ``1 + slot*ny*nx + y*nx + x``, so they ARE a ``(slots, y, x)`` C-order
tensor aligned pixel for pixel with the spatial planes. A flat linear layer over that many outputs
is three orders of magnitude more parameters than the inner product here, and it is the structural
cause of tile spam: a shared-weight head cannot memorise one output unit the way a flat head can.
The query is conditioned on a CARD EMBEDDING rather than on the slot index because slot 0 holds a
different card every cycle -- a slot-indexed head has to learn a card-agnostic detector -- and
because an embedding is what carries anything learnt here across a change of deck.

*GroupNorm, never BatchNorm.* BatchNorm computes a different function at rollout (a small batch,
running statistics) than at update (a large batch, training mode). That is not a rounding
difference; it would mean the log-probability stored with a transition is not the log-probability
of the action that was taken, and the whole importance ratio rests on that identity.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
import torch
from torch import Tensor, nn

from royalegym.action import NOOP

from ..api.policy import NetworkFactory
from ..errors import PreflightError
from ..obs_layout import hand_fields, id_planes
from ..rollout.envspec import digest_of
from ..seeding import TORCH_INIT, derive_int
from .actor_critic import ClashActor, ClashCritic, SeparateActorCritic, SharedTrunkActorCritic

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..api.policy import ActorCritic, ObsBatch
    from ..api.rollout import EnvSpec
    from ..config import ArchSpec

__all__ = [
    "POLICY_HEADS",
    "ClashTrunk",
    "DefaultNetworkFactory",
    "FactoredPolicyHead",
    "FactoredStages",
    "PointerPolicyHead",
    "ResBlock",
    "ValueHead",
    "build_policy_head",
    "head_meta",
    "resolve_dtype",
]

#: The gain the initialisation gives a hidden layer.
HIDDEN_GAIN: float = math.sqrt(2.0)
#: The gain on the policy head's output layers. Near zero, so the opening policy is near uniform
#: over the legal set rather than confidently wrong about it.
HEAD_GAIN: float = 0.01
#: The gain on the value head's output layer.
VALUE_GAIN: float = 1.0
#: The standard deviation of the card table at initialisation.
EMBED_STD: float = 0.02
#: Width of the embedding each tile's card id is given before the stem, per side and per frame.
#: A constant rather than a config field on purpose: a new ``NetConfig`` field would change the
#: architecture digest of every run, and so refuse every checkpoint and snapshot on disk, for a
#: feature most of them never switched on. It enters the digest only when ``card_ids`` exists.
CARD_ID_EMBED: int = 8

#: What ``NetConfig.policy_head`` may name.
POLICY_HEADS: tuple[str, ...] = ("pointer", "factored")
#: The fill the factored head masks its stages with before each ``log_softmax``. Finite, so that
#: a stage's log-probability plus another's never overflows to -inf on the way to the flat vector,
#: and far enough below any logit that its probability is exactly zero.
_STAGE_FILL: float = -1e9

#: What ``NetConfig.autocast_dtype`` may name. float32 means no autocast at all.
DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(dtype: Any) -> torch.dtype:
    """A ``torch.dtype`` from a dtype, a name, or None.

    The architecture carries its autocast dtype as a string because the config is JSON; this
    takes either, so a caller that already holds the dtype does not have to spell it back out as
    a name.
    """
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return DTYPES[str(dtype)]
    except KeyError:
        raise PreflightError(
            f"autocast dtype {dtype!r} is not one of {', '.join(sorted(DTYPES))}"
        ) from None


def _orthogonal(weight: Tensor, gain: float, generator: torch.Generator) -> None:
    """Orthogonal initialisation of one weight, drawn from a named generator.

    Passing the generator explicitly is what keeps the initial weights a function of the master
    seed alone rather than of whatever else has drawn from a global one first.
    """
    nn.init.orthogonal_(weight, gain=gain, generator=generator)


def _zero_bias(module: nn.Conv2d | nn.Linear) -> None:
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _coord_planes(tiles: tuple[int, int]) -> Tensor:
    """``(2, H, W)``: normalised y and x, in [0, 1].

    The arena is not translation-invariant. Own half and enemy half, the river, the two bridges
    and the tower rectangles are all absolute positions, so a convolution that cannot tell where
    it is has to spend capacity rediscovering the board at every position.
    """
    height, width = tiles
    ys = torch.arange(height, dtype=torch.float32) / max(height - 1, 1)
    xs = torch.arange(width, dtype=torch.float32) / max(width - 1, 1)
    return torch.stack(
        [ys[:, None].expand(height, width), xs[None, :].expand(height, width)], dim=0
    )


def _pool(features: Tensor) -> Tensor:
    """``(B, 2C)``: the mean and the maximum of every channel over the board.

    Both, because they answer different questions -- how much of a thing is on the board, and
    whether any of it is anywhere -- and a value that turns on "is the enemy king tower exposed"
    needs the second one.
    """
    return torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=-1)


def _logit_scale(arch: ArchSpec) -> float:
    """The constant the inner product is multiplied by.

    ``rsqrt_c`` keeps the tile logits at unit scale whatever ``net.channels`` is, so the same
    head gain and the same entropy coefficient mean the same thing on a wider network.
    """
    if arch.logit_scale == "rsqrt_c":
        return float(arch.channels) ** -0.5
    if arch.logit_scale in ("none", "one"):
        return 1.0
    raise PreflightError(f"net.logit_scale {arch.logit_scale!r} is not one of 'rsqrt_c', 'none'")


def _check_arch(spec: EnvSpec, arch: ArchSpec) -> None:
    """Everything about an architecture that can be refused before a tensor is allocated."""
    if arch.channels % arch.norm_groups:
        raise PreflightError(
            f"net.norm_groups {arch.norm_groups} does not divide net.channels {arch.channels}"
        )
    if arch.card_embed != arch.channels:
        raise PreflightError(
            f"net.card_embed {arch.card_embed} must equal net.channels {arch.channels}: the card "
            "embedding is one side of the pointer head's inner product against the policy "
            "feature map"
        )
    if arch.blocks < 0:
        raise PreflightError(f"net.blocks is {arch.blocks}")
    tiles_y, tiles_x = spec.tiles
    expected = spec.n_grid_actions + spec.n_buttons
    if expected != spec.n_actions:
        raise PreflightError(
            f"the action space is {spec.n_actions} wide, and a no-op plus {spec.hand_size} hand "
            f"slots over {tiles_y}x{tiles_x} tiles plus {spec.n_buttons} ability buttons is "
            f"{expected}. The pointer head writes the grid and then one logit per button"
        )
    _logit_scale(arch)
    if arch.policy_head not in POLICY_HEADS:
        raise PreflightError(
            f"net.policy_head {arch.policy_head!r} is not one of {', '.join(POLICY_HEADS)}"
        )
    if arch.policy_head == "factored":
        if arch.noop_bias != 0.0:
            raise PreflightError(
                f"net.noop_bias is {arch.noop_bias}, and the factored head has no no-op logit to "
                "add it to: its gate starts at net.factored_act_init instead"
            )
        if not 0.0 < arch.factored_act_init < 1.0:
            raise PreflightError(
                f"net.factored_act_init {arch.factored_act_init} is not a probability strictly "
                "between 0 and 1"
            )


def _arch_for_digest(arch: ArchSpec) -> Any:
    """The architecture as ``arch_digest`` reads it: every field but ``factored_act_init``.

    That one is where the factored head's gate starts, not a shape or a function of the weights,
    so a clone made with another value still loads. Leaving it out also keeps the digest of every
    network built before the field existed exactly what it was. The struct keeps the config's own
    field order, which is the order the canonical encoding writes a struct in.
    """
    fields = [
        (f.name, f.type, f.default)
        for f in msgspec.structs.fields(arch)
        if f.name != "factored_act_init"
    ]
    shape = msgspec.defstruct(type(arch).__name__, fields, frozen=True)
    return shape(**{name: getattr(arch, name) for name, _, _ in fields})


def _flat(value: Any) -> list[Any]:
    """A bound as a flat list, whether the space recorded it as a scalar or nested lists."""
    if isinstance(value, list | tuple):
        return [leaf for item in value for leaf in _flat(item)]
    return [value]


def _with_buttons(obs: ObsBatch, grid: int, n_buttons: int) -> Tensor:
    """The scalar vector, followed by the ability buttons' readiness when there are buttons.

    Read off the mask's tail rather than carried as a key of its own: the observation's
    ``ability_ready`` is those same bits, and the codec stores the mask once.
    """
    if not n_buttons:
        return obs.vector
    ready = obs.mask[:, grid : grid + n_buttons].to(obs.vector.dtype)
    return torch.cat([obs.vector, ready], dim=-1)


class ResBlock(nn.Module):
    """Conv-norm-ReLU twice, plus the identity. The body is ``net.blocks`` of these."""

    def __init__(self, channels: int, norm_groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, channels)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return torch.relu(x + h)

    def initialise(self, generator: torch.Generator) -> None:
        for conv in (self.conv1, self.conv2):
            _orthogonal(conv.weight, HIDDEN_GAIN, generator)
            _zero_bias(conv)


class ClashTrunk(nn.Module):
    """Board planes, mask planes, coordinates and the broadcast scalar vector, through a stem and
    a residual body.

    The scalar vector is embedded and broadcast into every tile rather than concatenated after
    the pooling, so that elixir, the hand and the clock modulate what the convolution sees at a
    position instead of only correcting its summary afterwards.
    """

    def __init__(self, spec: EnvSpec, arch: ArchSpec) -> None:
        super().__init__()
        _check_arch(spec, arch)
        self.tiles = spec.tiles
        self.frame_stack = spec.frame_stack
        self.channels = arch.channels
        self.coord_conv = arch.coord_conv
        stacked = spec.frame_stack * (spec.n_planes + spec.obs_space["mask_planes"].shape[0])
        self.in_channels = stacked + (2 if arch.coord_conv else 0) + arch.vector_embed
        # The ability buttons' readiness rides with the scalar vector: whether a hero or a
        # champion can act now is a fact about the whole state, like elixir. It is read off the
        # mask's tail, which is the same bits as the observation's ability_ready.
        self.grid = spec.n_grid_actions
        self.n_buttons = spec.n_buttons
        self.vector_embed = nn.Linear(spec.vector_size + spec.n_buttons, arch.vector_embed)
        # D2: the card on each tile, embedded before the stem, as the hand slots are. Sized from
        # the space's own bound so the table follows the catalogue, and index 0 -- an empty tile
        # -- is fixed at zero: no card there, and nothing to learn about it.
        self.card_ids_embed: nn.Embedding | None = None
        if "card_ids" in spec.obs_space:
            ids = spec.obs_space["card_ids"]
            vocabulary = int(max(float(h) for h in _flat(ids.high))) + 1
            self.card_ids_embed = nn.Embedding(vocabulary, CARD_ID_EMBED, padding_idx=0)
            # The spells' planes (``spell_ids``) hold the same ids and go through this table too.
            self.in_channels += spec.frame_stack * id_planes(spec.obs_space) * CARD_ID_EMBED
        self.stem = nn.Conv2d(self.in_channels, arch.channels, 3, padding=1)
        self.stem_norm = nn.GroupNorm(arch.norm_groups, arch.channels)
        self.body = nn.ModuleList(
            ResBlock(arch.channels, arch.norm_groups) for _ in range(arch.blocks)
        )
        # Not persistent: the coordinates are a constant of the arena's geometry rather than
        # something learnt, and a snapshot should carry weights only.
        self.register_buffer("coords", _coord_planes(spec.tiles), persistent=False)

    def forward(self, obs: ObsBatch) -> Tensor:
        """``(B, C, H, W)``."""
        spatial = obs.spatial
        batch = spatial.shape[0]
        parts = [spatial, obs.mask_planes.to(spatial.dtype)]
        if self.coord_conv:
            parts.append(self.coords.expand(batch, -1, -1, -1).to(spatial.dtype))
        embedded = self.vector_embed(_with_buttons(obs, self.grid, self.n_buttons))
        parts.append(embedded.to(spatial.dtype)[:, :, None, None].expand(-1, -1, *self.tiles))
        if self.card_ids_embed is not None:
            if obs.card_ids is None:
                raise ValueError(
                    "this network reads card_ids and the batch carries none; it was built for an "
                    "observation with card identity switched on"
                )
            ids = self.card_ids_embed(obs.card_ids.long())  # (B, k*2, H, W, E)
            parts.append(ids.permute(0, 1, 4, 2, 3).flatten(1, 2).to(spatial.dtype))
        x = torch.cat(parts, dim=1)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"the trunk takes {self.in_channels} input channels -- a frame stack of "
                f"{self.frame_stack} over the board and the mask planes, the coordinates and the "
                f"vector embedding -- and this batch assembles {x.shape[1]}"
            )
        x = torch.relu(self.stem_norm(self.stem(x)))
        for block in self.body:
            x = block(x)
        return x

    def initialise(self, generator: torch.Generator) -> None:
        _orthogonal(self.vector_embed.weight, HIDDEN_GAIN, generator)
        _zero_bias(self.vector_embed)
        _orthogonal(self.stem.weight, HIDDEN_GAIN, generator)
        _zero_bias(self.stem)
        for block in self.body:
            block.initialise(generator)
        if self.card_ids_embed is not None:
            # Drawn last, so a network without card identity draws exactly what it always drew.
            with torch.no_grad():
                self.card_ids_embed.weight.normal_(0.0, EMBED_STD, generator=generator)
                self.card_ids_embed.weight[0].zero_()


class PointerPolicyHead(nn.Module):
    """One logit per (hand slot, tile), as the inner product of that tile's feature with a query
    built from the card in that slot, plus one logit for the no-op.

    The output order is the action space's own: the tile logits are a ``(slots, y, x)`` tensor
    flattened in C order behind the no-op at index 0, which is what ``GridActionParser.encode``
    computes. ``tile_logits`` is public so that the identity can be checked against the parser
    rather than argued about.
    """

    def __init__(self, spec: EnvSpec, arch: ArchSpec) -> None:
        super().__init__()
        _check_arch(spec, arch)
        self.hand = hand_fields(spec)
        self.tiles = spec.tiles
        self.n_actions = spec.n_actions
        self.noop_bias = float(arch.noop_bias)
        self.scale = _logit_scale(arch)
        self.feature = nn.Conv2d(arch.channels, arch.channels, 3, padding=1)
        self.feature_norm = nn.GroupNorm(arch.norm_groups, arch.channels)
        self.card_embed = nn.Embedding(self.hand.onehot_width, arch.card_embed)
        # The two scalars are the card's cost and whether it is affordable right now. They ride
        # beside the embedding rather than being folded into it because they are properties of
        # the state, not of the card.
        self.query = nn.Linear(arch.card_embed + 2, arch.channels)
        self.query_bias = nn.Linear(arch.card_embed + 2, 1)
        self.noop = nn.Linear(2 * arch.channels, 1)
        # One logit per ability button, from the same pooled summary as the no-op's: a press,
        # like a wait, is a decision about the whole board rather than about a tile. None when
        # the environment has no buttons, so such a network holds exactly what it always held.
        self.n_buttons = spec.n_buttons
        self.buttons = nn.Linear(2 * arch.channels, spec.n_buttons) if spec.n_buttons else None

    def _mapped(self, features: Tensor) -> Tensor:
        return torch.relu(self.feature_norm(self.feature(features)))

    def _queries(self, vector: Tensor) -> tuple[Tensor, Tensor]:
        """``(B, P, C)`` queries and ``(B, P, 1)`` per-slot biases, from the hand fields.

        The three fields are looked up by name in the layout the observation builder declares, so
        a field that moves upstream moves here and a field that disappears is a refusal naming
        it, rather than a silently wrong slice of somebody else's numbers.
        """
        hand = self.hand
        onehot = vector[:, hand.card_onehot].view(-1, hand.hand_size, hand.onehot_width)
        card_id = onehot.argmax(dim=-1)
        embedded = self.card_embed(card_id)
        cost = vector[:, hand.cost][..., None].to(embedded.dtype)
        affordable = vector[:, hand.affordable][..., None].to(embedded.dtype)
        q_in = torch.cat([embedded, cost, affordable], dim=-1)
        return self.query(q_in), self.query_bias(q_in)

    def _tiles(self, mapped: Tensor, obs: ObsBatch) -> Tensor:
        """The tile logits from an already-mapped feature map.

        The one place they are computed. ``tile_logits`` and ``forward`` both come through here
        rather than each writing the expression out, so the identity the action layout rests on
        is checked against the parser and never against a stale second copy of the head.
        """
        query, query_bias = self._queries(obs.vector)
        tiles = torch.einsum("bcyx,bpc->bpyx", mapped, query.to(mapped.dtype))
        return tiles * self.scale + query_bias.to(tiles.dtype)[..., None]

    def tile_logits(self, features: Tensor, obs: ObsBatch) -> Tensor:
        """``(B, P, H, W)``: the logit of deploying hand slot ``p`` at tile ``(y, x)``."""
        return self._tiles(self._mapped(features), obs)

    def forward(self, features: Tensor, obs: ObsBatch) -> Tensor:
        """``(B, n_actions)`` float32, raw: no softmax, no clamp, no mask. The no-op, then the
        tiles in the parser's ``(slot, y, x)`` order, then one logit per ability button."""
        mapped = self._mapped(features)
        tiles = self._tiles(mapped, obs)
        pooled = _pool(mapped)
        noop = self.noop(pooled) + self.noop_bias
        parts = [noop.to(tiles.dtype), tiles.reshape(tiles.shape[0], -1)]
        if self.buttons is not None:
            parts.append(self.buttons(pooled).to(tiles.dtype))
        return torch.cat(parts, dim=-1).float()

    def initialise(self, generator: torch.Generator) -> None:
        _orthogonal(self.feature.weight, HIDDEN_GAIN, generator)
        _zero_bias(self.feature)
        with torch.no_grad():
            self.card_embed.weight.normal_(0.0, EMBED_STD, generator=generator)
        for linear in (self.query, self.query_bias, self.noop):
            _orthogonal(linear.weight, HEAD_GAIN, generator)
            _zero_bias(linear)
        if self.buttons is not None:
            # Drawn last, so a network without buttons draws exactly what it always drew.
            _orthogonal(self.buttons.weight, HEAD_GAIN, generator)
            _zero_bias(self.buttons)


class FactoredStages(NamedTuple):
    """The factored head's three stages, as float32 log-probabilities.

    ``wait`` and ``act`` are ``(B,)`` and sum to one in probability. ``candidate`` is
    ``(B, P + K)``: the hand slots, then the ability buttons, given act. ``tile`` is ``(B, P, T)``:
    each slot's tiles in the parser's ``y * nx + x`` order, given that slot. An illegal entry has
    probability zero; a slot with no legal tile has a tile row nothing reaches.
    """

    wait: Tensor
    act: Tensor
    candidate: Tensor
    tile: Tensor


class FactoredPolicyHead(PointerPolicyHead):
    """Wait or act; then, given act, which hand slot or ability button; then which tile.

    The pointer head's tile map and card queries are kept: the tile stage is each slot's tile
    logits from the same inner product, normalised over that slot's legal tiles. The candidate
    stage scores a slot from its card query beside the pooled board, and a button from the pooled
    board alone. The gate scores act against wait from the pooled board, starting at
    ``net.factored_act_init``.

    ``forward`` writes the product as the flat vector the pointer head writes, in log space:
    ``log p(no-op) = log P(wait)``, ``log p(s, t) = log P(act) + log P(s | act) + log P(t | s)``
    and ``log p(button b) = log P(act) + log P(b | act)``. A slot is a candidate iff any of its
    tiles is legal and a button iff its mask bit is set; with no candidate the gate waits. So the
    masked distribution built on it is this one, and everything downstream of the logits --
    sampling, the stored log-probability, a decode plugin -- is unchanged.
    """

    def __init__(self, spec: EnvSpec, arch: ArchSpec) -> None:
        super().__init__(spec, arch)
        del self.noop
        channels = arch.channels
        self.grid = spec.n_grid_actions
        self.act_init = float(arch.factored_act_init)
        self.gate = nn.Linear(2 * channels, 1)
        self.slot = nn.Linear(3 * channels, 1)

    def stages(self, features: Tensor, obs: ObsBatch) -> FactoredStages:
        """The three stages on the rows of ``obs``, masked by its mask."""
        mapped = self._mapped(features)
        batch = mapped.shape[0]
        hand = self.hand.hand_size
        tiles = self._tiles(mapped, obs).float().reshape(batch, hand, -1)
        legal_tiles = obs.mask[:, NOOP + 1 : self.grid].view(batch, hand, -1)
        tile = torch.log_softmax(tiles.masked_fill(~legal_tiles, _STAGE_FILL), dim=-1)

        pooled = _pool(mapped)
        query, _ = self._queries(obs.vector)
        board = pooled[:, None, :].expand(-1, hand, -1)
        slot = self.slot(torch.cat([query.to(pooled.dtype), board], dim=-1)).squeeze(-1)
        parts = [slot.float()]
        if self.buttons is not None:
            parts.append(self.buttons(pooled).float())
        legal = torch.cat([legal_tiles.any(-1), obs.mask[:, self.grid :]], dim=-1)
        candidate = torch.log_softmax(
            torch.cat(parts, dim=-1).masked_fill(~legal, _STAGE_FILL), dim=-1
        )

        act_logit = self.gate(pooled).squeeze(-1).float()
        act_logit = act_logit.masked_fill(~legal.any(-1), _STAGE_FILL)
        gate = torch.log_softmax(torch.stack([torch.zeros_like(act_logit), act_logit], -1), -1)
        return FactoredStages(gate[:, 0], gate[:, 1], candidate, tile)

    def forward(self, features: Tensor, obs: ObsBatch) -> Tensor:
        """``(B, n_actions)`` float32: the flat log-probabilities, illegal entries included (the
        masked distribution built on them removes those)."""
        wait, act, candidate, tile = self.stages(features, obs)
        hand = self.hand.hand_size
        grid = act[:, None, None] + candidate[:, :hand, None] + tile
        parts = [wait[:, None], grid.reshape(grid.shape[0], -1)]
        if self.buttons is not None:
            parts.append(act[:, None] + candidate[:, hand:])
        return torch.cat(parts, dim=-1)

    def initialise(self, generator: torch.Generator) -> None:
        _orthogonal(self.feature.weight, HIDDEN_GAIN, generator)
        _zero_bias(self.feature)
        with torch.no_grad():
            self.card_embed.weight.normal_(0.0, EMBED_STD, generator=generator)
        for linear in (self.query, self.query_bias, self.slot, self.gate):
            _orthogonal(linear.weight, HEAD_GAIN, generator)
            _zero_bias(linear)
        if self.buttons is not None:
            _orthogonal(self.buttons.weight, HEAD_GAIN, generator)
            _zero_bias(self.buttons)
        with torch.no_grad():
            self.gate.bias.fill_(math.log(self.act_init / (1.0 - self.act_init)))


def head_meta(arch: ArchSpec) -> dict[str, Any]:
    """What an actor's ``spec.json`` meta says about its policy head, so that a reader rebuilds
    the same net from the folder alone: the head's name and the factored gate's starting value,
    which ``arch_digest`` leaves out. Empty for the pointer head, so its folders are what they
    always were. A tool that writes a ``SnapshotSpec`` of its own, rather than taking a run's
    ``artifact_spec()``, puts this in the spec's ``meta``.
    """
    if arch.policy_head == "pointer":
        return {}
    return {"policy_head": arch.policy_head, "factored_act_init": arch.factored_act_init}


def build_policy_head(spec: EnvSpec, arch: ArchSpec) -> PointerPolicyHead:
    """The policy head ``net.policy_head`` names. Every place that builds an actor -- the factory,
    a snapshot's bare actor, a saved bot -- comes through here, so none of them can build the
    other head than the weights were trained in."""
    _check_arch(spec, arch)
    if arch.policy_head == "factored":
        return FactoredPolicyHead(spec, arch)
    return PointerPolicyHead(spec, arch)


class ValueHead(nn.Module):
    """One scalar from the pooled board, the scalar vector and a summary of the mask.

    ``legal_frac`` is the fraction of each hand slot's tiles that are legal now. The mask is a
    function of the state, so a summary of it is a legitimate value feature and a mildly helpful
    one: it says how constrained this position is. The critic never sees the full mask, and it
    never receives the entropy gradient.
    """

    def __init__(self, spec: EnvSpec, arch: ArchSpec) -> None:
        super().__init__()
        _check_arch(spec, arch)
        self.hand_size = spec.hand_size
        self.grid = spec.n_grid_actions
        self.n_buttons = spec.n_buttons
        self.vector_embed = nn.Linear(spec.vector_size + spec.n_buttons, arch.vector_embed)
        width = 2 * arch.channels + arch.vector_embed + spec.hand_size
        self.hidden = nn.Linear(width, arch.value_hidden)
        self.out = nn.Linear(arch.value_hidden, 1)

    def forward(self, features: Tensor, obs: ObsBatch) -> Tensor:
        """``(B,)`` float32."""
        pooled = _pool(features)
        embedded = self.vector_embed(_with_buttons(obs, self.grid, self.n_buttons))
        embedded = embedded.to(pooled.dtype)
        legal = obs.mask[:, NOOP + 1 : self.grid].view(obs.mask.shape[0], self.hand_size, -1)
        legal_frac = legal.to(pooled.dtype).mean(dim=-1)
        x = torch.cat([pooled, embedded, legal_frac], dim=-1)
        return self.out(torch.relu(self.hidden(x))).squeeze(-1).float()

    def initialise(self, generator: torch.Generator) -> None:
        for linear in (self.vector_embed, self.hidden):
            _orthogonal(linear.weight, HIDDEN_GAIN, generator)
            _zero_bias(linear)
        _orthogonal(self.out.weight, VALUE_GAIN, generator)
        _zero_bias(self.out)


class DefaultNetworkFactory(NetworkFactory):
    """Builds the shipped pair, and says what it built.

    The weights are drawn on the CPU from the ``torch/init`` stream and moved to the device
    afterwards, so a run's initial parameters are a function of its master seed and nothing else
    -- not of the device it started on, and not of what has already drawn from a global
    generator.
    """

    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)

    def generator(self) -> torch.Generator:
        """A CPU generator seeded from ``torch/init``."""
        return torch.Generator().manual_seed(derive_int(self.master_seed, TORCH_INIT))

    def build(
        self, spec: EnvSpec, arch: ArchSpec, device: Any = "cpu", dtype: Any = None
    ) -> ActorCritic:
        _check_arch(spec, arch)
        if arch.init != "orthogonal":
            raise PreflightError(f"net.init {arch.init!r} is not one of 'orthogonal'")
        torch_device = torch.device(device)
        autocast_dtype = resolve_dtype(arch.autocast_dtype if dtype is None else dtype)

        generator = self.generator()
        actor = ClashActor(ClashTrunk(spec, arch), build_policy_head(spec, arch))
        actor.initialise(generator)
        model: ActorCritic
        if arch.separate_trunks:
            critic = ClashCritic(ClashTrunk(spec, arch), ValueHead(spec, arch))
            critic.initialise(generator)
            model = SeparateActorCritic(actor, critic, arch, torch_device, autocast_dtype)
        else:
            value_head = ValueHead(spec, arch)
            value_head.initialise(generator)
            model = SharedTrunkActorCritic(actor, value_head, arch, torch_device, autocast_dtype)
        model.to(torch_device)
        model.arch_digest = self.arch_digest(spec, arch)
        return model

    def arch_digest(self, spec: EnvSpec, arch: ArchSpec) -> str:
        """sha256 of the canonical JSON of the architecture and the environment that shaped it.

        Everything that decides a tensor's shape is in here: the architecture whole, the
        observation space, the frame stack, the catalogue size, the action width, and the names
        of the classes that were assembled. A snapshot or a checkpoint from a different one is
        then refused by name instead of arriving as a shape error from the middle of a load.
        """
        return digest_of(
            {
                "arch": _arch_for_digest(arch),
                "obs_space": {
                    key: {"shape": k.shape, "dtype": k.dtype, "low": k.low, "high": k.high}
                    for key, k in spec.obs_space.items()
                },
                "frame_stack": spec.frame_stack,
                "num_cards": spec.num_cards,
                "n_actions": spec.n_actions,
                "hand_size": spec.hand_size,
                "tiles": spec.tiles,
                "trunk": ClashTrunk.__name__,
                "head": (
                    FactoredPolicyHead
                    if arch.policy_head == "factored"
                    else PointerPolicyHead
                ).__name__,
                "actor_critic": (
                    SeparateActorCritic.__name__
                    if arch.separate_trunks
                    else SharedTrunkActorCritic.__name__
                ),
                # Only when there are card ids to embed, so every digest before D2 is unchanged.
                **({"card_id_embed": CARD_ID_EMBED} if "card_ids" in spec.obs_space else {}),
                # Only when the spells' ids are on, so every digest before them is unchanged.
                **(
                    {"spell_id_planes": int(spec.obs_space["spell_ids"].shape[0])}
                    if "spell_ids" in spec.obs_space
                    else {}
                ),
            }
        )

    def check_digest(
        self, spec: EnvSpec, arch: ArchSpec, digest: str, *, source: str = "the weights"
    ) -> None:
        """Refuse a load whose architecture is not this one, with both digests in the message."""
        current = self.arch_digest(spec, arch)
        if digest != current:
            raise PreflightError(
                f"{source} were built from arch_digest {digest}, and this run's architecture "
                f"hashes to {current}. A network built from a different architecture, "
                "observation space, frame stack or action width cannot be loaded into this one"
            )
