"""The most load-bearing index identity in the harness, checked exhaustively against the parser.

The pointer head emits its tile logits as a ``(hand slot, y, x)`` tensor and flattens it in C
order behind the no-op. Every action the learner ever takes, every mask bit it ever reads and
every logit it ever backpropagates through depends on that flattening being the same order the
action space uses -- and nothing about it is checkable by reading either file, because both are
correct on their own terms.

So it is checked by asking the parser, for every non-no-op action there is, rather than by
repeating its arithmetic here. If RoyaleGym ever renumbers the space, this file fails first and
says so, instead of the learner quietly deploying into the wrong half of the arena.
"""

from __future__ import annotations

from typing import Any

import pytest

from royalelearn.api.policy import ObsBatch


@pytest.fixture(scope="module")
def parser(vec_env: Any) -> Any:
    """The action parser the environment is actually running."""
    return vec_env.envs[0].action_parser


@pytest.fixture(scope="module")
def grid(parser: Any) -> tuple[int, int, int]:
    """``(slots, ny, nx)``, read off the parser rather than off the arena."""
    shape = parser.mask_plane_shape()
    assert shape is not None, "the default parser's action space is a grid"
    return shape


def test_the_space_is_a_noop_plus_the_grid(parser: Any, grid: tuple[int, int, int]) -> None:
    slots, ny, nx = grid
    assert parser.noop() == 0
    assert parser.space.n == 1 + slots * ny * nx


def test_every_action_is_the_argmax_of_its_one_hot_plane(
    torch: Any, parser: Any, grid: tuple[int, int, int]
) -> None:
    """Exhaustive over every non-no-op action.

    Row ``i`` of the identity matrix, viewed as ``(slots, ny, nx)``, is the one-hot of the
    ``i``-th cell in C order. Flattened, its argmax is where a head that reshapes in C order puts
    that cell's logit -- and that has to be ``encode(slot, x, y) - 1`` for every cell, the minus
    one being the no-op the head prepends.
    """
    slots, ny, nx = grid
    n_tiles = slots * ny * nx
    encoded = torch.tensor(
        [
            parser.encode(slot, x, y)
            for slot in range(slots)
            for y in range(ny)
            for x in range(nx)
        ]
    )
    one_hots = torch.eye(n_tiles).view(n_tiles, slots, ny, nx)
    assert torch.equal(one_hots.reshape(n_tiles, -1).argmax(dim=-1) + 1, encoded)
    assert int(encoded.min()) == 1
    assert int(encoded.max()) == parser.space.n - 1


def test_encode_and_decode_are_inverses(parser: Any, grid: tuple[int, int, int]) -> None:
    slots, ny, nx = grid
    for slot in range(slots):
        for y in range(ny):
            for x in range(nx):
                assert parser.decode(parser.encode(slot, x, y)) == (slot, x, y)


def test_the_head_flattens_tiles_in_the_parsers_order(
    torch: Any, env_spec: Any, parser: Any, grid: tuple[int, int, int]
) -> None:
    """The head's own output, split back into planes, is its ``tile_logits`` unchanged.

    Together with the identity above, this is the whole chain: ``tile_logits[b, slot, y, x]``
    lands at ``encode(slot, x, y)`` of the logit vector, and index 0 is the no-op.
    """
    from royalelearn.config import NetConfig
    from royalelearn.learn.nets import DefaultNetworkFactory

    slots, ny, nx = grid
    arch = NetConfig(channels=8, blocks=1, norm_groups=4, card_embed=8, value_hidden=16,
                     device="cpu", autocast_dtype="float32")
    model = DefaultNetworkFactory(20).build(env_spec, arch, "cpu")
    generator = torch.Generator().manual_seed(11)
    batch = 3
    mask = torch.rand(batch, env_spec.n_actions, generator=generator) < 0.4
    mask[:, parser.noop()] = True
    obs = ObsBatch(
        spatial=torch.rand(
            batch, env_spec.spatial_shape[0], ny, nx, generator=generator
        ),
        mask_planes=mask[:, 1:].view(batch, slots, ny, nx).float(),
        vector=torch.rand(batch, env_spec.vector_size, generator=generator),
        mask=mask,
    )
    with torch.no_grad():
        features = model.actor.trunk(obs)
        logits = model.actor.head(features, obs)
        tiles = model.actor.head.tile_logits(features, obs)
    assert logits.shape == (batch, parser.space.n)
    assert torch.equal(logits[:, 1:].view(batch, slots, ny, nx), tiles.float())
