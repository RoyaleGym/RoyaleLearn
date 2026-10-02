"""How a frozen policy picks its action: sampled, its most likely action, or a decode of your own.

``stochastic`` samples the masked distribution; ``argmax`` takes its most likely action. Anything
else is yours to write: ``plugin:<module>:<function>`` names a function that takes a batch of rows'
masked log-probabilities ``(B, n_actions)``, their mask ``(B, n_actions)`` and their observation
vector ``(B, V)``, all on the run's device, and returns ``(B,)`` integer actions. It should be
deterministic and batched, like the two built in. Every action it returns is checked against the
mask, and an illegal one stops the run naming the plugin.

The learner's own seats always sample, because that is the policy the update trains. These modes
are for frozen opponents (``ladder.opponent_mode``) and for evaluation (``ladder.release_mode``).
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from .distribution import MaskedCategorical

__all__ = [
    "ARGMAX",
    "PLUGIN",
    "STOCHASTIC",
    "decode_actions",
    "mode_problems",
    "parse_mode",
    "resolve_decoder",
]

STOCHASTIC, ARGMAX, PLUGIN = "stochastic", "argmax", "plugin"
_PLUGIN_PREFIX = PLUGIN + ":"

#: A plugin decode: masked log-probabilities, mask and observation vector in, actions out.
Decoder = Callable[[Any, Any, Any], Any]


def parse_mode(mode: str) -> tuple[str, str]:
    """``(kind, target)``: the target is ``<module>:<function>`` for a plugin, else empty."""
    if mode in (STOCHASTIC, ARGMAX):
        return mode, ""
    if mode.startswith(_PLUGIN_PREFIX):
        module, _, function = mode[len(_PLUGIN_PREFIX) :].partition(":")
        if module and function:
            return PLUGIN, f"{module}:{function}"
        raise ValueError(
            f"decode mode {mode!r} names no function: write plugin:<module>:<function>"
        )
    name = mode.split(":", 1)[0]
    raise ValueError(
        f"decode {name!r} is not built in; 'stochastic' and 'argmax' are, or supply your own "
        "with plugin:<module>:<function>"
    )


def mode_problems(mode: str, where: str) -> list[str]:
    try:
        parse_mode(mode)
    except ValueError as exc:
        return [f"{where}: {exc}"]
    return []


@functools.cache
def resolve_decoder(mode: str) -> Decoder | None:
    """The plugin function ``mode`` names, imported, or None for a built-in mode. A module or
    function that cannot be found is a ``ValueError`` naming it."""
    kind, target = parse_mode(mode)
    if kind != PLUGIN:
        return None
    module_name, function_name = target.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"decode plugin {target!r}: cannot import {module_name!r} ({exc})"
        ) from exc
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(
            f"decode plugin {target!r}: {module_name} has no function {function_name!r}"
        )
    return function  # type: ignore[no-any-return]


def decode_actions(
    distribution: MaskedCategorical,
    mode: str,
    uniforms: Tensor,
    *,
    vector: Tensor,
) -> Tensor:
    """``(B,)`` int64 actions of ``distribution`` under ``mode``; ``uniforms`` feed a sample, and
    a plugin is handed the rows' observation ``vector`` too."""
    import torch

    kind, target = parse_mode(mode)
    if kind == STOCHASTIC:
        return distribution.sample(uniforms)
    if kind == ARGMAX:
        return distribution.mode()
    decoder = resolve_decoder(mode)
    assert decoder is not None
    mask = distribution.mask
    actions = torch.as_tensor(decoder(distribution.log_probs, mask, vector), device=mask.device)
    if actions.shape != (mask.shape[0],) or actions.dtype.is_floating_point:
        raise ValueError(
            f"decode plugin {target!r} returned {actions.dtype} {tuple(actions.shape)}; it must "
            f"return one integer action per row, shape ({mask.shape[0]},)"
        )
    actions = actions.to(torch.int64)
    in_range = (actions >= 0) & (actions < mask.shape[1])
    legal = in_range & mask.gather(1, actions.clamp(0, mask.shape[1] - 1)[:, None])[:, 0]
    if not bool(legal.all()):
        rows = (~legal).nonzero().flatten()[:8].tolist()
        raise ValueError(
            f"decode plugin {target!r} chose an illegal action on rows {rows}: every action it "
            "returns must be allowed by that row's mask"
        )
    return actions
