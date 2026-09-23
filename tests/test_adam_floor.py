"""Where Adam stops being adaptive, and why the actor reaches it and the critic does not.

Adam's step is ``lr * m / (sqrt(v) + eps)``. The division is the whole point: a parameter with a
tiny gradient and a tiny second moment still moves about ``lr``, which is what makes one learning
rate work across layers whose gradients differ by orders of magnitude. That holds only while
``sqrt(v) >> eps``. Below the eps floor the denominator stops tracking the gradient and the step
becomes ``lr * m / eps`` -- proportional to the gradient again, and for a small gradient, small.

On 2026-09-22 the train session measured ``grad_norm_actor`` at 0.0068 against a critic at 26.6,
a KL of 1e-5 per update and a clip fraction of zero, and called the asymmetry unexplained. It is
the floor: with ``adam_eps`` at its default 1e-5, most of the actor's second moments sit under it
and the actor's effective step is a fraction of the one its learning rate names, while the critic
-- four orders of magnitude up in gradient norm -- gets the adaptive step the schedule assumes.
Measured on a three-iteration toy run before this key existed: 57.6% of the actor's parameters
under the floor against 33.7% of the critic's, and 99.2% of the actor's on the real run.

The key does not decide anything. It puts the number in the row so that the next person reads it
instead of inferring it from a gradient ratio.
"""

from __future__ import annotations

import pytest
import torch

from royalelearn.learn.ppo import adam_eps_floor_frac


def optimizer_at(gradient: float, *, eps: float, steps: int = 20) -> torch.optim.Adam:
    """One parameter carrying a constant gradient, stepped until its second moment settles."""
    param = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([param], lr=1e-3, eps=eps)
    for _ in range(steps):
        opt.zero_grad()
        param.grad = torch.full((4,), gradient)
        opt.step()
    return opt


def test_a_gradient_far_below_eps_is_entirely_on_the_floor() -> None:
    """sqrt(v) settles at about the gradient itself, so 1e-8 against eps 1e-5 is under it."""
    assert adam_eps_floor_frac(optimizer_at(1e-8, eps=1e-5)) == pytest.approx(1.0)


def test_a_gradient_far_above_eps_is_entirely_off_it() -> None:
    assert adam_eps_floor_frac(optimizer_at(1.0, eps=1e-5)) == pytest.approx(0.0)


def test_the_floor_moves_with_eps_and_not_with_the_gradient() -> None:
    """The same gradient, adaptive under one epsilon and floored under another.

    This is the arm the train session measured 11x more actor movement on. If the fraction did
    not move here, the key would be reporting the gradient under another name.
    """
    gradient = 1e-6
    assert adam_eps_floor_frac(optimizer_at(gradient, eps=1e-5)) == pytest.approx(1.0)
    assert adam_eps_floor_frac(optimizer_at(gradient, eps=1e-8)) == pytest.approx(0.0)


def test_an_optimizer_that_has_not_stepped_reports_nothing() -> None:
    """No second moments is not "no parameters on the floor"; a 0.0 would read as adaptive."""
    param = torch.zeros(4, requires_grad=True)
    assert adam_eps_floor_frac(torch.optim.Adam([param], lr=1e-3)) is None


def test_a_mixture_reports_the_share_of_parameters_and_not_of_tensors() -> None:
    """A big floored tensor beside a small adaptive one is mostly floored.

    Counting tensors instead of entries would answer 0.5 here, which is how a network whose one
    large layer is frozen reads as half healthy.
    """
    small = torch.zeros(1, requires_grad=True)
    big = torch.zeros(99, requires_grad=True)
    opt = torch.optim.Adam([small, big], lr=1e-3, eps=1e-5)
    for _ in range(20):
        opt.zero_grad()
        small.grad = torch.full((1,), 1.0)
        big.grad = torch.full((99,), 1e-9)
        opt.step()

    assert adam_eps_floor_frac(opt) == pytest.approx(0.99)


def test_the_real_update_publishes_both_optimizers(tmp_path) -> None:
    """End to end: the two keys are in the row a real iteration writes, and they differ.

    The asymmetry is the finding, so a test that only checked the keys existed would pass on an
    implementation that reported the same number twice.
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_coordinator import coordinator, tiny_config

    config = tiny_config(tmp_path)
    with coordinator(config) as run:
        run.learn(until_timesteps=config.ppo.timesteps_per_iteration * 2)
        run_dir = Path(run.run_dir)

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    last = rows[-1]
    actor = last["ppo/adam_eps_floor_frac_actor"]
    critic = last["ppo/adam_eps_floor_frac_critic"]
    assert 0.0 <= actor <= 1.0 and 0.0 <= critic <= 1.0
    assert actor != critic, (
        "both optimizers reported the same share of parameters under the eps floor, which is "
        "what a key computed once and published twice would do"
    )
