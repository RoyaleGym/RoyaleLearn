"""The determinism tiers, and the process settings each one needs.

Three tiers, of which the first is unconditional (``docs/harness-spec.md`` section 5.1):

T1 env-exact   given the run identity, every episode's engine state-hash sequence, every
               observation, every mask and every reward are bit-identical, on any machine,
               forever. It costs nothing and is a property of the engine and of name-addressed
               seeding, so there is no switch for it and ``apply`` does not mention it.
T2 run-exact   T1, and every gradient, parameter and metric row bit-identical on the same device
               class and torch/CUDA build. The default, at 10-20% of throughput.
T3 throughput  T1 only; the learner may pick nondeterministic kernels.

Two of the settings T2 needs cannot be applied from here, because they must be in place BEFORE
torch initialises CUDA and before numpy imports its BLAS: ``CUBLAS_WORKSPACE_CONFIG`` and the
three BLAS thread counts. They are environment variables, the entry point sets them, and
``apply`` asserts rather than sets, so a run that would have been silently non-reproducible dies
at start-up naming the entry point instead of producing a curve that cannot be repeated.

This module imports torch inside the functions that need it. Importing it at module scope would
put a 300 MB, three-second dependency on the import path of ``royalelearn.cli``, which has to
work in an environment without torch at all.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Any

from .errors import PreflightError

__all__ = [
    "ACCEPTED_CUBLAS_VALUES",
    "BLAS_THREAD_VARS",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLAS_WORKSPACE_VALUE",
    "TIERS",
    "apply",
    "apply_blas_thread_env",
    "apply_cublas_workspace_config",
    "require_cublas_workspace_config",
]

#: The tiers a config may ask for. T1 is unconditional and is therefore not one of them.
TIERS: tuple[str, ...] = ("run_exact", "throughput")

CUBLAS_WORKSPACE_CONFIG = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_VALUE = ":4096:8"
#: Both values cuBLAS accepts as deterministic; the second trades a little throughput for memory.
ACCEPTED_CUBLAS_VALUES: tuple[str, ...] = (":4096:8", ":16:8")

#: Set to "1" in every rollout worker before numpy is imported. The thread count changes the
#: order of BLAS reductions, and the workers are engine-bound rather than BLAS-bound, so pinning
#: it costs nothing there and buys a float sum that does not depend on how busy the machine was.
BLAS_THREAD_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def apply_cublas_workspace_config(env: MutableMapping[str, str] | None = None) -> str:
    """Set ``CUBLAS_WORKSPACE_CONFIG`` if it is unset, and return its value.

    Called by the entry point before torch is imported. An existing value is left alone even when
    it is not one this harness would have chosen: the operator's own setting wins, and
    ``require_cublas_workspace_config`` is what decides whether it is good enough.
    """
    environ = os.environ if env is None else env
    return environ.setdefault(CUBLAS_WORKSPACE_CONFIG, CUBLAS_WORKSPACE_VALUE)


def apply_blas_thread_env(env: MutableMapping[str, str] | None = None) -> None:
    """Pin the BLAS thread counts to one. Must run before numpy is imported."""
    environ = os.environ if env is None else env
    for name in BLAS_THREAD_VARS:
        environ[name] = "1"


def require_cublas_workspace_config(
    entry_point: str = "royalelearn.cli", env: MutableMapping[str, str] | None = None
) -> None:
    """Refuse to run T2 without a deterministic cuBLAS workspace.

    cuBLAS reads this variable once, when CUDA initialises, so setting it here would be too late
    and setting it silently would be worse than not setting it at all: the run would look
    reproducible and would not be.
    """
    environ = os.environ if env is None else env
    value = environ.get(CUBLAS_WORKSPACE_CONFIG)
    if value in ACCEPTED_CUBLAS_VALUES:
        return
    seen = "unset" if value is None else repr(value)
    raise PreflightError(
        f"{CUBLAS_WORKSPACE_CONFIG} is {seen}, and determinism.tier='run_exact' needs one of "
        f"{', '.join(ACCEPTED_CUBLAS_VALUES)} set BEFORE torch initialises CUDA. "
        f"{entry_point} sets it from the config; set it in the environment if you are calling "
        "the harness as a library."
    )


def apply(
    tier: str, *, torch_threads: int = 1, entry_point: str = "royalelearn.cli"
) -> dict[str, Any]:
    """Put the process into ``tier`` and return what was applied, for the record.

    T2 keeps bf16 autocast. Deterministic kernels are bit-reproducible run to run at any
    precision, so the tier costs nothing in precision: it buys exactly what it says, which is two
    runs of one identity agreeing row for row. tf32 is off because it is not bit-reproducible
    across the different batch shapes the rollout forward and the update forward use.
    """
    if tier not in TIERS:
        raise ValueError(f"determinism tier {tier!r} is not one of {TIERS}")

    import torch

    applied: dict[str, Any] = {"tier": tier, "torch_threads": torch_threads}
    torch.set_num_threads(torch_threads)
    if tier == "run_exact":
        require_cublas_workspace_config(entry_point)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        applied.update(
            deterministic_algorithms=True,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            allow_tf32=False,
            cublas_workspace_config=os.environ.get(CUBLAS_WORKSPACE_CONFIG),
        )
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        applied.update(deterministic_algorithms=False, cudnn_benchmark=True)
    return applied
