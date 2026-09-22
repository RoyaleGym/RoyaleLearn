"""Name-addressed random streams.

Every generator in the harness is derived from the run's master seed and a PATH -- a string
naming the consumer, such as ``"ppo/minibatch/iteration/4/epoch/1"`` -- rather than by spawning
children of a root ``SeedSequence`` in the order the code happens to ask for them. Positional
spawning makes every stream downstream of a new consumer move, so adding a diagnostic that draws
one number changes the actions a policy takes. Here a stream is a pure function of its name, so a
config change that ought to be irrelevant is irrelevant, and two runs of the same identity draw
the same numbers however their code paths are ordered.

The namespace below is the whole of it. A new consumer adds a row rather than reusing a
neighbour's path, because two consumers on one path advance each other's stream.
"""

from __future__ import annotations

import hashlib
import struct
from typing import NamedTuple

import numpy as np

__all__ = [
    "ACT_CYCLE",
    "ENV_SHARD",
    "EVAL_BOOTSTRAP",
    "EVAL_MATCH",
    "EVAL_SEED_SET",
    "MATCH_BATTLE",
    "PPO_MINIBATCH",
    "SCRIPTED_SLOT",
    "STREAMS",
    "TORCH_CUDA",
    "TORCH_GLOBAL",
    "TORCH_INIT",
    "Stream",
    "derive_generator",
    "derive_int",
    "derive_seedseq",
    "stream_path",
]


def derive_seedseq(master_seed: int, path: str) -> np.random.SeedSequence:
    """The ``SeedSequence`` for one named stream of one run.

    The path is hashed to a 128-bit spawn key rather than appended to the entropy, so that the
    master seed stays the run's single number and the path cannot collide with a seed value.
    blake2b because it is in the standard library, is fast on short strings, and -- the property
    that matters -- is fixed forever, which a string hash with per-process randomisation is not.
    """
    digest = hashlib.blake2b(path.encode("utf-8"), digest_size=16).digest()
    return np.random.SeedSequence(entropy=master_seed, spawn_key=struct.unpack("<4I", digest))


def derive_generator(master_seed: int, path: str) -> np.random.Generator:
    """A PCG64 generator for one named stream. PCG64 is numpy's default and is stable across
    versions and platforms, which is what makes a recorded run reproducible on another machine."""
    return np.random.Generator(np.random.PCG64(derive_seedseq(master_seed, path)))


def derive_int(master_seed: int, path: str) -> int:
    """A reproducible 63-bit integer, for env seeds.

    63 rather than 64 bits because the seed crosses into ``ClashSelfPlayVecEnv.reset(seed=...)``
    and gymnasium's seeding refuses a value that does not fit a signed 64-bit integer.
    """
    words = derive_seedseq(master_seed, path).generate_state(2, dtype=np.uint32).astype(np.uint64)
    return int((words[0] | (words[1] << np.uint64(32))) >> np.uint64(1))


class Stream(NamedTuple):
    """One row of the namespace: a path template and what draws from it."""

    template: str
    consumer: str


#: The one ``ClashSelfPlayVecEnv.reset(seed=...)`` of a shard; ``generation`` counts respawns.
ENV_SHARD = "env/worker/{worker}/shard/{shard}/gen/{generation}"
#: The matchmaker's draw for one episode of one battle. Addressed by the battle and its reset
#: ordinal rather than by the iteration and the slot, so an episode meets the same opponent
#: however the iteration boundary happens to fall across it.
MATCH_BATTLE = "match/battle/{battle}/ordinal/{ordinal}"
#: The ``(R,)`` uniform vector that drives action sampling at one cycle.
ACT_CYCLE = "act/iteration/{iteration}/cycle/{cycle}"
#: A worker-side scripted opponent's generator, per slot.
SCRIPTED_SLOT = "scripted/worker/{worker}/slot/{slot}/gen/{generation}"
#: The minibatch permutation of one epoch of one iteration.
PPO_MINIBATCH = "ppo/minibatch/iteration/{iteration}/epoch/{epoch}"
#: The fixed evaluation seed set, drawn once at run start.
EVAL_SEED_SET = "eval/seed_set"
#: One evaluation battle: a comparison, a seed index and which side the candidate took.
EVAL_MATCH = "eval/match/{comparison}/{seed_index}/{side}"
#: The bootstrap resampling of one comparison.
EVAL_BOOTSTRAP = "eval/bootstrap/{comparison}"
#: Network initialisation.
TORCH_INIT = "torch/init"
#: ``torch.manual_seed`` and ``torch.cuda.manual_seed_all`` at start-up.
TORCH_GLOBAL = "torch/global"
TORCH_CUDA = "torch/cuda"

#: The namespace, in the order ``docs/determinism.md`` prints it. Every stream the harness draws
#: from is here; a path that is not is a bug, and ``stream_path`` refuses one.
STREAMS: tuple[Stream, ...] = (
    Stream(ENV_SHARD, "the vec env reset of one shard, once per respawn generation"),
    Stream(MATCH_BATTLE, "the matchmaker's assignment for one episode of one battle"),
    Stream(ACT_CYCLE, "the uniforms that drive action sampling at one cycle"),
    Stream(SCRIPTED_SLOT, "one worker-side scripted opponent"),
    Stream(PPO_MINIBATCH, "the minibatch permutation of one epoch"),
    Stream(EVAL_SEED_SET, "the frozen evaluation seed set"),
    Stream(EVAL_MATCH, "one evaluation battle"),
    Stream(EVAL_BOOTSTRAP, "the bootstrap resampling of one comparison"),
    Stream(TORCH_INIT, "network initialisation"),
    Stream(TORCH_GLOBAL, "torch.manual_seed at start-up"),
    Stream(TORCH_CUDA, "torch.cuda.manual_seed_all at start-up"),
)

_TEMPLATES = frozenset(stream.template for stream in STREAMS)


def stream_path(template: str, /, **fields: object) -> str:
    """Fill one row of ``STREAMS``.

    Going through here rather than writing an f-string at the call site means a typo in a path is
    a ``KeyError`` naming the template at the moment it is drawn, instead of a private stream that
    quietly works and silently fails to be the stream anyone meant.
    """
    if template not in _TEMPLATES:
        raise KeyError(f"{template!r} is not a stream in seeding.STREAMS")
    return template.format(**fields)
