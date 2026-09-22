"""Every ABC and struct the harness is written against.

This subpackage imports numpy and msgspec and nothing else that matters: torch appears in type
annotations only, behind ``TYPE_CHECKING``. That is what lets a rollout worker, the CLI's
config and identity commands, and ``import royalelearn`` itself run in an environment with no
torch installed -- and what keeps a worker's resident memory three hundred megabytes smaller
than the parent's.

The concrete implementations live outside ``api/`` and may import whatever they need:

    RolloutSource        rollout/farm.py, rollout/inline.py
    ActorCritic          learn/actor_critic.py
    ObsCodec             rollout/codec.py
    ExperienceBuffer     learn/buffer.py
    AdvantageEstimator   learn/gae.py
    Update               learn/ppo.py
    Schedule             learn/schedules.py
    Matchmaker, Rater    ladder/matchmaker.py, ladder/rating.py
    MetricsSink, Alarm   metrics/sinks.py, metrics/alarms.py
    CheckpointStore      checkpoint.py
"""

from __future__ import annotations

from .advantage import AdvantageEstimator, AdvantageStats
from .buffer import CodecTable, ExperienceBuffer, ObsCodec
from .checkpoint import Checkpointable, CheckpointStore, Manifest, RngState
from .ladder import (
    ConditionResult,
    EvictionPolicy,
    GateDecision,
    Matchmaker,
    PromotionGate,
    Rater,
    RatingTable,
    SnapshotStore,
)
from .metrics import Alarm, AlarmResult, MetricRow, MetricsSink, MetricValue
from .policy import (
    ActionDistribution,
    Actor,
    ActorCritic,
    ActResult,
    BackpropResult,
    Critic,
    NetworkFactory,
    ObsBatch,
)
from .rollout import (
    Assignment,
    Close,
    Defer,
    EnvSpec,
    EpisodeRecord,
    ObsKeySpec,
    Plan,
    RolloutRound,
    RolloutSource,
    SetState,
    SlotPlan,
    Spaces,
    Step,
    WorkerCommand,
    WorkerFailure,
)
from .schedule import Schedule, ScheduleState
from .update import Update, UpdateResult

__all__ = [
    "ActResult",
    "ActionDistribution",
    "Actor",
    "ActorCritic",
    "AdvantageEstimator",
    "AdvantageStats",
    "Alarm",
    "AlarmResult",
    "Assignment",
    "BackpropResult",
    "CheckpointStore",
    "Checkpointable",
    "Close",
    "CodecTable",
    "ConditionResult",
    "Critic",
    "Defer",
    "EnvSpec",
    "EpisodeRecord",
    "EvictionPolicy",
    "ExperienceBuffer",
    "GateDecision",
    "Manifest",
    "Matchmaker",
    "MetricRow",
    "MetricValue",
    "MetricsSink",
    "NetworkFactory",
    "ObsBatch",
    "ObsCodec",
    "ObsKeySpec",
    "Plan",
    "PromotionGate",
    "Rater",
    "RatingTable",
    "RngState",
    "RolloutRound",
    "RolloutSource",
    "Schedule",
    "ScheduleState",
    "SetState",
    "SlotPlan",
    "SnapshotStore",
    "Spaces",
    "Step",
    "Update",
    "UpdateResult",
    "WorkerCommand",
    "WorkerFailure",
]
