from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from typing_extensions import TypedDict

from livekit import rtc

from ..language import LanguageCode
from ..llm import ChatContext
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import is_given


@dataclass
class TurnDetectionEvent:
    type: Literal["eot_prediction"]
    end_of_turn_probability: float
    last_speaking_time: float
    detection_delay: float | None = None
    """Latest input audio creation time -> prediction receive time."""
    inference_duration: float | None = None
    """Server-side model inference time."""
    backchannel_probability: float | None = None
    """How appropriate it is for the agent to backchannel at this pause.
    ``None`` when the detector does not produce one (e.g. the local mini model)."""


class _TurnDetector(Protocol):
    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None: ...
    async def supports_language(self, language: LanguageCode | None) -> bool: ...

    async def predict_end_of_turn(
        self, chat_ctx: ChatContext, *, timeout: float | None = None
    ) -> float: ...


@runtime_checkable
class _StreamingTurnDetectorStream(Protocol):
    """I/O stream for the streaming turn detector."""

    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...
    @property
    def is_fallback(self) -> bool: ...
    @property
    def prediction_timeout(self) -> float: ...

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None: ...
    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None: ...
    async def supports_language(self, language: LanguageCode | None) -> bool: ...

    def predict(self) -> asyncio.Future[TurnDetectionEvent]: ...
    def cancel_inference(self, *, timed_out: bool = False) -> None: ...
    def flush(self, reason: str | None = None) -> None: ...
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...
    def end_input(self) -> None: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class _StreamingTurnDetector(Protocol):
    """Turn detector that processes streaming data."""

    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _StreamingTurnDetectorStream: ...


TurnDetectionMode = (
    Literal["stt", "vad", "realtime_llm", "manual"] | _TurnDetector | _StreamingTurnDetector
)
"""
The mode of turn detection to use.

- "stt": use speech-to-text result to detect the end of the user's turn
- "vad": use VAD to detect the start and end of the user's turn
- "realtime_llm": use server-side turn detection provided by the realtime LLM
- "manual": manually manage the turn detection
- _TurnDetector: use the default mode with the provided turn detector

(default) If not provided, automatically choose the best mode based on
    available models (realtime_llm -> vad -> stt -> manual)
If the needed model (VAD, STT, or RealtimeModel) is not provided, fallback to the default mode.
"""


class EndpointingOptions(TypedDict, total=False):
    """Configuration for endpointing.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).
    """

    mode: Literal["fixed", "dynamic"]
    """Endpointing mode. ``"fixed"`` for fixed delay, ``"dynamic"`` for dynamic delay. Defaults to ``"fixed"``."""
    min_delay: float
    """Minimum time (s) since last detected speech before declaring the
    user's turn complete. Defaults to ``0.5``."""
    max_delay: float
    """Maximum time (s) the agent waits before terminating the turn.
    Defaults to ``3.0``."""
    alpha: float
    """Exponential moving average coefficient for dynamic endpointing.
    The higher the value, the more weight is given to the history.
    Defaults to ``0.9``. Only applies when mode is ``dynamic``."""


_ENDPOINTING_DEFAULTS: EndpointingOptions = {
    "mode": "fixed",
    "min_delay": 0.5,
    "max_delay": 3.0,
    "alpha": 0.9,
}

_STREAMING_ENDPOINTING_DEFAULTS: EndpointingOptions = {
    "mode": "fixed",
    "min_delay": 0.3,
    "max_delay": 2.5,
    "alpha": 0.9,
}


class InterruptionOptions(TypedDict, total=False):
    """Configuration for interruption handling.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).

    ``mode`` absent means the session picks the best available strategy.
    """

    enabled: bool
    """Whether interruptions are enabled. Defaults to ``True``."""
    mode: Literal["adaptive", "vad"]
    """Interruption handling strategy. ``"adaptive"`` for ML-based
    detection, ``"vad"`` for simple voice-activity detection.
    Absent means auto-detect."""
    discard_audio_if_uninterruptible: bool
    """Drop buffered audio while the agent speaks and cannot be
    interrupted. Defaults to ``True``."""
    min_duration: float
    """Minimum speech length (s) to register as an interruption.
    Defaults to ``0.5``."""
    min_words: int
    """Minimum word count to consider an interruption (STT only).
    Defaults to ``0``."""
    resume_false_interruption: bool
    """Resume the agent's speech after a false interruption.
    Defaults to ``True``."""
    false_interruption_timeout: float | None
    """Seconds of silence after an interruption before it is
    classified as false. ``None`` disables. Defaults to ``2.0``."""
    backchannel_boundary: float | tuple[float, float] | None
    """Seconds near the start/end of each agent turn during which overlapping
    speech classified as a backchannel by the adaptive detector is suppressed
    (events flagged as interruptions still pass through). Use a tuple to apply
    different values for start and end separately. ``None`` disables. Defaults
    to ``(1.0, 1.0)``. The end value preserves transcripts received near the
    end of agent speech."""


_INTERRUPTION_DEFAULTS: InterruptionOptions = {
    "enabled": True,
    "discard_audio_if_uninterruptible": True,
    "min_duration": 0.5,
    "min_words": 0,
    "resume_false_interruption": True,
    "false_interruption_timeout": 2.0,
    "backchannel_boundary": (1.0, 1.0),
}


class PreemptiveGenerationOptions(TypedDict, total=False):
    """Configuration for preemptive generation."""

    enabled: bool
    """Whether preemptive generation is enabled. Defaults to ``True``."""

    preemptive_tts: bool
    """Whether to also run TTS preemptively before the turn is confirmed.
    When ``False`` (default), only LLM runs preemptively; TTS starts once the
    turn is confirmed and the speech is scheduled."""

    max_speech_duration: float
    """Maximum user speech duration (s) for which preemptive generation
    is attempted. Beyond this threshold, preemptive generation is skipped
    since long utterances are more likely to change and users may expect
    slower responses. Defaults to ``10.0``."""

    max_retries: int
    """Maximum number of preemptive generation attempts per user turn.
    The counter resets when the turn completes. Defaults to ``3``."""


_PREEMPTIVE_GENERATION_DEFAULTS: PreemptiveGenerationOptions = {
    "enabled": True,
    "preemptive_tts": False,
    "max_speech_duration": 10.0,
    "max_retries": 3,
}


class UserTurnLimitOptions(TypedDict, total=False):
    """Configuration for detecting when a user has been speaking too long
    without the agent successfully responding.

    The framework tracks accumulated word count and wall-clock duration
    across consecutive user turns. Counters only reset when the agent
    transitions to ``speaking`` state (i.e., produces audio output).

    Both thresholds default to ``None`` (disabled). Set at least one to
    enable the feature.
    """

    max_words: int | None
    """Maximum accumulated word count before triggering. Uses the
    framework's WordTokenizer for counting. ``None`` disables word-based
    limiting. Defaults to ``None``."""

    max_duration: float | None
    """Maximum wall-clock duration (seconds) since the user first started
    speaking in the current accumulation window. ``None`` disables
    duration-based limiting. Defaults to ``None``."""


_USER_TURN_LIMIT_DEFAULTS: UserTurnLimitOptions = {
    "max_words": None,
    "max_duration": None,
}


class BackchannelOptions(TypedDict, total=False):
    """Configuration for the default ``Agent.backchannel_node``: short acknowledgements
    ("mm-hmm", "right") played while the user is still speaking, without taking the turn.

    Disabled by default. An opportunity arises at a pause the turn detector scores as
    one or, when the turn is owned by a realtime model's server-side detection or by
    manual control, at a pause reported by the client-side VAD (a ``vad`` instance must
    be set on the session). Every opportunity goes through ``Agent.backchannel_node``;
    these options drive the default implementation only. An agent that overrides the
    node decides on its own, from a persona or learned user preferences, and ignores
    the options unless it reads them. The chosen clip is played through the session's
    audio output as an interruptible speech that is never added to the chat context.
    """

    enabled: bool
    """Whether the default node backchannels. Defaults to ``False``."""
    audio: list[str]
    """Audio files to pick from at random, one per backchannel, in any format the
    session's codecs decode. Defaults to ``[]``, which plays nothing."""
    min_interval: float
    """Minimum seconds between two backchannels. Defaults to ``4.0``."""
    min_speech: float
    """Seconds of user speech since the agent last replied before the first
    backchannel. Defaults to ``2.0``."""
    max_per_turn: int
    """Backchannels at most since the agent last replied. Defaults to ``3``."""
    fire_probability: float
    """Chance to backchannel at an eligible pause, so the rhythm is not mechanical.
    Defaults to ``0.8``."""
    volume: float
    """Gain applied to the clip; a backchannel stays under the speaker's voice.
    Defaults to ``0.6``."""
    max_end_of_turn_probability: float
    """A turn-detector opportunity whose end-of-turn probability is above this
    value is skipped: a reply is imminent and would collide. Defaults to ``0.5``."""
    pause_window: float
    """Seconds after the VAD end of speech during which the pause is offered to the
    node again every ``pause_step`` seconds while the user stays silent, with
    ``pause_duration`` growing. ``0`` offers each pause once. Defaults to ``1.2``."""
    pause_step: float
    """Seconds between repeated offers of the same pause. Defaults to ``0.25``."""


_BACKCHANNEL_DEFAULTS: BackchannelOptions = {
    "enabled": False,
    "audio": [],
    "min_interval": 4.0,
    "min_speech": 2.0,
    "max_per_turn": 3,
    "fire_probability": 0.8,
    "volume": 0.6,
    "max_end_of_turn_probability": 0.5,
    "pause_window": 1.2,
    "pause_step": 0.25,
}


@dataclass
class UserEmotion:
    """What an ``Agent.emotion_node`` heard in the user's voice over its latest window.

    The vocabulary of ``label`` is the node's own; the dimensions are optional and
    ``0..1`` when given. An estimate, not a fact about the user."""

    label: str
    """The dominant label, in the node's vocabulary ("neutral", "sad", "angry", ...)."""
    confidence: float
    """How sure the node is of ``label``, ``0..1``."""
    scores: dict[str, float] = field(default_factory=dict)
    """Every label the node scored, when it scores more than one."""
    arousal: float | None = None
    """Calm (``0``) to excited (``1``), when the node estimates it."""
    valence: float | None = None
    """Negative (``0``) to positive (``1``), when the node estimates it."""
    window_duration: float = 0.0
    """Seconds of audio behind this estimate."""
    created_at: float = field(default_factory=time.time)


@dataclass
class BackchannelOpportunity:
    """A pause in the user's speech at which the agent may backchannel.

    Passed to ``Agent.backchannel_node``. The counters describe the user's current
    turn: everything since the agent's last reply ended."""

    source: Literal["vad", "turn_detector"]
    """What found the pause: the client-side VAD, or the turn detector's prediction."""
    user_speech_duration: float
    """Seconds of user speech since the agent's last reply."""
    count_in_turn: int
    """Backchannels already played since the agent's last reply."""
    since_last: float | None
    """Seconds since the previous backchannel, ``None`` before the first one."""
    end_of_turn_probability: float | None = None
    """The turn detector's end-of-turn probability at this pause, when it ran."""
    backchannel_probability: float | None = None
    """The turn detector's backchannel probability at this pause, when it ran."""
    language: str | None = None
    """The language the turn detector saw, when known."""
    user_emotion: UserEmotion | None = None
    """The latest estimate from ``Agent.emotion_node``, when one runs."""
    transcript: str = ""
    """What the user has said since the agent's last reply, as transcribed so far
    (final segments plus the current interim one); empty without a transcriber."""
    pause_duration: float = 0.0
    """Seconds the user has been silent at this opportunity. A VAD pause is offered
    again while the silence lasts (see ``BackchannelOptions.pause_window``), each time
    with the longer pause, so a node can hold a longer acknowledgement for a pause
    that has proven long."""
    created_at: float = field(default_factory=time.time)


@dataclass
class BackchannelResponse:
    """What ``Agent.backchannel_node`` returns when the agent should backchannel."""

    audio: str | AsyncIterable[rtc.AudioFrame]
    """A file path the session's codecs decode, or audio frames (pre-synthesized)."""
    volume: float = 1.0
    """Gain applied to the audio before playout."""
    text: str = ""
    """What the clip says, forwarded to the transcription output; empty for none."""


class TurnHandlingOptions(TypedDict, total=False):
    """Configuration for the turn handling system.

    Can be passed as a plain dict::

        AgentSession(
            turn_handling={
                "endpointing": {"min_delay": 0.3},
                "interruption": {"enabled": False},
                "preemptive_generation": {"preemptive_tts": True},
            },
        )

    All keys are optional and default to sensible values.
    """

    turn_detection: TurnDetectionMode | None
    """Strategy for deciding when the user has finished speaking.
    Absent means the session auto-selects."""
    endpointing: EndpointingOptions
    """Endpointing configuration. Defaults to ``{"min_delay": 0.5, "max_delay": 3.0}``."""
    interruption: InterruptionOptions
    """Interruption handling configuration. Use ``{"enabled": False}`` to disable."""
    preemptive_generation: PreemptiveGenerationOptions
    """Preemptive generation configuration. Use ``{"enabled": False}`` to disable."""
    user_turn_limit: UserTurnLimitOptions
    """User turn limit configuration. Use ``{"max_words": 50}`` to enable."""
    backchannel: BackchannelOptions
    """Options of the default ``Agent.backchannel_node``. Use
    ``{"enabled": True, "audio": [...]}`` to enable it."""


def _resolve_preemptive_generation(
    config: PreemptiveGenerationOptions | None = None,
) -> PreemptiveGenerationOptions:
    """Fill in defaults for missing keys."""
    if config is None:
        return PreemptiveGenerationOptions(**_PREEMPTIVE_GENERATION_DEFAULTS)
    return PreemptiveGenerationOptions(**{**_PREEMPTIVE_GENERATION_DEFAULTS, **config})


def _resolve_endpointing(
    config: EndpointingOptions | None = None,
    *,
    turn_detection: TurnDetectionMode | None = None,
) -> EndpointingOptions:
    """Fill in defaults for missing keys.

    When ``turn_detection`` is a streaming turn detector, keys the caller did
    not provide fall back to the tighter streaming defaults instead of the
    legacy ones."""
    base = (
        _STREAMING_ENDPOINTING_DEFAULTS
        if isinstance(turn_detection, _StreamingTurnDetector)
        else _ENDPOINTING_DEFAULTS
    )
    if config is None:
        return EndpointingOptions(**base)
    return EndpointingOptions(**{**base, **config})


def _resolve_interruption(
    config: InterruptionOptions | None = None,
) -> InterruptionOptions:
    """Fill in defaults for missing keys (``mode`` stays absent if not provided)."""
    if config is None:
        return InterruptionOptions(**_INTERRUPTION_DEFAULTS)
    return InterruptionOptions(**{**_INTERRUPTION_DEFAULTS, **config})


def _resolve_user_turn_limit(
    config: UserTurnLimitOptions | None = None,
) -> UserTurnLimitOptions:
    """Fill in defaults for missing keys."""
    if config is None:
        return UserTurnLimitOptions(**_USER_TURN_LIMIT_DEFAULTS)
    return UserTurnLimitOptions(**{**_USER_TURN_LIMIT_DEFAULTS, **config})


def _resolve_backchannel(
    config: BackchannelOptions | None = None,
) -> BackchannelOptions:
    """Fill in defaults for missing keys; the audio list is copied."""
    resolved = BackchannelOptions(**{**_BACKCHANNEL_DEFAULTS, **(config or {})})
    resolved["audio"] = list(resolved.get("audio") or [])
    return resolved


def _migrate_turn_handling(
    min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    false_interruption_timeout: NotGivenOr[float | None] = NOT_GIVEN,
    turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
    discard_audio_if_uninterruptible: NotGivenOr[bool] = NOT_GIVEN,
    min_interruption_duration: NotGivenOr[float] = NOT_GIVEN,
    min_interruption_words: NotGivenOr[int] = NOT_GIVEN,
    allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    resume_false_interruption: NotGivenOr[bool] = NOT_GIVEN,
    agent_false_interruption_timeout: NotGivenOr[float | None] = NOT_GIVEN,
    preemptive_generation: NotGivenOr[bool | PreemptiveGenerationOptions] = NOT_GIVEN,
) -> TurnHandlingOptions:
    """Build a TurnHandlingOptions from deprecated keyword arguments."""
    if is_given(agent_false_interruption_timeout):
        false_interruption_timeout = agent_false_interruption_timeout

    result: TurnHandlingOptions = {}

    # endpointing — only include keys that were explicitly provided
    endpointing_opts: EndpointingOptions = {}
    if is_given(min_endpointing_delay):
        endpointing_opts["min_delay"] = min_endpointing_delay
    if is_given(max_endpointing_delay):
        endpointing_opts["max_delay"] = max_endpointing_delay
    if endpointing_opts:
        result["endpointing"] = endpointing_opts

    # interruption — only include keys that were explicitly provided
    interruption: InterruptionOptions = {}
    if allow_interruptions is False:
        interruption["enabled"] = False
    if is_given(discard_audio_if_uninterruptible):
        interruption["discard_audio_if_uninterruptible"] = discard_audio_if_uninterruptible
    if is_given(min_interruption_duration):
        interruption["min_duration"] = min_interruption_duration
    if is_given(min_interruption_words):
        interruption["min_words"] = min_interruption_words
    if is_given(false_interruption_timeout):
        interruption["false_interruption_timeout"] = false_interruption_timeout
    if is_given(resume_false_interruption):
        interruption["resume_false_interruption"] = resume_false_interruption
    if interruption:
        result["interruption"] = interruption

    if is_given(turn_detection):
        result["turn_detection"] = turn_detection

    if isinstance(preemptive_generation, dict):
        result["preemptive_generation"] = PreemptiveGenerationOptions(**preemptive_generation)
    elif is_given(preemptive_generation):
        result["preemptive_generation"] = {"enabled": preemptive_generation}

    return result
