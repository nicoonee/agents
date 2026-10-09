"""Agent backchannels: ``turn_handling["backchannel"]`` resolves to the documented defaults
and stays off unless asked for; the default ``Agent.backchannel_node`` applies those options
to an opportunity; an agent that overrides the node decides on its own; a clip's gain is
applied before playout."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from livekit import rtc

from livekit.agents import (
    Agent,
    AgentSession,
    BackchannelOpportunity,
    BackchannelOptions,
    BackchannelResponse,
)
from livekit.agents.voice.agent_activity import _scale_volume
from livekit.agents.voice.turn import _BACKCHANNEL_DEFAULTS, _resolve_backchannel

pytestmark = pytest.mark.unit


def test_defaults_are_off() -> None:
    resolved = _resolve_backchannel(None)
    assert resolved == {**_BACKCHANNEL_DEFAULTS, "audio": []}
    assert resolved["enabled"] is False
    assert resolved["min_interval"] == 4.0
    assert resolved["min_speech"] == 2.0
    assert resolved["max_per_turn"] == 3
    assert resolved["fire_probability"] == 0.8
    assert resolved["volume"] == 0.6
    assert resolved["max_end_of_turn_probability"] == 0.5


def test_partial_config_fills_the_rest_and_copies_the_clips() -> None:
    clips = ["a.wav"]
    resolved = _resolve_backchannel(BackchannelOptions(enabled=True, audio=clips))
    clips.append("b.wav")
    assert resolved["enabled"] is True
    assert resolved["audio"] == ["a.wav"]
    assert resolved["min_interval"] == _BACKCHANNEL_DEFAULTS["min_interval"]


def test_session_exposes_backchannel_options() -> None:
    session = AgentSession(
        turn_handling={"backchannel": {"enabled": True, "audio": ["mhm.wav"], "min_interval": 6}}
    )
    opts = session.options.backchannel
    assert opts["enabled"] is True
    assert opts["audio"] == ["mhm.wav"]
    assert opts["min_interval"] == 6
    assert opts["min_speech"] == 2.0
    assert AgentSession().options.backchannel["enabled"] is False


def _agent_on(session: AgentSession) -> Agent:
    """An agent whose activity is a stand-in carrying only the session (the node reads
    nothing else from it)."""
    agent = Agent(instructions="")
    agent._activity = SimpleNamespace(_session=session)  # type: ignore[assignment]
    return agent


def _opportunity(**kw: object) -> BackchannelOpportunity:
    base: dict[str, object] = {
        "source": "vad",
        "user_speech_duration": 5.0,
        "count_in_turn": 0,
        "since_last": None,
    }
    base.update(kw)
    return BackchannelOpportunity(**base)  # type: ignore[arg-type]


def test_default_node_is_silent_when_disabled() -> None:
    agent = _agent_on(AgentSession(turn_handling={"backchannel": {"audio": ["mhm.wav"]}}))
    assert agent.backchannel_node(_opportunity()) is None


def test_default_node_applies_the_options() -> None:
    session = AgentSession(
        turn_handling={
            "backchannel": {
                "enabled": True,
                "audio": ["mhm.wav"],
                "fire_probability": 1.0,
                "volume": 0.5,
                "min_speech": 2.0,
                "min_interval": 4.0,
                "max_per_turn": 2,
                "max_end_of_turn_probability": 0.5,
            }
        }
    )
    agent = _agent_on(session)
    response = agent.backchannel_node(_opportunity())
    assert isinstance(response, BackchannelResponse)
    assert response.audio == "mhm.wav"
    assert response.volume == 0.5
    assert agent.backchannel_node(_opportunity(user_speech_duration=1.0)) is None
    assert agent.backchannel_node(_opportunity(since_last=3.0)) is None
    assert agent.backchannel_node(_opportunity(since_last=4.5)) is not None
    assert agent.backchannel_node(_opportunity(count_in_turn=2)) is None
    assert agent.backchannel_node(_opportunity(end_of_turn_probability=0.9)) is None
    assert agent.backchannel_node(_opportunity(end_of_turn_probability=0.2)) is not None


def test_default_node_never_fires_at_zero_probability() -> None:
    session = AgentSession(
        turn_handling={"backchannel": {"enabled": True, "audio": ["a.wav"], "fire_probability": 0}}
    )
    agent = _agent_on(session)
    assert all(agent.backchannel_node(_opportunity()) is None for _ in range(20))


def test_an_agent_overrides_the_node() -> None:
    class Persona(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="")
            self.seen: list[BackchannelOpportunity] = []

        def backchannel_node(
            self, opportunity: BackchannelOpportunity
        ) -> BackchannelResponse | None:
            self.seen.append(opportunity)
            if opportunity.count_in_turn:
                return None
            return BackchannelResponse(audio="aha.wav", volume=0.4, text="aha")

    agent = Persona()
    # the options stay off, the override decides regardless
    agent._activity = SimpleNamespace(_session=AgentSession())  # type: ignore[assignment]
    first = agent.backchannel_node(_opportunity())
    assert isinstance(first, BackchannelResponse)
    assert (first.audio, first.volume, first.text) == ("aha.wav", 0.4, "aha")
    assert agent.backchannel_node(_opportunity(count_in_turn=1)) is None
    assert len(agent.seen) == 2


def test_scale_volume_halves_the_samples() -> None:
    async def frames():  # type: ignore[no-untyped-def]
        data = np.full(480, 10000, dtype=np.int16)
        yield rtc.AudioFrame(
            data=data.tobytes(), sample_rate=48000, num_channels=1, samples_per_channel=480
        )

    async def run() -> list[rtc.AudioFrame]:
        return [f async for f in _scale_volume(frames(), 0.5)]

    (frame,) = asyncio.run(run())
    assert np.frombuffer(frame.data, dtype=np.int16).max() == 5000
    assert frame.sample_rate == 48000 and frame.samples_per_channel == 480


def test_default_emotion_node_is_off_and_the_opportunity_carries_the_estimate() -> None:
    from livekit.agents import UserEmotion

    async def frames():  # type: ignore[no-untyped-def]
        yield rtc.AudioFrame(data=b"\x00" * 960, sample_rate=48000, num_channels=1, samples_per_channel=480)

    assert Agent(instructions="").emotion_node(frames()) is None
    emotion = UserEmotion(label="sad", confidence=0.7, arousal=0.2)
    opportunity = _opportunity(user_emotion=emotion)
    assert opportunity.user_emotion is emotion
    assert _opportunity().user_emotion is None
