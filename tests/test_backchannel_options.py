"""``turn_handling["backchannel"]`` resolves to a full option set with the documented
defaults, is off unless asked for, and keeps its own copy of the clip list."""

from __future__ import annotations

import pytest

from livekit.agents import AgentSession, BackchannelOptions
from livekit.agents.voice.turn import _BACKCHANNEL_DEFAULTS, _resolve_backchannel

pytestmark = pytest.mark.unit


def test_defaults_are_off() -> None:
    resolved = _resolve_backchannel(None)
    assert resolved == {**_BACKCHANNEL_DEFAULTS, "audio": []}
    assert resolved["enabled"] is False
    assert resolved["min_interval"] == 4.0
    assert resolved["min_speech"] == 2.0
    assert resolved["max_end_of_turn_probability"] == 0.5


def test_partial_config_fills_the_rest() -> None:
    resolved = _resolve_backchannel(BackchannelOptions(enabled=True, audio=["a.wav"]))
    assert resolved["enabled"] is True
    assert resolved["audio"] == ["a.wav"]
    assert resolved["min_interval"] == _BACKCHANNEL_DEFAULTS["min_interval"]


def test_audio_list_is_copied() -> None:
    clips = ["a.wav"]
    resolved = _resolve_backchannel(BackchannelOptions(audio=clips))
    clips.append("b.wav")
    assert resolved["audio"] == ["a.wav"]


def test_session_exposes_backchannel_options() -> None:
    session = AgentSession(
        turn_handling={"backchannel": {"enabled": True, "audio": ["mhm.wav"], "min_interval": 6}}
    )
    opts = session.options.backchannel
    assert opts["enabled"] is True
    assert opts["audio"] == ["mhm.wav"]
    assert opts["min_interval"] == 6
    assert opts["min_speech"] == 2.0

    default_session = AgentSession()
    assert default_session.options.backchannel["enabled"] is False
    assert default_session.options.backchannel["audio"] == []
