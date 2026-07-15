from __future__ import annotations

import asyncio
import json
import time

from integrations.aura_persona_gateway.runtime import AuraRuntimeConfig
from integrations.hermes_lily_cli import gateway


def test_knowledge_stream_bypasses_persona_guard() -> None:
    answer = "电池与充电：超薄版支持快充和反向充电。"

    assert gateway.stream_bridge_response_needs_guard(answer, transcript="型号有什么不一样？")
    assert not gateway.stream_bridge_response_requires_guard(
        answer,
        transcript="型号有什么不一样？",
        knowledge_stream=True,
    )


def test_persona_stream_keeps_unfounded_claim_guard() -> None:
    answer = "你最近一直赶项目，整个人有点乱。"

    assert gateway.stream_bridge_response_requires_guard(
        answer,
        transcript="我想聊聊。",
        knowledge_stream=False,
    )


def test_stepfun_finish_allows_healthy_synthesis_longer_than_timeout(tmp_path) -> None:
    device_frames: list[bytes | str] = []

    class DeviceWebsocket:
        async def send(self, payload):
            device_frames.append(payload)

    class StepWebsocket:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    async def scenario():
        session = gateway.StepfunWsTtsSession(
            DeviceWebsocket(),
            AuraRuntimeConfig(
                persona_home=str(tmp_path / "persona-home"),
                tts_enabled=True,
                tts_provider="stepfun",
                tts_model="stepaudio-2.5-tts",
                tts_voice="voice-tone-test",
                tts_base_url="https://api.stepfun.com/step_plan/v1",
                tts_api_key="unit-key",
            ),
            turn_id=9,
            stream_id=1,
            started=time.monotonic(),
        )
        step_ws = StepWebsocket()
        session._step_ws = step_ws
        session.session_id = "sid-long-answer"
        session.timeout = 0.01

        async def healthy_long_receiver() -> None:
            await asyncio.sleep(0.03)
            session.audio_bytes = 4096
            session.done_seen = True

        session._receiver_task = asyncio.create_task(healthy_long_receiver())
        result = await session.finish()
        return result, step_ws.sent

    result, sent = asyncio.run(scenario())

    assert result.ok is True
    assert result.detail == ""
    assert result.audio_bytes == 4096
    assert sent[-1]["type"] == "tts.text.done"
    assert any(isinstance(frame, bytes) and frame[12] == 1 for frame in device_frames)
