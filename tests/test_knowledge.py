from __future__ import annotations

import sqlite3
from pathlib import Path

from integrations.aura_persona_gateway.config import PersonaGatewayConfig
from integrations.aura_persona_gateway.runtime import AuraRuntimeConfig
from integrations.aura_persona_gateway.store import LilyPersonaStore
from integrations.aura_persona_gateway.turn import AuraPersonaGateway
from integrations.hermes_lily_cli.bridge import HermesLilyBridge, HermesLilyConfig


def _kb_gateway(tmp_path, **runtime_extra) -> AuraPersonaGateway:
    persona_home = tmp_path / "persona-home"
    (persona_home / "persona").mkdir(parents=True, exist_ok=True)
    config = PersonaGatewayConfig(
        enabled=True,
        persona_home=str(persona_home),
        companion_home=str(tmp_path / "companion-home"),
        hermes_home=str(tmp_path / "hermes-home"),
        admin_token="unit-test-token",
        include_debug_context=True,
    )
    store = LilyPersonaStore(config.companion_db_path)
    bridge = HermesLilyBridge(HermesLilyConfig(command=("hermes",), provider="base", model="base-model"))
    runtime = AuraRuntimeConfig(
        persona_home=config.persona_home,
        aura_model_mode="aura_model",
        aura_model_provider="stepfun",
        aura_model_model="step-3.5-flash",
        aura_model_base_url="https://api.stepfun.com/step_plan/v1",
        aura_model_api_key="stepfun-unit-key",
        aura_model_reasoning_effort="none",
        kb_qa_enabled=True,
        **runtime_extra,
    )
    return AuraPersonaGateway(config=config, store=store, bridge=bridge, runtime_config=runtime)


def _im_message_count(db_path: str) -> int:
    if not Path(db_path).exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT count(*) FROM companion_im_message").fetchone()[0])


def _patch_llm_forbidden(monkeypatch) -> None:
    class _BoomLlm:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("KB 问答走在线知识库应用，不应实例化 LLM 客户端")

    monkeypatch.setattr("integrations.aura_persona_gateway.turn.DirectLlmClient", _BoomLlm)


# ---------------------------------------------------------------------------
# 在线知识库应用后端（RAG 云端完成）
# ---------------------------------------------------------------------------


def _aliyun_runtime_extra() -> dict:
    return {
        "kb_backend": "aliyun_app",
        "kb_aliyun_endpoint": "https://llm-unit.cn-beijing.maas.aliyuncs.com/api/v2/apps/knowledge/chat",
        "kb_aliyun_api_key": "sk-unit-aliyun",
        "kb_aliyun_agent_id": "aid-unit",
        "kb_fallback_text": "这个问题我需要再参详，改日为你解答。",
    }


class _FakeAliyunClient:
    """模拟阿里云 KB 流式客户端。"""

    events: list = []

    def __init__(self, config) -> None:
        self.config = config

    def stream(self, user_text):
        yield from type(self).events

    def run(self, user_text):
        response = ""
        final = {}
        for event in self.stream(user_text):
            if event.get("type") == "delta":
                response += event.get("text") or ""
            else:
                final = dict(event)
        if response and not final.get("response"):
            final["response"] = response
        return final


def test_aliyun_kb_stream_forwards_deltas(tmp_path, monkeypatch) -> None:
    _patch_llm_forbidden(monkeypatch)
    _FakeAliyunClient.events = [
        {"type": "delta", "text": "南无"},
        {"type": "delta", "text": "阿弥陀佛。"},
        {"type": "final", "ok": True, "status": "completed", "response": "南无阿弥陀佛。",
         "evidence": {"route": "aliyun_kb", "latency_ms": 42}},
    ]
    monkeypatch.setattr("integrations.aura_persona_gateway.turn.AliyunKbClient", _FakeAliyunClient)
    gateway = _kb_gateway(tmp_path, **_aliyun_runtime_extra())

    events = list(
        gateway.run_direct_turn_stream("什么是净土？", metadata={"source": "aura-lily-gateway"})
    )
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["南无", "阿弥陀佛。"]
    payload = events[-1]["payload"]
    assert payload["ok"] is True
    assert payload["response"] == "南无阿弥陀佛。"
    evidence = payload["evidence"]
    assert evidence["route"] == "kb_qa"
    assert evidence["kb_backend"] == "aliyun_app"
    assert evidence["kb_hit"] is True
    assert _im_message_count(gateway.config.companion_db_path) == 2


def test_aliyun_kb_stream_failure_replies_fallback(tmp_path, monkeypatch) -> None:
    _patch_llm_forbidden(monkeypatch)
    _FakeAliyunClient.events = [
        {"type": "final", "ok": False, "status": "failed", "response": "",
         "evidence": {"error": "aliyun_kb_http_401"}},
    ]
    monkeypatch.setattr("integrations.aura_persona_gateway.turn.AliyunKbClient", _FakeAliyunClient)
    gateway = _kb_gateway(tmp_path, **_aliyun_runtime_extra())

    events = list(
        gateway.run_direct_turn_stream("什么是净土？", metadata={"source": "aura-lily-gateway"})
    )
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["这个问题我需要再参详，改日为你解答。"]
    payload = events[-1]["payload"]
    assert payload["ok"] is True  # 有可播报的回退话术，回合本身算完成
    assert payload["evidence"]["kb_hit"] is False
    assert payload["evidence"]["error"] == "aliyun_kb_http_401"


def test_aliyun_kb_inactive_without_credentials(tmp_path, monkeypatch) -> None:
    """缺 API Key 时 KB 问答模式不生效，回退普通人格对话。"""
    gateway = _kb_gateway(
        tmp_path,
        kb_backend="aliyun_app",
        kb_aliyun_endpoint="https://llm-unit.example/api",
        kb_aliyun_api_key="",
        kb_aliyun_agent_id="aid-unit",
    )
    assert gateway._kb_qa_active() is False


def test_aliyun_kb_non_stream_turn(tmp_path, monkeypatch) -> None:
    _patch_llm_forbidden(monkeypatch)
    _FakeAliyunClient.events = [
        {"type": "delta", "text": "一心念佛。"},
        {"type": "final", "ok": True, "status": "completed", "response": "一心念佛。",
         "evidence": {"route": "aliyun_kb"}},
    ]
    monkeypatch.setattr("integrations.aura_persona_gateway.turn.AliyunKbClient", _FakeAliyunClient)
    gateway = _kb_gateway(tmp_path, **_aliyun_runtime_extra())

    result = gateway.run_turn("如何修行？", metadata={"source": "aura-lily-gateway"})
    assert result.ok is True
    assert result.response == "一心念佛。"
    assert result.evidence["kb_backend"] == "aliyun_app"


def test_aliyun_kb_client_parses_cumulative_sse(monkeypatch) -> None:
    """客户端本体：兼容累积式 SSE（每条带全量文本）只发增量。"""
    from integrations.aura_persona_gateway.aliyun_kb import AliyunKbClient, AliyunKbConfig

    lines = [
        b'data: {"output":{"text":"\xe5\x8d\x97\xe6\x97\xa0"}}\n',
        b'data: {"output":{"text":"\xe5\x8d\x97\xe6\x97\xa0\xe9\x98\xbf\xe5\xbc\xa5\xe9\x99\x80\xe4\xbd\x9b"}}\n',
        b"data: [DONE]\n",
    ]

    class _FakeRes:
        def __iter__(self):
            return iter(lines)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "integrations.aura_persona_gateway.aliyun_kb.urlopen", lambda req, timeout: _FakeRes()
    )
    client = AliyunKbClient(
        AliyunKbConfig(endpoint="https://unit.example/chat", api_key="sk-unit", agent_id="aid-unit")
    )
    events = list(client.stream("test"))
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["南无", "阿弥陀佛"]
    final = events[-1]
    assert final["type"] == "final"
    assert final["ok"] is True
    assert final["response"] == "南无阿弥陀佛"
