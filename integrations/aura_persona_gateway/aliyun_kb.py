"""阿里云百炼知识库应用（KB chat）客户端。

慈光版的问答后端：设备语音 → ASR → 本模块（阿里云知识库应用，RAG 由云端完成）
→ TTS。与本地 KB（KnowledgeStore + embedding 检索 + DirectLlm 生成）互斥，
由 runtime_config.kb_backend == "aliyun_app" 选择。

API 形态（阿里云大模型服务 MaaS 应用接口）::

    POST {endpoint}
    Authorization: Bearer <api_key>
    {
      "input": {"messages": [{"role": "user", "content": "..."}]},
      "parameters": {"agent_options": {"agent_id": "aid-..."}},
      "stream": true
    }

流式响应为 SSE data: 行。不同应用形态的 text 路径不完全一致，
`_extract_text` 做多路径兼容；并兼容“全量累积文本”与“增量文本”两种流。
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = ["AliyunKbConfig", "AliyunKbClient"]


@dataclass(frozen=True)
class AliyunKbConfig:
    endpoint: str
    api_key: str
    agent_id: str
    timeout_seconds: float = 30.0


def _extract_text(payload: Mapping[str, Any]) -> str:
    """从一条流式 payload 里尽力取出文本（多形态兼容）。"""
    output = payload.get("output")
    if isinstance(output, Mapping):
        text = output.get("text")
        if isinstance(text, str) and text:
            return text
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                    return message["content"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            delta = first.get("delta")
            if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
                return delta["content"]
            message = first.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return message["content"]
    data = payload.get("data")
    if isinstance(data, Mapping) and isinstance(data.get("text"), str):
        return data["text"]
    if isinstance(payload.get("text"), str):
        return payload["text"]
    return ""


def _extract_error(payload: Mapping[str, Any]) -> str:
    for key in ("code", "error_code"):
        code = payload.get(key)
        if code and str(code) not in {"200", "ok", "OK", "Success"}:
            message = payload.get("message") or payload.get("error_msg") or ""
            return f"{code}: {message}".strip(": ")
    error = payload.get("error")
    if isinstance(error, Mapping):
        return str(error.get("message") or error.get("code") or "unknown_error")
    return ""


def _iter_sse_payloads(response: Any) -> Iterator[Mapping[str, Any]]:
    for raw_line in response:
        line = bytes(raw_line or b"").strip()
        if not line:
            continue
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if line == b"[DONE]":
            break
        try:
            payload = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, Mapping):
            yield payload


def _latency_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


class AliyunKbClient:
    def __init__(self, config: AliyunKbConfig) -> None:
        self.config = config

    def _request(self, user_text: str, *, stream: bool) -> Request:
        body = {
            "input": {"messages": [{"role": "user", "content": str(user_text or "").strip()}]},
            "parameters": {"agent_options": {"agent_id": self.config.agent_id}},
            "stream": bool(stream),
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return Request(
            self.config.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=headers,
        )

    def stream(self, user_text: str) -> Iterator[dict[str, Any]]:
        """产出 {"type":"delta","text":...} 序列 + 结尾 {"type":"final",...}。

        兼容累积式流（每条 payload 带全量文本）：只 yield 新增后缀。
        """
        request_id = f"aliyun-kb-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        if not (self.config.endpoint and self.config.api_key and self.config.agent_id):
            yield {
                "type": "final",
                "ok": False,
                "status": "failed",
                "response": "",
                "evidence": {"error": "aliyun_kb_not_configured", "request_id": request_id},
            }
            return
        emitted = ""
        error_text = ""
        first_delta_ms = 0
        try:
            req = self._request(user_text, stream=True)
            with urlopen(req, timeout=max(1.0, float(self.config.timeout_seconds or 30.0))) as res:
                for payload in _iter_sse_payloads(res):
                    err = _extract_error(payload)
                    if err:
                        error_text = err
                        continue
                    text = _extract_text(payload)
                    if not text:
                        continue
                    if text.startswith(emitted) and len(text) > len(emitted):
                        delta = text[len(emitted):]   # 累积式流
                        emitted = text
                    elif text == emitted:
                        continue
                    else:
                        delta = text                   # 增量式流
                        emitted += text
                    if not first_delta_ms:
                        first_delta_ms = _latency_ms(started)
                    yield {"type": "delta", "text": delta}
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            yield {
                "type": "final",
                "ok": False,
                "status": "failed",
                "response": emitted,
                "evidence": {
                    "error": f"aliyun_kb_http_{exc.code}",
                    "detail": detail,
                    "request_id": request_id,
                    "latency_ms": _latency_ms(started),
                },
            }
            return
        except (OSError, URLError) as exc:
            yield {
                "type": "final",
                "ok": False,
                "status": "failed",
                "response": emitted,
                "evidence": {
                    "error": f"aliyun_kb_{exc.__class__.__name__}",
                    "request_id": request_id,
                    "latency_ms": _latency_ms(started),
                },
            }
            return
        ok = bool(emitted.strip()) and not error_text
        yield {
            "type": "final",
            "ok": ok,
            "status": "completed" if ok else "failed",
            "response": emitted.strip(),
            "evidence": {
                "route": "aliyun_kb",
                "request_id": request_id,
                "latency_ms": _latency_ms(started),
                "first_delta_ms": first_delta_ms,
                **({"error": error_text} if error_text else {}),
            },
        }

    def run(self, user_text: str) -> dict[str, Any]:
        """非流式：聚合 stream() 的结果。"""
        response_text = ""
        final: dict[str, Any] = {}
        for event in self.stream(user_text):
            if event.get("type") == "delta":
                response_text += str(event.get("text") or "")
            elif event.get("type") == "final":
                final = dict(event)
        if response_text.strip() and not str(final.get("response") or "").strip():
            final["response"] = response_text.strip()
        return final
