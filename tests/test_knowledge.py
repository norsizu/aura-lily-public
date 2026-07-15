from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from integrations.aura_persona_gateway.config import PersonaGatewayConfig
from integrations.aura_persona_gateway.knowledge import (
    EmbeddingClient,
    EmbeddingConfig,
    KnowledgeStore,
    chunk_text,
    default_kb_db_path,
    extract_text,
    file_type_of,
    kb_search,
    process_document,
)
from integrations.aura_persona_gateway.runtime import AuraRuntimeConfig
from integrations.aura_persona_gateway.store import LilyPersonaStore
from integrations.aura_persona_gateway.turn import AuraPersonaGateway
from integrations.hermes_lily_cli.bridge import HermesLilyBridge, HermesLilyConfig


def test_chunk_text_splits_on_paragraphs_with_overlap() -> None:
    paragraph_a = "第一段的内容。" * 20  # 140 chars
    paragraph_b = "第二段的内容。" * 20
    paragraph_c = "第三段的内容。" * 20
    text = f"{paragraph_a}\n\n{paragraph_b}\n\n{paragraph_c}"
    chunks = chunk_text(text, chunk_size=200, overlap=30)
    assert chunks
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert "第一段的内容" in chunks[0]
    # overlap: the second chunk carries the tail of the first
    if len(chunks) > 1:
        assert chunks[1][:10] in chunks[0] + "\n" + chunks[1]


def test_chunk_text_hard_cuts_oversize_paragraph() -> None:
    text = "无标点超长文本" * 200  # no sentence breaks, one paragraph
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) >= len(text) * 0.95


def test_chunk_text_empty_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_file_type_and_extract_text_txt() -> None:
    assert file_type_of("说明.TXT") == "txt"
    assert file_type_of("readme.md") == "md"
    assert file_type_of("data.pdf") == "pdf"
    assert file_type_of("word.docx") == "docx"
    assert file_type_of("evil.exe") == ""
    assert extract_text("a.txt", "你好世界".encode("utf-8")) == "你好世界"
    assert extract_text("a.md", b"\xef\xbb\xbfhello") == "hello"


def test_knowledge_store_kb_and_document_crud(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    kb = store.create_kb("测试库")
    assert kb["id"].startswith("kb-")
    assert store.get_kb(kb["id"])["name"] == "测试库"

    doc = store.create_document(kb["id"], filename="a.txt", file_type="txt")
    assert doc["status"] == "pending"
    store.set_document_status(doc["id"], status="ready", char_count=10, chunk_count=2)
    docs = store.list_documents(kb["id"])
    assert len(docs) == 1
    assert docs[0]["status"] == "ready"
    assert docs[0]["chunk_count"] == 2

    store.replace_chunks(doc["id"], kb["id"], ["甲", "乙"], [[1.0, 0.0], [0.0, 1.0]])
    kbs = store.list_kbs()
    assert kbs[0]["doc_count"] == 1
    assert kbs[0]["chunk_count"] == 2

    # cascade delete
    assert store.delete_kb(kb["id"]) is True
    assert store.list_kbs() == []
    assert store.list_documents(kb["id"]) == []
    assert store.search(kb["id"], [1.0, 0.0], top_k=5, score_threshold=0.0) == []


def test_store_search_returns_ranked_hits_above_threshold(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    kb = store.create_kb("向量库")
    doc = store.create_document(kb["id"], filename="v.txt", file_type="txt")
    vectors = [
        [1.0, 0.0, 0.0],  # identical to query -> score 1.0
        [1.0, 1.0, 0.0],  # cos = 0.707
        [0.0, 0.0, 1.0],  # orthogonal -> filtered
    ]
    store.replace_chunks(doc["id"], kb["id"], ["最相关", "次相关", "无关"], vectors)

    hits = store.search(kb["id"], [1.0, 0.0, 0.0], top_k=5, score_threshold=0.5)
    assert [hit["content"] for hit in hits] == ["最相关", "次相关"]
    assert math.isclose(hits[0]["score"], 1.0, abs_tol=1e-5)
    assert hits[0]["filename"] == "v.txt"

    top1 = store.search(kb["id"], [1.0, 0.0, 0.0], top_k=1, score_threshold=0.0)
    assert len(top1) == 1
    assert top1[0]["content"] == "最相关"


def test_embed_sends_task_only_for_jina_models(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_urlopen(req, timeout=0):
        import json as _json
        from io import BytesIO

        body = _json.loads(req.data.decode("utf-8"))
        captured.append(body)
        count = len(body["input"])
        payload = {"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(count)]}

        class _Res(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _Res(_json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("integrations.aura_persona_gateway.knowledge.urlopen", fake_urlopen)

    jina = EmbeddingClient(EmbeddingConfig(api_key="k", model="jina-embeddings-v3"))
    jina.embed(["你好"], task="retrieval.passage")
    assert captured[-1]["task"] == "retrieval.passage"
    jina.embed_query("问题")
    assert captured[-1]["task"] == "retrieval.query"

    other = EmbeddingClient(EmbeddingConfig(api_key="k", model="step-embedding"))
    other.embed(["你好"])
    assert "task" not in captured[-1]


def test_process_document_success_and_failure(tmp_path, monkeypatch) -> None:
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    kb = store.create_kb("文档库")
    doc = store.create_document(kb["id"], filename="a.txt", file_type="txt")

    class FakeEmbedder:
        def embed(self, texts, *, task="retrieval.passage"):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    result = process_document(
        store, FakeEmbedder(), doc_id=doc["id"], kb_id=kb["id"],
        filename="a.txt", raw="这是知识库测试内容。".encode("utf-8"),
    )
    assert result["ok"] is True
    assert store.get_document(doc["id"])["status"] == "ready"
    assert store.get_document(doc["id"])["chunk_count"] >= 1

    doc2 = store.create_document(kb["id"], filename="b.txt", file_type="txt")
    result2 = process_document(
        store, FakeEmbedder(), doc_id=doc2["id"], kb_id=kb["id"],
        filename="b.txt", raw=b"",
    )
    assert result2["ok"] is False
    assert store.get_document(doc2["id"])["status"] == "failed"
    assert store.get_document(doc2["id"])["error"]


def test_kb_search_roundtrip(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "kb.sqlite3")
    kb = store.create_kb("检索库")
    doc = store.create_document(kb["id"], filename="a.txt", file_type="txt")
    store.replace_chunks(doc["id"], kb["id"], ["命中内容"], [[1.0, 0.0]])

    class FakeEmbedder:
        def embed_query(self, text):
            return [1.0, 0.0]

    hits = kb_search(
        store, FakeEmbedder(), kb_id=kb["id"], query="问题",
        top_k=5, score_threshold=0.4,
    )
    assert hits and hits[0]["content"] == "命中内容"
    assert kb_search(
        store, FakeEmbedder(), kb_id=kb["id"], query="  ",
        top_k=5, score_threshold=0.4,
    ) == []


# ---------------------------------------------------------------- turn 级问答模式


class _FakeEmbeddingClient:
    def __init__(self, config) -> None:
        self.config = config

    def embed(self, texts, *, task="retrieval.passage"):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _kb_gateway(tmp_path, *, kb_active_id: str, **runtime_extra) -> AuraPersonaGateway:
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
        kb_active_id=kb_active_id,
        kb_embedding_api_key="jina-unit-key",
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
            raise AssertionError("KB 未命中时不应实例化 LLM 客户端")

    monkeypatch.setattr("integrations.aura_persona_gateway.turn.DirectLlmClient", _BoomLlm)


def test_kb_qa_stream_miss_replies_fallback_without_llm(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("integrations.aura_persona_gateway.turn.EmbeddingClient", _FakeEmbeddingClient)
    _patch_llm_forbidden(monkeypatch)
    gateway = _kb_gateway(tmp_path, kb_active_id="kb-does-not-exist")

    events = list(
        gateway.run_direct_turn_stream(
            "公司年假有几天？",
            metadata={"source": "aura-lily-gateway"},
        )
    )
    assert events[0]["type"] == "delta"
    assert events[0]["text"] == "我的知识库里没有相关的信息。"
    final = events[-1]
    assert final["type"] == "final"
    payload = final["payload"]
    assert payload["ok"] is True
    evidence = payload["evidence"]
    assert evidence["route"] == "kb_qa"
    assert evidence["kb_hit"] is False
    assert evidence["aura_model_billing_scope"] == "step_plan"
    # 用户 + aura 两条消息都要落库
    assert _im_message_count(gateway.config.companion_db_path) == 2


def test_kb_qa_stream_hit_streams_llm_answer(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("integrations.aura_persona_gateway.turn.EmbeddingClient", _FakeEmbeddingClient)

    class _FakeLlm:
        configs: list = []
        prompts: list = []

        def __init__(self, config) -> None:
            _FakeLlm.configs.append(config)

        def stream(self, prompt, *, metadata=None):
            _FakeLlm.prompts.append(prompt)
            yield {"type": "delta", "text": "年假有"}
            yield {"type": "delta", "text": "十天。"}
            yield {
                "type": "final",
                "ok": True,
                "status": "completed",
                "response": "年假有十天。",
                "evidence": {"provider": "stepfun"},
            }

    monkeypatch.setattr("integrations.aura_persona_gateway.turn.DirectLlmClient", _FakeLlm)

    persona_home = tmp_path / "persona-home"
    kstore = KnowledgeStore(default_kb_db_path(persona_home))
    kb = kstore.create_kb("单测库")
    doc = kstore.create_document(kb["id"], filename="hr.txt", file_type="txt")
    kstore.replace_chunks(doc["id"], kb["id"], ["公司年假是十天。"], [[1.0, 0.0]])

    gateway = _kb_gateway(tmp_path, kb_active_id=kb["id"])
    events = list(
        gateway.run_direct_turn_stream(
            "公司年假有几天？",
            metadata={"source": "aura-lily-gateway"},
        )
    )
    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert "".join(deltas) == "年假有十天。"
    payload = events[-1]["payload"]
    assert payload["ok"] is True
    assert payload["response"] == "年假有十天。"
    evidence = payload["evidence"]
    assert evidence["route"] == "kb_qa"
    assert evidence["kb_hit"] is True
    assert evidence["kb_id"] == kb["id"]
    assert evidence["kb_scores"] and evidence["kb_scores"][0] >= 0.9
    assert evidence["aura_model_billing_scope"] == "step_plan"
    # 提示词：<data> 包资料 + 用户问题
    prompt = _FakeLlm.prompts[-1]
    assert "<data>" in prompt and "公司年假是十天。" in prompt and "公司年假有几天？" in prompt
    # LLM 配置必须复用 Step Plan 且带 KB 严格 system prompt
    llm_config = _FakeLlm.configs[-1]
    assert llm_config.base_url == "https://api.stepfun.com/step_plan/v1"
    assert llm_config.max_tokens == 512
    assert "我的知识库里没有相关的信息。" in llm_config.system_prompt
    assert "先给结论" in llm_config.system_prompt
    assert _im_message_count(gateway.config.companion_db_path) == 2


def test_kb_qa_short_query_miss_replies_hint(tmp_path, monkeypatch) -> None:
    """短查询未命中时提示补充细节，而不是报“知识库里没有”。"""
    monkeypatch.setattr("integrations.aura_persona_gateway.turn.EmbeddingClient", _FakeEmbeddingClient)
    _patch_llm_forbidden(monkeypatch)
    gateway = _kb_gateway(tmp_path, kb_active_id="kb-does-not-exist")

    events = list(
        gateway.run_direct_turn_stream(
            "多少钱？",
            metadata={"source": "aura-lily-gateway"},
        )
    )
    assert events[0]["type"] == "delta"
    assert "具体一点" in events[0]["text"]
    evidence = events[-1]["payload"]["evidence"]
    assert evidence["kb_hit"] is False
    assert evidence["kb_short_query"] is True
    # 长问题未命中仍回标准兜底句
    events = list(
        gateway.run_direct_turn_stream(
            "请问公司年假制度一共有多少天呢？",
            metadata={"source": "aura-lily-gateway", "speculative": True},
        )
    )
    assert events[0]["text"] == "我的知识库里没有相关的信息。"


def test_kb_qa_short_query_expands_with_prefix(tmp_path, monkeypatch) -> None:
    """配置主语前缀后，短查询检索前自动拼前缀。"""

    class _RecordingEmbeddingClient(_FakeEmbeddingClient):
        queries: list = []

        def embed_query(self, text):
            _RecordingEmbeddingClient.queries.append(text)
            return [1.0, 0.0]

    monkeypatch.setattr("integrations.aura_persona_gateway.turn.EmbeddingClient", _RecordingEmbeddingClient)

    class _FakeLlm:
        def __init__(self, config) -> None:
            pass

        def stream(self, prompt, *, metadata=None):
            yield {"type": "delta", "text": "三百五。"}
            yield {"type": "final", "ok": True, "status": "completed", "response": "三百五。", "evidence": {}}

    monkeypatch.setattr("integrations.aura_persona_gateway.turn.DirectLlmClient", _FakeLlm)

    persona_home = tmp_path / "persona-home"
    kstore = KnowledgeStore(default_kb_db_path(persona_home))
    kb = kstore.create_kb("单测库")
    doc = kstore.create_document(kb["id"], filename="price.txt", file_type="txt")
    kstore.replace_chunks(doc["id"], kb["id"], ["墨灵上车价350元。"], [[1.0, 0.0]])

    gateway = _kb_gateway(tmp_path, kb_active_id=kb["id"], kb_query_prefix="墨灵")
    events = list(
        gateway.run_direct_turn_stream(
            "多少钱？",
            metadata={"source": "aura-lily-gateway", "speculative": True},
        )
    )
    assert _RecordingEmbeddingClient.queries[-1] == "墨灵多少钱？"
    evidence = events[-1]["payload"]["evidence"]
    assert evidence["kb_hit"] is True
    assert evidence["kb_query_expanded"] == "墨灵多少钱？"
    # 长查询不加前缀；已含前缀的短查询也不重复加
    list(gateway.run_direct_turn_stream("请问这台阅读器现在卖多少钱？", metadata={"speculative": True}))
    assert _RecordingEmbeddingClient.queries[-1] == "请问这台阅读器现在卖多少钱？"
    list(gateway.run_direct_turn_stream("墨灵多少钱", metadata={"speculative": True}))
    assert _RecordingEmbeddingClient.queries[-1] == "墨灵多少钱"


def test_kb_qa_speculative_turn_skips_message_save(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("integrations.aura_persona_gateway.turn.EmbeddingClient", _FakeEmbeddingClient)
    _patch_llm_forbidden(monkeypatch)
    gateway = _kb_gateway(tmp_path, kb_active_id="kb-does-not-exist")

    events = list(
        gateway.run_direct_turn_stream(
            "公司年假有几天？",
            metadata={"source": "aura-lily-gateway", "speculative": True},
        )
    )
    assert events[0]["text"] == "我的知识库里没有相关的信息。"
    payload = events[-1]["payload"]
    assert payload["evidence"]["speculative"] is True
    assert _im_message_count(gateway.config.companion_db_path) == 0


def test_kb_qa_run_turn_miss_and_error_paths(tmp_path, monkeypatch) -> None:
    _patch_llm_forbidden(monkeypatch)
    monkeypatch.setattr("integrations.aura_persona_gateway.turn.EmbeddingClient", _FakeEmbeddingClient)
    gateway = _kb_gateway(tmp_path, kb_active_id="kb-does-not-exist")
    result = gateway.run_turn("公司年假有几天？", metadata={"source": "aura-lily-gateway"})
    assert result.ok is True
    assert result.response == "我的知识库里没有相关的信息。"
    assert result.evidence["route"] == "kb_qa"
    assert result.evidence["kb_hit"] is False

    # embedding 失败 → 明确的"检索不可用"话术，而不是"没有资料"
    class _BrokenEmbeddingClient:
        def __init__(self, config) -> None:
            raise RuntimeError("embedding down")

    monkeypatch.setattr("integrations.aura_persona_gateway.turn.EmbeddingClient", _BrokenEmbeddingClient)
    result2 = gateway.run_turn("公司年假有几天？", metadata={"source": "aura-lily-gateway"})
    assert result2.ok is True
    assert result2.response == "知识库检索暂时不可用，请稍后再试。"
    assert result2.evidence["kb_error"]


# ---------------------------------------------------------------- server 级 KB 管理端点


def test_http_kb_admin_roundtrip(tmp_path, monkeypatch) -> None:
    import base64 as _base64
    import json as _json
    import threading
    import time as _time
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    monkeypatch.setenv("AURA_PERSONA_HOME", str(tmp_path / "persona-home"))
    monkeypatch.setenv("AURA_COMPANION_HOME", str(tmp_path / "companion-home"))
    monkeypatch.setenv("AURA_LILY_ADMIN_PASSWORD", "unit-pass")
    monkeypatch.setattr("integrations.hermes_lily_cli.server.EmbeddingClient", _FakeEmbeddingClient)

    from integrations.hermes_lily_cli.server import build_config, make_handler, parse_args

    handler = make_handler(build_config(parse_args([])))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{server.server_port}"
    auth = _base64.b64encode(b"admin:unit-pass").decode("ascii")
    headers = {"content-type": "application/json", "authorization": f"Basic {auth}"}

    def call(path: str, payload: dict | None = None) -> dict:
        data = _json.dumps(payload).encode("utf-8") if payload is not None else None
        req = Request(base + path, data=data, headers=headers, method="POST" if data is not None else "GET")
        with urlopen(req, timeout=5) as res:
            return _json.loads(res.read().decode("utf-8"))

    try:
        # 未登录一律 401
        try:
            urlopen(base + "/admin/kb/list", timeout=5)
        except HTTPError as exc:
            assert exc.code == 401
        else:  # pragma: no cover
            raise AssertionError("expected 401")

        # 先在后台保存 embedding key（复用现有 runtime 保存端点，返回脱敏）
        saved = call("/admin/aura/runtime", {"kb_embedding_api_key": "jina-unit-key"})
        assert saved["ok"] is True
        assert saved["config"]["kb_embedding_api_key_configured"] is True
        assert "jina-unit-key" not in _json.dumps(saved)

        kb = call("/admin/kb/create", {"name": "接口测试库"})["kb"]
        listing = call("/admin/kb/list")
        assert listing["ok"] is True
        assert [item["id"] for item in listing["kbs"]] == [kb["id"]]

        # 非法扩展名直接拒绝
        try:
            call("/admin/kb/upload", {
                "kb_id": kb["id"],
                "filename": "evil.exe",
                "content_base64": _base64.b64encode(b"boom").decode("ascii"),
            })
        except HTTPError as exc:
            assert exc.code == 400
        else:  # pragma: no cover
            raise AssertionError("expected 400")

        content = "Aura 是一个语音助手。\n\n它部署在树莓派上。"
        uploaded = call("/admin/kb/upload", {
            "kb_id": kb["id"],
            "filename": "intro.txt",
            "content_base64": _base64.b64encode(content.encode("utf-8")).decode("ascii"),
        })
        assert uploaded["ok"] is True
        doc_id = uploaded["doc"]["id"]

        # 后台线程处理，轮询到 ready
        status = ""
        for _ in range(50):
            docs = call(f"/admin/kb/docs?kb_id={kb['id']}")["docs"]
            status = docs[0]["status"]
            if status in {"ready", "failed"}:
                break
            _time.sleep(0.1)
        assert status == "ready"

        hits = call("/admin/kb/search", {"kb_id": kb["id"], "query": "Aura 是什么"})
        assert hits["ok"] is True
        assert hits["hits"] and "Aura" in hits["hits"][0]["content"]

        # 原始文件要落盘，供 reindex 使用
        raw_files = list((tmp_path / "persona-home" / "knowledge" / "files").glob(f"{doc_id}*"))
        assert len(raw_files) == 1

        reindexed = call("/admin/kb/doc/reindex", {"doc_id": doc_id})
        assert reindexed["ok"] is True
        for _ in range(50):
            docs = call(f"/admin/kb/docs?kb_id={kb['id']}")["docs"]
            if docs[0]["status"] in {"ready", "failed"}:
                break
            _time.sleep(0.1)
        assert docs[0]["status"] == "ready"

        assert call("/admin/kb/doc/delete", {"doc_id": doc_id})["ok"] is True
        assert list((tmp_path / "persona-home" / "knowledge" / "files").glob(f"{doc_id}*")) == []

        deleted = call("/admin/kb/delete", {"kb_id": kb["id"]})
        assert deleted["ok"] is True
        assert call("/admin/kb/list")["kbs"] == []
    finally:
        server.shutdown()
        thread.join(timeout=3)
