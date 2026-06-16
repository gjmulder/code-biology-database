"""Tests for openrouter_agent — a minimal Python OpenRouter client + tool loop.

Fully offline: the HTTP POST is monkeypatched. No real network, no API key
needed. Covers auth header construction, the chat-completion call, the
multi-step tool-calling loop, and 429/5xx retry with backoff.
"""

import json

import pytest

import openrouter_agent as oa


class FakeResp:
    """Stand-in for requests.Response."""

    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def _completion(content=None, tool_calls=None):
    """Build an OpenAI-shape chat-completion response."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}]}


# --- client construction ---------------------------------------------------

def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(oa.OpenRouterError):
        oa.OpenRouterClient()


def test_auth_header_is_bearer_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    client = oa.OpenRouterClient()
    assert client._headers()["Authorization"] == "Bearer sk-test-123"


# --- call_model ------------------------------------------------------------

def test_call_model_posts_and_returns_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["body"] = json
        return FakeResp(200, _completion(content="hello"))

    monkeypatch.setattr(oa.requests, "post", fake_post)
    client = oa.OpenRouterClient()
    msg = client.call_model("some/model", [{"role": "user", "content": "hi"}])

    assert msg["content"] == "hello"
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == "some/model"


def test_call_model_passes_response_format_and_tools(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    captured = {}
    monkeypatch.setattr(oa.requests, "post",
                        lambda url, headers, json, timeout: captured.update(body=json) or FakeResp(200, _completion("ok")))
    client = oa.OpenRouterClient()
    client.call_model("m", [{"role": "user", "content": "x"}],
                      tools=[{"type": "function"}], response_format={"type": "json_object"})
    assert captured["body"]["tools"] == [{"type": "function"}]
    assert captured["body"]["response_format"] == {"type": "json_object"}


# --- retry / backoff -------------------------------------------------------

def test_call_model_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(oa.time, "sleep", lambda *_: None)  # no real waiting
    responses = [FakeResp(429, headers={"Retry-After": "0"}), FakeResp(200, _completion("done"))]
    monkeypatch.setattr(oa.requests, "post", lambda *a, **k: responses.pop(0))
    client = oa.OpenRouterClient(max_retries=3)
    assert client.call_model("m", [{"role": "user", "content": "x"}])["content"] == "done"


def test_call_model_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(oa.time, "sleep", lambda *_: None)
    monkeypatch.setattr(oa.requests, "post", lambda *a, **k: FakeResp(503))
    client = oa.OpenRouterClient(max_retries=2)
    with pytest.raises(oa.OpenRouterError):
        client.call_model("m", [{"role": "user", "content": "x"}])


def test_4xx_other_than_429_does_not_retry(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    calls = []
    monkeypatch.setattr(oa.time, "sleep", lambda *_: None)

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResp(400, {"error": "bad request"})

    monkeypatch.setattr(oa.requests, "post", fake_post)
    client = oa.OpenRouterClient(max_retries=5)
    with pytest.raises(oa.OpenRouterError):
        client.call_model("m", [{"role": "user", "content": "x"}])
    assert len(calls) == 1  # 400 is fatal, no retry


# --- tool-calling loop -----------------------------------------------------

def test_run_agent_executes_tool_then_returns_final_text(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_time", "arguments": json.dumps({"tz": "UTC"})},
    }
    turns = [_completion(tool_calls=[tool_call]), _completion(content="it is noon")]
    monkeypatch.setattr(oa.requests, "post", lambda *a, **k: FakeResp(200, turns.pop(0)))

    seen = {}

    def get_time(tz):
        seen["tz"] = tz
        return {"time": "12:00"}

    client = oa.OpenRouterClient()
    tools = [oa.tool("get_time", "get the time", {"type": "object",
             "properties": {"tz": {"type": "string"}}}, get_time)]
    result = client.run_agent("m", "sys", "what time?", tools=tools, max_steps=5)

    assert seen["tz"] == "UTC"
    assert result["content"] == "it is noon"


def test_run_agent_honours_step_cap(monkeypatch):
    """If the model keeps calling tools, the loop stops at max_steps."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    tool_call = {"id": "c", "type": "function",
                 "function": {"name": "noop", "arguments": "{}"}}
    # Always returns a tool call -> would loop forever without the cap.
    monkeypatch.setattr(oa.requests, "post",
                        lambda *a, **k: FakeResp(200, _completion(tool_calls=[tool_call])))
    calls = []
    tools = [oa.tool("noop", "noop", {"type": "object", "properties": {}},
                     lambda: calls.append(1) or {"ok": True})]
    client = oa.OpenRouterClient()
    client.run_agent("m", "sys", "go", tools=tools, max_steps=3)
    assert len(calls) == 3  # exactly the cap, no infinite loop
