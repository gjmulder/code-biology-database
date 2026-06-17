"""A minimal OpenRouter agent (Python).

OpenRouter's official agent skill (openrouter.ai/skills/create-agent) ships a
TypeScript SDK. This project is Python (CLAUDE.md rule 3), so this module
implements the same agent pattern — a chat-completion client plus a multi-step
tool-calling loop with a step cap — directly against OpenRouter's
OpenAI-compatible HTTP API.

Design points carried from the skill:
  * model IDs are passed in, never hardcoded (they change frequently);
  * the agentic loop is bounded by ``max_steps`` (the skill's ``stepCountIs(n)``);
  * tools are plain callables wrapped with :func:`tool`.

Free OpenRouter models (e.g. ``nvidia/nemotron-3-ultra-550b-a55b:free``) are
rate-limited, so 429/5xx responses are retried with exponential backoff — that
rate limit, not local compute, is the throughput governor for the criterion-3
judge that consumes this client.
"""

import json
import logging
import os
import random
import time

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 600  # seconds; large-context calls can be slow


class OpenRouterError(RuntimeError):
    """Raised for missing credentials or non-recoverable HTTP failures."""


def tool(name, description, parameters, func):
    """Wrap a Python callable as an OpenAI-style tool definition.

    Returns a dict with the JSON schema the API expects plus the bound callable
    under ``_func`` so :meth:`OpenRouterClient.run_agent` can execute it.
    """
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
        "_func": func,
    }


class OpenRouterClient:
    """Thin client over OpenRouter's chat-completions endpoint."""

    def __init__(self, api_key=None, max_retries=4, base_url=BASE_URL,
                 referer="https://github.com/code-biology-database", title="code-biology-database"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.title = title

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional OpenRouter attribution headers (ranking / dashboards).
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

    def _build_body(self, model, messages, tools=None, response_format=None,
                    temperature=None, max_tokens=None, reasoning=None, provider=None):
        """Assemble the chat-completion request body, omitting unset options.

        ``reasoning`` (e.g. ``{"effort": "high"}``) and ``provider`` (OpenRouter routing
        preferences, e.g. pinning the implicit-caching DeepSeek first-party endpoint) are
        passed through verbatim. ``tools`` may carry the internal ``_func`` key — stripped here.
        """
        body = {"model": model, "messages": messages}
        if tools:
            body["tools"] = [_strip_func(t) for t in tools]
        if response_format is not None:
            body["response_format"] = response_format
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if reasoning is not None:
            body["reasoning"] = reasoning
        if provider is not None:
            body["provider"] = provider
        return body

    def _post(self, body):
        """POST a prepared body and return the full response JSON.

        Retries 429 and 5xx with exponential backoff (honouring ``Retry-After``); other 4xx
        are fatal.
        """
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            resp = requests.post(url, headers=self._headers(), json=body, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_retries:
                    self._sleep_before_retry(resp, attempt)
                    continue
                raise OpenRouterError(f"giving up after {attempt + 1} attempts: HTTP {resp.status_code}")
            raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    def call_model(self, model, messages, tools=None, response_format=None,
                   temperature=None, max_tokens=None, reasoning=None, provider=None):
        """POST a chat completion and return the assistant message dict."""
        body = self._build_body(model, messages, tools, response_format,
                                temperature, max_tokens, reasoning, provider)
        return self._post(body)["choices"][0]["message"]

    def call_model_usage(self, model, messages, tools=None, response_format=None,
                         temperature=None, max_tokens=None, reasoning=None, provider=None):
        """Like :meth:`call_model` but also return the response ``usage`` block.

        Returns ``(message, usage)``. ``usage`` carries ``prompt_tokens`` /
        ``completion_tokens`` and, on an implicit-caching provider, the
        ``prompt_tokens_details.cached_tokens`` needed to bill the cache-read discount.
        """
        body = self._build_body(model, messages, tools, response_format,
                                temperature, max_tokens, reasoning, provider)
        data = self._post(body)
        return data["choices"][0]["message"], (data.get("usage") or {})

    def _sleep_before_retry(self, resp, attempt):
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 2 ** attempt
        else:
            delay = 2 ** attempt
        delay += random.uniform(0, 0.5)  # jitter to de-correlate retries
        logger.warning("OpenRouter HTTP %d; retrying in %.1fs (attempt %d)",
                       resp.status_code, delay, attempt + 1)
        time.sleep(delay)

    def run_agent(self, model, instructions, user_input, tools=None, max_steps=5,
                  response_format=None, temperature=None):
        """Run a bounded tool-calling loop and return the final assistant message.

        Mirrors the skill's agentic loop: call the model, execute any requested
        tools, feed results back, repeat — up to ``max_steps`` tool rounds.
        """
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input},
        ]
        by_name = {t["function"]["name"]: t["_func"] for t in (tools or [])}

        for _ in range(max_steps):
            msg = self.call_model(model, messages, tools=tools,
                                  response_format=response_format, temperature=temperature)
            messages.append(_assistant_turn(msg))
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return msg
            for call in tool_calls:
                messages.append(self._execute_tool_call(call, by_name))

        # Step cap hit: do a final no-tools call so the model can answer plainly.
        return self.call_model(model, messages, response_format=response_format, temperature=temperature)

    def _execute_tool_call(self, call, by_name):
        fn_name = call["function"]["name"]
        raw_args = call["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
            logger.warning("tool %s got unparseable arguments: %r", fn_name, raw_args)
        func = by_name.get(fn_name)
        if func is None:
            result = {"error": f"unknown tool {fn_name}"}
        else:
            result = func(**args)
        return {
            "role": "tool",
            "tool_call_id": call.get("id"),
            "name": fn_name,
            "content": json.dumps(result),
        }


def _strip_func(t):
    """Return a tool dict without the internal ``_func`` callable."""
    return {k: v for k, v in t.items() if k != "_func"}


def _assistant_turn(msg):
    """Normalise an assistant message for appending to the running transcript."""
    turn = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        turn["tool_calls"] = msg["tool_calls"]
    return turn
