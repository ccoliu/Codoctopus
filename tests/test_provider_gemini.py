# ---------------------------------------------------
# Codoctopus — GeminiProvider structured-output regression test
#
# Gemini's response_schema is its own OpenAPI-subset dialect: it rejects
# additionalProperties outright, so to_strict_schema's OpenAI/Ollama-style
# strict-JSON-Schema dict fails at the API with a cryptic "Unknown name
# additional_properties" error. The google-genai SDK converts a Pydantic
# model class to Gemini's dialect itself, so _complete must hand it the
# class, not a pre-rendered dict — this only mocks the network call, so it
# catches a regression back to passing a dict without needing a real API key.
# ---------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

from google.genai import types
from pydantic import BaseModel

from codoctopus.llm.providers.gemini import GeminiProvider, _to_gemini, list_models
from codoctopus.llm.types import Message, ToolSpec


class Worksheet(BaseModel):
    goal: str
    steps: list[str]


async def test_structured_output_passes_the_model_class_not_a_schema_dict(monkeypatch):
    provider = GeminiProvider(model="gemini-2.0-flash", api_key="fake-key")

    captured: dict = {}

    class FakeResponse:
        text = '{"goal": "g", "steps": []}'
        function_calls = None
        candidates = []
        usage_metadata = None

    async def fake_generate_content(*, model, contents, config):
        captured["config"] = config
        return FakeResponse()

    monkeypatch.setattr(provider._client.aio.models, "generate_content", fake_generate_content)

    await provider._complete(
        [Message.user("plan it")],
        system=None,
        tools=None,
        output_schema=Worksheet,
        max_tokens=100,
    )

    assert captured["config"].response_schema is Worksheet


async def test_tool_parameters_are_sanitized_for_gemini(monkeypatch):
    # additionalProperties and exclusiveMinimum are both real, live-observed
    # rejections: the SDK's own FunctionDeclaration(parameters=...) pydantic
    # validation raises "extra_forbidden" for either one locally (this test
    # would fail with a validation error, not just a wrong assertion, if the
    # sanitizing regressed) — exactly what to_strict_schema produces for a
    # tool with a `Field(gt=0)`-constrained argument, e.g. RunTestsTool's
    # timeout_seconds.
    provider = GeminiProvider(model="gemini-2.0-flash", api_key="fake-key")

    captured: dict = {}

    class FakeResponse:
        text = ""
        function_calls = None
        candidates = []
        usage_metadata = None

    async def fake_generate_content(*, model, contents, config):
        captured["config"] = config
        return FakeResponse()

    monkeypatch.setattr(provider._client.aio.models, "generate_content", fake_generate_content)

    tool = ToolSpec(
        name="run_tests",
        description="Run tests",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 600},
            },
            "required": ["path", "timeout_seconds"],
            "additionalProperties": False,
        },
    )

    await provider._complete(
        [Message.user("go")], system=None, tools=[tool], output_schema=None, max_tokens=100
    )

    # The SDK converts the dict into its own Schema object; the field that
    # matters is additional_properties staying unset (None) so it's omitted
    # from the JSON actually sent — a set value is what the live API rejects.
    params = captured["config"].tools[0].function_declarations[0].parameters
    assert params.additional_properties is None
    assert params.properties["path"].type == types.Type.STRING
    assert params.properties["timeout_seconds"].minimum == 0
    assert params.properties["timeout_seconds"].maximum == 600


class _FakeAsyncPage:
    """Minimal stand-in for genai's AsyncPager: only needs to support `async for`."""

    def __init__(self, items):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item


async def test_list_models_keeps_only_chat_capable_models_sorted(monkeypatch):
    async def fake_list(self, **kw):
        return _FakeAsyncPage(
            [
                SimpleNamespace(name="models/gemini-2.5-pro", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-2.0-flash", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"]),
            ]
        )

    monkeypatch.setattr("google.genai.models.AsyncModels.list", fake_list)

    assert await list_models(api_key="fake-key") == ["gemini-2.0-flash", "gemini-2.5-pro"]


# --- thought_signature round-trip -----------------------------------------
#
# Gemini rejects a later turn that replays a function call without the exact
# thought_signature bytes its own response attached to that call's Part —
# "Function call is missing a thought_signature in functionCall parts" (a
# real 400, seen live). response.function_calls loses this (it only exposes
# FunctionCall, not the Part around it), so _complete walks parts by hand.


async def test_a_function_calls_thought_signature_is_captured_and_replayed(monkeypatch):
    provider = GeminiProvider(model="gemini-2.5-pro", api_key="fake-key")
    signature_bytes = b"opaque-signature-bytes"

    fake_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(name="write_file", args={"path": "hello.py"}),
                            thought_signature=signature_bytes,
                        )
                    ],
                )
            )
        ]
    )

    async def fake_generate_content(*, model, contents, config):
        return fake_response

    monkeypatch.setattr(provider._client.aio.models, "generate_content", fake_generate_content)

    completion = await provider._complete(
        [Message.user("go")], system=None, tools=None, output_schema=None, max_tokens=100
    )

    assert len(completion.tool_calls) == 1
    call = completion.tool_calls[0]
    assert call.name == "write_file"
    assert call.extra["gemini_thought_signature"]  # non-empty base64 string

    # A later turn replaying this call must reproduce the exact original bytes.
    content = _to_gemini(Message.assistant(tool_calls=[call]), types)
    assert content.parts[0].thought_signature == signature_bytes


def test_a_tool_call_with_no_signature_still_converts():
    from codoctopus.llm.types import ToolCall

    call = ToolCall(id="c1", name="write_file", arguments={"path": "hello.py"})

    content = _to_gemini(Message.assistant(tool_calls=[call]), types)

    assert content.parts[0].function_call.name == "write_file"
    assert content.parts[0].thought_signature is None
