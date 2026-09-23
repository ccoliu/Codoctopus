# ---------------------------------------------------
# Codoctopus — LLM layer tests
#
# These run without any API key: a scripted provider stands in for a vendor,
# which is exactly what the abstraction is for.
# ---------------------------------------------------

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from codoctopus.llm import (
    Completion,
    Message,
    Provider,
    ProviderError,
    StructuredOutputError,
    ToolCall,
    ToolResult,
    available_providers,
    get_provider,
    list_models,
    register_provider,
)
from codoctopus.llm.schema import to_strict_schema


class Worksheet(BaseModel):
    goal: str
    steps: list[str]


class ScriptedProvider(Provider):
    """Replays a fixed list of replies and records what it was asked."""

    name = "scripted"
    default_model = "scripted-1"

    def __init__(self, model: str | None = None, *, replies: list[str] | None = None, structured: bool = False, **options: Any) -> None:
        super().__init__(model, **options)
        self.replies = list(replies or [])
        self.structured = structured
        self.calls: list[dict[str, Any]] = []

    @property
    def supports_structured_output(self) -> bool:
        return self.structured

    async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs) -> Completion:
        self.calls.append({"messages": list(messages), "system": system, "tools": tools, "schema": output_schema})
        text = self.replies.pop(0) if self.replies else ""
        parsed = output_schema.model_validate_json(text) if output_schema and text else None
        return Completion(text=text, model=self.model, parsed=parsed)


# --- schema rendering ------------------------------------------------


def test_strict_schema_is_self_contained_and_closed():
    class Inner(BaseModel):
        value: int

    class Outer(BaseModel):
        name: str
        inner: Inner

    schema = to_strict_schema(Outer)

    assert "$defs" not in schema
    assert "$ref" not in str(schema), "nested models must be inlined, not referenced"
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["inner", "name"]
    assert schema["properties"]["inner"]["properties"]["value"]["type"] == "integer"


def test_strict_schema_keeps_a_free_form_dict_field_usable():
    """A dict[str, X] field is a mapping, not a fixed-shape object — its
    additionalProperties is the value schema and must not be clobbered to
    False, or the field becomes an object that may hold no keys at all."""

    class WithHeaders(BaseModel):
        headers: dict[str, str] = {}

    schema = to_strict_schema(WithHeaders)
    headers_schema = schema["properties"]["headers"]

    assert headers_schema["type"] == "object"
    assert headers_schema["additionalProperties"] == {"type": "string"}
    assert "required" not in headers_schema, "a mapping type has no fixed keys to require"


def test_strict_schema_survives_a_recursive_model():
    class Node(BaseModel):
        label: str
        child: "Node | None" = None

    Node.model_rebuild()
    schema = to_strict_schema(Node)

    assert "$ref" not in str(schema)
    assert schema["properties"]["label"]["type"] == "string"


# --- structured output -----------------------------------------------


async def test_native_structured_output_is_passed_through():
    provider = ScriptedProvider(replies=['{"goal": "ship", "steps": ["a", "b"]}'], structured=True)

    result = await provider.complete([Message.user("plan it")], output_schema=Worksheet)

    assert isinstance(result.parsed, Worksheet)
    assert result.parsed.steps == ["a", "b"]
    assert provider.calls[0]["schema"] is Worksheet, "native providers receive the schema"


async def test_fallback_parses_json_from_a_code_fence():
    provider = ScriptedProvider(replies=['```json\n{"goal": "ship", "steps": ["a"]}\n```'])

    result = await provider.complete([Message.user("plan it")], output_schema=Worksheet)

    assert result.parsed == Worksheet(goal="ship", steps=["a"])
    assert provider.calls[0]["schema"] is None, "fallback providers must not receive the schema"
    assert "JSON Schema" in provider.calls[0]["system"]


async def test_fallback_retries_with_the_validation_error():
    provider = ScriptedProvider(
        replies=['{"goal": "ship"}', '{"goal": "ship", "steps": ["a"]}']
    )

    result = await provider.complete([Message.user("plan it")], output_schema=Worksheet)

    assert result.parsed == Worksheet(goal="ship", steps=["a"])
    assert len(provider.calls) == 2
    retry_prompt = provider.calls[1]["messages"][-1].content
    assert "did not match the schema" in retry_prompt


async def test_fallback_gives_up_with_a_clear_error():
    provider = ScriptedProvider(replies=["not json", "still not json", "nope"])

    with pytest.raises(StructuredOutputError, match="schema-valid output after 3 attempts"):
        await provider.complete([Message.user("plan it")], output_schema=Worksheet, schema_retries=2)


async def test_a_tool_call_short_circuits_the_schema_fallback():
    """A model reaching for a tool is not yet done; do not force JSON on it."""

    class ToolThenText(ScriptedProvider):
        async def _complete(self, messages, *, system, tools, output_schema, max_tokens, **kwargs):
            self.calls.append({"messages": list(messages), "system": system, "tools": tools, "schema": output_schema})
            return Completion(
                text="", model=self.model, tool_calls=[ToolCall(id="c1", name="read", arguments={})]
            )

    provider = ToolThenText()
    result = await provider.complete([Message.user("go")], output_schema=Worksheet)

    assert result.wants_tools
    assert result.parsed is None
    assert len(provider.calls) == 1, "must not retry a turn that asked for a tool"


# --- vendor exceptions are wrapped into ProviderError ------------------


async def test_an_unexpected_exception_from_complete_is_wrapped_in_providererror():
    class Broken(ScriptedProvider):
        async def _complete(self, *args, **kwargs):
            raise TypeError("vendor SDK blew up for its own reasons")

    with pytest.raises(ProviderError, match="Provider 'scripted' call failed"):
        await Broken().complete([Message.user("go")])


async def test_an_unexpected_exception_from_the_schema_fallback_is_wrapped_too():
    class Broken(ScriptedProvider):
        async def _complete(self, *args, **kwargs):
            raise TypeError("vendor SDK blew up for its own reasons")

    with pytest.raises(ProviderError, match="Provider 'scripted' call failed"):
        await Broken(structured=False).complete([Message.user("go")], output_schema=Worksheet)


async def test_a_providererror_raised_by_complete_passes_through_unwrapped():
    class AlreadyClean(ScriptedProvider):
        async def _complete(self, *args, **kwargs):
            raise StructuredOutputError("already a clean ProviderError")

    with pytest.raises(StructuredOutputError, match="already a clean ProviderError"):
        await AlreadyClean().complete([Message.user("go")])


# --- registry ---------------------------------------------------------


def test_custom_provider_resolves_through_the_registry():
    register_provider("scripted", lambda model=None, **kw: ScriptedProvider(model, **kw))

    provider = get_provider("scripted:my-model")

    assert isinstance(provider, ScriptedProvider)
    assert provider.model == "my-model"


def test_bare_provider_name_takes_the_default_model():
    register_provider("scripted", lambda model=None, **kw: ScriptedProvider(model, **kw))

    assert get_provider("scripted").model == "scripted-1"


def test_unknown_provider_lists_what_is_available():
    with pytest.raises(Exception, match="Unknown provider 'nope'"):
        get_provider("nope:model")


def test_gateway_is_a_separate_builtin_provider_from_openai():
    assert "gateway" in available_providers()
    assert "openai" in available_providers()


def test_gateway_without_a_base_url_is_a_clear_providererror():
    with pytest.raises(ProviderError, match="base_url"):
        get_provider("gateway:some-local-model")


def test_an_exception_raised_while_constructing_a_provider_is_wrapped_too():
    def broken_factory(model=None, **kw):
        raise TypeError("vendor client validates credentials eagerly and blew up")

    register_provider("broken", broken_factory)

    with pytest.raises(ProviderError, match="Could not construct provider 'broken'"):
        get_provider("broken:model")


# --- list_models --------------------------------------------------------


async def test_list_models_dispatches_to_a_custom_providers_list_models():
    async def fake_list_models(**kw):
        return ["model-a", "model-b"]

    register_provider("scripted", lambda model=None, **kw: ScriptedProvider(model, **kw), list_models=fake_list_models)

    assert await list_models("scripted") == ["model-a", "model-b"]


async def test_list_models_passes_options_through():
    captured = {}

    async def fake_list_models(**kw):
        captured.update(kw)
        return []

    register_provider("scripted", lambda model=None, **kw: ScriptedProvider(model, **kw), list_models=fake_list_models)

    await list_models("scripted", api_key="secret", base_url="http://x")

    assert captured == {"api_key": "secret", "base_url": "http://x"}


async def test_list_models_wraps_an_unexpected_exception():
    async def broken_list_models(**kw):
        raise TypeError("vendor SDK blew up")

    register_provider("scripted", lambda model=None, **kw: ScriptedProvider(model, **kw), list_models=broken_list_models)

    with pytest.raises(ProviderError, match="Could not list models for 'scripted'"):
        await list_models("scripted")


async def test_list_models_for_a_provider_without_support_raises_cleanly():
    # A name no other test in this file registers a list_models for — reusing
    # "scripted" here would be order-dependent: register_provider only adds
    # to the list_models registry, it never clears a name's previous entry.
    register_provider("scripted-no-list-models", lambda model=None, **kw: ScriptedProvider(model, **kw))

    with pytest.raises(ProviderError, match="does not support listing models"):
        await list_models("scripted-no-list-models")


async def test_list_models_for_an_unknown_provider_raises_cleanly():
    with pytest.raises(ProviderError, match="Unknown provider 'nope'"):
        await list_models("nope")


# --- transcript round-trip -------------------------------------------


def test_completion_folds_back_into_the_transcript():
    completion = Completion(
        text="calling out",
        model="m",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
    )

    message = completion.to_message()

    assert message.role == "assistant"
    assert message.tool_calls[0].name == "read_file"


def test_tool_results_form_their_own_turn():
    message = Message.results([ToolResult(call_id="c1", content="ok")])

    assert message.role == "user"
    assert message.tool_results[0].call_id == "c1"
    assert message.content is None
