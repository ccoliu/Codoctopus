# ---------------------------------------------------
# Codoctopus — Provider registry
#
# Resolves a model string like "anthropic:claude-opus-5" to a Provider.
# Adapters are imported lazily so installing one vendor SDK is enough.
# ---------------------------------------------------

from __future__ import annotations

from typing import Any, Callable

from codoctopus.llm.base import Provider, ProviderError

#: provider name -> module path holding its Provider subclass
_BUILTIN: dict[str, tuple[str, str]] = {
    "anthropic": ("codoctopus.llm.providers.anthropic", "AnthropicProvider"),
    "openai": ("codoctopus.llm.providers.openai", "OpenAIProvider"),
    "gemini": ("codoctopus.llm.providers.gemini", "GeminiProvider"),
    "ollama": ("codoctopus.llm.providers.ollama", "OllamaProvider"),
}

_CUSTOM: dict[str, Callable[..., Provider]] = {}


def register_provider(name: str, factory: Callable[..., Provider]) -> None:
    """Add a provider so `get_provider("<name>:<model>")` can reach it."""
    _CUSTOM[name] = factory


def available_providers() -> list[str]:
    return sorted({*_BUILTIN, *_CUSTOM})


def parse_model_ref(ref: str) -> tuple[str, str | None]:
    """
    Split "provider:model" into its parts.

    A bare string with no colon is treated as a provider name, letting callers
    write "ollama" and take that provider's default model.
    """
    if ":" not in ref:
        return ref, None
    provider, model = ref.split(":", 1)
    return provider, model or None


def get_provider(ref: str, **options: Any) -> Provider:
    """Build the provider named by `ref`, e.g. "anthropic:claude-opus-5"."""
    name, model = parse_model_ref(ref)

    if name in _CUSTOM:
        factory: Callable[..., Provider] = _CUSTOM[name]
    else:
        if name not in _BUILTIN:
            raise ProviderError(
                f"Unknown provider '{name}'. Available: {', '.join(available_providers())}"
            )
        module_path, class_name = _BUILTIN[name]
        module = __import__(module_path, fromlist=[class_name])
        factory = getattr(module, class_name)

    # Construction can fail for the same vendor-SDK reasons a call can (e.g.
    # a client that validates credentials eagerly) — wrap those here too, for
    # the same reason complete() wraps them: a caller catching ProviderError
    # shouldn't see a raw SDK exception just because the failure happened one
    # step earlier than the request itself.
    try:
        return factory(model=model, **options)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"Could not construct provider '{name}': {exc}") from exc
