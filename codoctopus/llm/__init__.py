# ---------------------------------------------------
# Codoctopus — LLM layer
#
# The public surface for talking to a model. Import from here, not from the
# provider modules, so swapping providers stays a configuration change.
# ---------------------------------------------------

from codoctopus.llm.base import (
    Provider,
    ProviderError,
    ProviderNotInstalled,
    StructuredOutputError,
)
from codoctopus.llm.registry import (
    available_providers,
    get_provider,
    list_models,
    parse_model_ref,
    register_provider,
)
from codoctopus.llm.types import (
    Completion,
    Message,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "Completion",
    "Message",
    "Provider",
    "ProviderError",
    "ProviderNotInstalled",
    "StopReason",
    "StructuredOutputError",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "available_providers",
    "get_provider",
    "list_models",
    "parse_model_ref",
    "register_provider",
]
