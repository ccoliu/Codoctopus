# ---------------------------------------------------
# Codoctopus — Domain registry
#
# Same lazy-import pattern as codoctopus.llm.registry: a name maps to a module
# path, imported only when actually requested. "coding" is registered here as
# a forward reference — codoctopus/domains/coding.py doesn't need to exist yet
# for this file to load, only at the moment get_domain("coding") is called.
# ---------------------------------------------------

from __future__ import annotations

from typing import Callable

from codoctopus.domains.base import Domain

#: domain name -> (module path, class name), imported lazily
_BUILTIN: dict[str, tuple[str, str]] = {
    "coding": ("codoctopus.domains.coding", "CodingDomain"),
    "research": ("codoctopus.domains.research", "ResearchDomain"),
}

_CUSTOM: dict[str, Callable[[], Domain]] = {}


def register_domain(name: str, factory: Callable[[], Domain]) -> None:
    """Add a domain so get_domain(name) can reach it."""
    _CUSTOM[name] = factory


def available_domains() -> list[str]:
    return sorted({*_BUILTIN, *_CUSTOM})


def get_domain(name: str) -> Domain:
    if name in _CUSTOM:
        return _CUSTOM[name]()

    if name not in _BUILTIN:
        raise ValueError(f"Unknown domain '{name}'. Available: {', '.join(available_domains())}")

    module_path, class_name = _BUILTIN[name]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)()
