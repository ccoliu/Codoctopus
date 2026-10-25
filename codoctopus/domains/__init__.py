# ---------------------------------------------------
# Codoctopus — Domains
#
# Import from here, not from the individual domain modules.
# ---------------------------------------------------

from codoctopus.domains.base import Domain, VerifyResult, verify_plan_result
from codoctopus.domains.registry import available_domains, get_domain, register_domain

__all__ = ["Domain", "VerifyResult", "available_domains", "get_domain", "register_domain", "verify_plan_result"]
