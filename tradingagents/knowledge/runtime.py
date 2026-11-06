"""Worker configuration assembly.

The one non-obvious requirement: the ingestion worker must never route through
the ``knowledge_base`` vendor. If it did, a job would read the corpus's own
previous output, re-store it, and the corpus would refresh itself forever
without ever contacting a real vendor — stale data that looks perfectly fresh.
:func:`live_vendor_config` strips it from every vendor chain before the worker
installs its config.
"""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

KNOWLEDGE_VENDOR = "knowledge_base"


def _strip(chain: str) -> str:
    """Remove the corpus vendor from one comma-separated chain."""
    kept = [v.strip() for v in str(chain).split(",") if v.strip() and v.strip() != KNOWLEDGE_VENDOR]
    # An empty chain would raise in route_to_vendor; "default" means "any
    # available vendor", which is the right meaning for a stripped chain.
    return ",".join(kept) if kept else "default"


def live_vendor_config(config: dict) -> dict:
    """Copy of ``config`` with the corpus removed from all vendor routing."""
    out = copy.deepcopy(config)
    for key in ("data_vendors", "tool_vendors"):
        table = out.get(key)
        if isinstance(table, dict):
            out[key] = {name: _strip(chain) for name, chain in table.items()}
    return out


def build_worker_config(overrides: dict | None = None) -> dict:
    """Full worker config: defaults, then overrides, with the corpus stripped."""
    from tradingagents.default_config import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        config.update(overrides)
    return live_vendor_config(config)


def install(config: dict) -> None:
    """Install the worker config as the process-wide dataflows config.

    ``dataflows.config`` is a module-level global every vendor call reads, so
    the worker sets it once at startup — the same contract the graph follows.
    """
    from tradingagents.dataflows.config import set_config

    set_config(config)
