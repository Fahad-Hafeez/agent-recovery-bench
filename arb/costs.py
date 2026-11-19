"""Deterministic token, latency and monetary accounting.

All costs are *synthetic but fixed*: a single published price sheet is applied
to token counts derived from the actual serialized payloads. This makes the
cost axis reproducible and comparable across policies, which is the only thing
ARB needs from it. It does not claim to predict any vendor's invoice.

The price sheet is versioned; leaderboard entries must state the sheet version
they were scored under (see ``docs/leaderboard_protocol.md``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PRICE_SHEET_VERSION = "arb-price-2026.11"


@dataclass(frozen=True)
class PriceSheet:
    """USD per token / per call / per millisecond of tool time."""

    input_token_usd: float = 2.5e-6
    output_token_usd: float = 10.0e-6
    tool_payload_token_usd: float = 0.5e-6
    tool_call_usd: float = 2.0e-4
    tool_ms_usd: float = 1.0e-7
    # Fixed orchestration overhead per step (context assembly, dispatch).
    step_overhead_usd: float = 1.0e-5
    # Baseline model latency per step, in ms.
    model_latency_ms: float = 240.0

    def as_dict(self) -> dict[str, float | str]:
        d = {"version": PRICE_SHEET_VERSION}
        d.update({k: getattr(self, k) for k in self.__dataclass_fields__})
        return d


PRICE = PriceSheet()


def _payload_tokens(obj: Any) -> int:
    """Token estimate from serialized length. Deterministic, dependency-free."""
    if obj is None:
        return 0
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, default=str)
    # ~4 characters per token is the standard rough estimator; we floor at 1 so
    # that every non-empty payload costs something.
    return max(1, len(text) // 4)


@dataclass
class StepCost:
    tokens_in: int
    tokens_out: int
    tool_payload_tokens: int
    latency_ms: float
    usd: float

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out + self.tool_payload_tokens

    def as_dict(self) -> dict[str, float]:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tool_payload_tokens": self.tool_payload_tokens,
            "tokens_total": self.tokens_total,
            "latency_ms": round(self.latency_ms, 3),
            "usd": round(self.usd, 8),
        }


def _jitter(key: str) -> float:
    """Deterministic pseudo-jitter in [0.85, 1.15) from a string key.

    Uses a stable 64-bit FNV-1a hash so results do not depend on PYTHONHASHSEED.
    """
    h = 0xCBF29CE484222325
    for b in key.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return 0.85 + 0.30 * ((h % 10_000) / 10_000.0)


def step_cost(
    *,
    tool: str,
    args: dict,
    result_payload: Any,
    step_index: int,
    base_latency_ms: float,
    jitter_key: str,
    price: PriceSheet = PRICE,
    context_tokens: int = 0,
) -> StepCost:
    """Account one agent step.

    ``context_tokens`` lets multi-agent steps carry the extra cost of a longer
    shared context (message history), which is a first-class part of the
    coordination overhead we want visible on the cost axis.
    """
    tokens_in = 700 + 40 * step_index + _payload_tokens(args) + context_tokens
    result_tokens = _payload_tokens(result_payload)
    tokens_out = 60 + result_tokens
    tool_tokens = result_tokens
    latency = price.model_latency_ms + base_latency_ms * _jitter(jitter_key)
    usd = (
        tokens_in * price.input_token_usd
        + tokens_out * price.output_token_usd
        + tool_tokens * price.tool_payload_token_usd
        + price.tool_call_usd
        + latency * price.tool_ms_usd
        + price.step_overhead_usd
    )
    return StepCost(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tool_payload_tokens=tool_tokens,
        latency_ms=latency,
        usd=usd,
    )
