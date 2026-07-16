"""
Resolves real YES/NO token IDs and neg_risk off the Polymarket CLOB
REST API, instead of guessing them as condition_id + "_YES"/"_NO" like
the old placeholder did. That placeholder never worked against the real
exchange, token IDs are big ERC-1155 integers assigned by the CTF, not
derived from the condition_id string.

GET /markets/{condition_id} is public, no auth needed. Response has a
`tokens` list with one entry per outcome (each carrying token_id and
outcome), plus a top-level `neg_risk` bool and `minimum_tick_size`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ResolvedMarket:
    condition_id: str
    yes_token_id: str
    no_token_id: str
    neg_risk: bool
    tick_size: float


class PolymarketMarketResolver:
    def __init__(self, rest_url: str, session: aiohttp.ClientSession):
        self._rest_url = rest_url.rstrip("/")
        self._session = session
        self._log = logger.bind(component="poly_market_resolver")

    async def resolve(self, condition_id: str) -> Optional[ResolvedMarket]:
        url = f"{self._rest_url}/markets/{condition_id}"
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status != 200:
                    self._log.error(
                        "market_resolve_failed",
                        condition_id=condition_id,
                        status=resp.status,
                    )
                    return None
                data = await resp.json()
        except Exception as exc:
            self._log.error("market_resolve_error", condition_id=condition_id, error=str(exc))
            return None

        tokens = data.get("tokens", [])
        yes_id = no_id = None
        for t in tokens:
            outcome = str(t.get("outcome", "")).strip().lower()
            if outcome == "yes":
                yes_id = str(t.get("token_id"))
            elif outcome == "no":
                no_id = str(t.get("token_id"))

        if yes_id is None or no_id is None:
            self._log.error(
                "market_missing_tokens",
                condition_id=condition_id,
                tokens=tokens,
            )
            return None

        return ResolvedMarket(
            condition_id=condition_id,
            yes_token_id=yes_id,
            no_token_id=no_id,
            neg_risk=bool(data.get("neg_risk", False)),
            tick_size=float(data.get("minimum_tick_size", 0.01)),
        )
