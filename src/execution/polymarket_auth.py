"""
Polymarket CLOB L2 (HMAC) request auth. This didn't exist before,
order_manager.py was posting to /order with zero auth headers, which
would just 401 against the real exchange.

L1 is the EIP-712 wallet signature you do once to mint L2 creds via
POST /auth/api-key (not implemented here, assumed to already exist in
config). L2 is HMAC-SHA256 over (timestamp + METHOD + path + body),
carried in POLY_ADDRESS / POLY_TIMESTAMP / POLY_API_KEY / POLY_PASSPHRASE
/ POLY_SIGNATURE headers on every authenticated request.

POLY_TIMESTAMP is unix seconds, not ms. Easy to get wrong since the V2
order struct's own timestamp field IS in ms.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PolyL2Credentials:
    api_key: str
    secret: str        # base64, from POST /auth/api-key
    passphrase: str
    address: str        # wallet these creds were derived for


class PolyL2Auth:
    """Builds the five POLY_* headers for authenticated CLOB requests."""

    def __init__(self, creds: PolyL2Credentials):
        self._creds = creds
        self._secret_bytes = base64.urlsafe_b64decode(self._pad_b64(creds.secret))

    def headers(self, method: str, path: str, body: Optional[str] = None) -> Dict[str, str]:
        """path is request path only ("/order"), no host or query string.
        body is the exact JSON string being sent, empty for GET/DELETE."""
        ts = str(int(time.time()))   # seconds, not ms
        message = ts + method.upper() + path + (body or "")
        sig = hmac.new(self._secret_bytes, message.encode(), hashlib.sha256).digest()
        signature = base64.urlsafe_b64encode(sig).decode()

        return {
            "POLY_ADDRESS": self._creds.address,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": ts,
            "POLY_API_KEY": self._creds.api_key,
            "POLY_PASSPHRASE": self._creds.passphrase,
        }

    @staticmethod
    def _pad_b64(s: str) -> str:
        return s + "=" * (-len(s) % 4)
