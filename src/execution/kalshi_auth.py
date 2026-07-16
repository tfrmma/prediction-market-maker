"""
Kalshi RSA-PSS request signing, shared between the WS feed (auth at
handshake) and the REST order endpoints (auth per request). Used to
live duplicated inside kalshi_feed.py, pulled out here once the order
manager needed the same signing logic.
"""
from __future__ import annotations

import base64

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiRsaSigner:
    def __init__(self, private_key_pem: str):
        self._key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None,
            backend=default_backend(),
        )

    def sign(self, timestamp_ms: int, method: str, path: str) -> str:
        """RSA-PSS SHA-256 over (timestamp_ms + method + path), base64-encoded.

        salt_length = digest size (32 bytes), per docs.kalshi.com's own
        quick-start example, not PSS.MAX_LENGTH like some third-party
        guides use, that fails verification.
        """
        msg = f"{timestamp_ms}{method}{path}".encode()
        sig = self._key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def headers(self, api_key_id: str, timestamp_ms: int, method: str, path: str) -> dict:
        return {
            "KALSHI-ACCESS-KEY": api_key_id,
            "KALSHI-ACCESS-SIGNATURE": self.sign(timestamp_ms, method, path),
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }
