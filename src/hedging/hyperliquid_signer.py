"""
Hyperliquid L1 action signing, the "phantom agent" scheme.

Hyperliquid doesn't EIP-712-sign the order JSON directly. It signs a
phantom Agent message whose connectionId is:

    keccak256(msgpack(action) + nonce.to_bytes(8, "big") + vault_byte [+ expiry])

vault_byte is 0x00 if there's no vault address, otherwise 0x01 followed
by the 20 raw address bytes. If you just EIP-712-sign the action dict
like a normal order, the exchange accepts the request and silently
rejects the signature, no error, it just doesn't fill. Scheme verified
against Hyperliquid's own hyperliquid-python-sdk reference implementation,
which is more precise on this than the prose in the docs.

The EIP-712 domain here is a fixed dummy, it has nothing to do with any
real deployed contract, it's just how L1 actions get routed through an
EVM signature:
    name: "Exchange", version: "1", chainId: 1337,
    verifyingContract: 0x0000000000000000000000000000000000000000
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import msgpack
import structlog
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak

logger = structlog.get_logger(__name__)

HL_DOMAIN = {
    "name": "Exchange",
    "version": "1",
    "chainId": 1337,
    "verifyingContract": "0x0000000000000000000000000000000000000000",
}

AGENT_TYPES = {
    "Agent": [
        {"name": "source", "type": "string"},
        {"name": "connectionId", "type": "bytes32"},
    ],
}


def _action_hash(
    action: Dict[str, Any],
    vault_address: Optional[str],
    nonce: int,
    expires_after: Optional[int] = None,
) -> bytes:
    # dict key order matters here, msgpack.packb doesn't sort keys, it
    # packs in insertion order, which has to match what the matching
    # engine expects. Don't rebuild `action` through anything that
    # might reorder it (like a set of kwargs).
    packed = msgpack.packb(action, use_bin_type=True)
    packed += nonce.to_bytes(8, "big")

    if vault_address is None:
        packed += b"\x00"
    else:
        addr = vault_address[2:] if vault_address.startswith("0x") else vault_address
        packed += b"\x01" + bytes.fromhex(addr)

    if expires_after is not None:
        packed += b"\x00" + expires_after.to_bytes(8, "big")

    return keccak(packed)


class HyperliquidSigner:
    """Signs L1 actions (orders, cancels) for POST /exchange."""

    def __init__(self, private_key: str, is_mainnet: bool = True):
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key
        self._account = Account.from_key(private_key)
        self._is_mainnet = is_mainnet
        self._log = logger.bind(address=self._account.address[:10] + "...")

    @property
    def address(self) -> str:
        return self._account.address

    def sign_action(
        self,
        action: Dict[str, Any],
        nonce: int,
        vault_address: Optional[str] = None,
        expires_after: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns the {r, s, v} signature dict the /exchange endpoint expects."""
        h = _action_hash(action, vault_address, nonce, expires_after)
        phantom_agent = {
            "source": "a" if self._is_mainnet else "b",
            "connectionId": h,
        }
        signable = encode_typed_data(
            domain_data=HL_DOMAIN,
            message_types=AGENT_TYPES,
            message_data=phantom_agent,
        )
        signed = self._account.sign_message(signable)
        return {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v,
        }

    @staticmethod
    def next_nonce() -> int:
        # HL wants a roughly-monotonic ms timestamp, not a sequential
        # counter, and rejects nonces too far from server time.
        return int(time.time() * 1000)
