"""
EIP-712 signing for Polymarket CLOB V2 orders.

Polymarket cut over to V2 on 2026-04-28, hard break, no back-compat.
V1-signed orders get rejected outright and every open V1 order got wiped
at cutover. https://docs.polymarket.com/v2-migration

V2 Order struct (verified against docs.polymarket.com, 2026-06 rev):

    salt            uint256
    maker           address
    signer          address   same as maker for a plain EOA
    tokenId         uint256   YES or NO ERC-1155 token id
    makerAmount     uint256   6-decimal, pUSD or outcome tokens
    takerAmount     uint256   6-decimal, pUSD or outcome tokens
    side            uint8     0=BUY, 1=SELL
    signatureType   uint8     0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE, 3=POLY_1271
    timestamp       uint256   ms, uniqueness only, not an expiry
    metadata        bytes32   zero unless you're using it
    builder         bytes32   zero if no builder code

taker, expiration, nonce and feeRateBps are gone from the signed struct
in V2. expiration still shows up in the POST /order body for GTD orders,
it's just not part of what gets signed anymore.

verifyingContract depends on the market: regular markets use CTF Exchange
V2, neg-risk markets use a different contract entirely. Callers must pass
neg_risk (read it off the market/orderbook response) or you'll get a
validly-signed order against the wrong domain that the matching engine
just silently drops, no error, no fill, nothing.

Collateral is pUSD now, 6 decimals, same precision as outcome tokens
under V2 (V1 had outcome tokens at 18 decimals, that asymmetry is gone).
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional

from eth_account import Account
from eth_account.messages import encode_typed_data

import structlog

logger = structlog.get_logger(__name__)


# Constants
POLY_CHAIN_ID = 137                        # Polygon Mainnet

# V2 Exchange contracts (post 2026-04-28 cutover). Verify against
# https://docs.polymarket.com/resources/contracts before going live ,
# Polymarket has changed these once already and may again.
EXCHANGE_V2            = "0xE111180000d2663C0091e4f400237545B87B996B"
NEG_RISK_EXCHANGE_V2    = "0xe2222d279d744050d28e00520010520000310F59"

USDC_DECIMALS  = 6
TOKEN_DECIMALS = 6   # V2: outcome tokens are ALSO 6 decimals (not 18 like V1)

EXCHANGE_DOMAIN_NAME    = "Polymarket CTF Exchange"
EXCHANGE_DOMAIN_VERSION = "2"

# Typed data type definitions , V2 Order struct (11 signed fields)
ORDER_TYPES = {
    "EIP712Domain": [
        {"name": "name",              "type": "string"},
        {"name": "version",           "type": "string"},
        {"name": "chainId",           "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Order": [
        {"name": "salt",           "type": "uint256"},
        {"name": "maker",          "type": "address"},
        {"name": "signer",         "type": "address"},
        {"name": "tokenId",        "type": "uint256"},
        {"name": "makerAmount",    "type": "uint256"},
        {"name": "takerAmount",    "type": "uint256"},
        {"name": "side",           "type": "uint8"},
        {"name": "signatureType",  "type": "uint8"},
        {"name": "timestamp",      "type": "uint256"},
        {"name": "metadata",       "type": "bytes32"},
        {"name": "builder",        "type": "bytes32"},
    ],
}

ZERO_BYTES32 = "0x" + "00" * 32


# Enums
class OrderSide(IntEnum):
    BUY  = 0
    SELL = 1


class SignatureType(IntEnum):
    EOA          = 0   # standard wallet
    POLY_PROXY   = 1   # legacy Polymarket proxy wallet
    GNOSIS_SAFE  = 2   # legacy Gnosis Safe wallet
    POLY_1271    = 3   # deposit wallet (ERC-1271), recommended for new API users


# Order Definition
@dataclass
class OrderParams:
    """
    Human-readable order parameters (floating point).
    Signer converts to EVM raw integers before signing.
    """
    token_id: str          # YES or NO outcome ERC1155 token ID (string of uint256)
    side: OrderSide
    price: float            # probability [0, 1], i.e. price per outcome token in pUSD
    size: float              # number of outcome tokens (not collateral)
    neg_risk: bool = False   # MUST be read from the market's `neg_risk` field
    expiration: Optional[int] = None  # unix seconds; GTD wire-body field, NOT signed
    builder_code: str = ZERO_BYTES32  # bytes32, zero unless you have a builder code
    metadata: str = ZERO_BYTES32


@dataclass
class SignedOrder:
    """Ready-to-submit order struct for Polymarket CLOB V2 REST API."""
    salt:           str    # uint256 as string
    maker:          str    # checksum address
    signer:         str    # checksum address (same as maker for EOA)
    token_id:       str    # uint256 as string
    maker_amount:   str    # uint256 as string (6-dec)
    taker_amount:   str    # uint256 as string (6-dec)
    side:           int    # 0 or 1
    signature_type: int
    timestamp:      str    # uint256 ms, as string
    metadata:       str    # bytes32 hex
    builder:        str    # bytes32 hex
    signature:      str    # hex string "0x..."
    expiration:     str = "0"   # wire-body only field (GTD), not part of the signed struct

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize for Polymarket CLOB V2 POST /order endpoint."""
        return {
            "salt":          self.salt,
            "maker":         self.maker,
            "signer":        self.signer,
            "tokenId":       self.token_id,
            "makerAmount":   self.maker_amount,
            "takerAmount":   self.taker_amount,
            "side":          "BUY" if self.side == 0 else "SELL",
            "signatureType": self.signature_type,
            "timestamp":     self.timestamp,
            "metadata":      self.metadata,
            "builder":       self.builder,
            "expiration":    self.expiration,
            "signature":     self.signature,
        }


# Signer
class EIP712Signer:
    """Stateless EIP-712 V2 order signer."""

    def __init__(
        self,
        private_key: str,
        signature_type: SignatureType = SignatureType.EOA,
    ):
        """signature_type defaults to EOA. POLY_1271 (deposit wallets) needs
        ERC-7739-wrapped signatures we don't implement here, use the
        official py-clob-client-v2 for that."""
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        self._account = Account.from_key(private_key)
        self._sig_type = int(signature_type)
        self._log = logger.bind(
            address=self._account.address[:10] + "...",
            sig_type=int(signature_type),
        )
        self._log.info("eip712_signer_ready", chain_id=POLY_CHAIN_ID)

    @property
    def address(self) -> str:
        return self._account.address

    def sign_order(self, params: OrderParams) -> SignedOrder:
        """Build, encode and sign a V2 order, ready to POST to the CLOB."""
        salt        = self._generate_salt()
        timestamp_ms = int(time.time() * 1000)   # V2: uniqueness via timestamp, no nonce
        verifying_contract = NEG_RISK_EXCHANGE_V2 if params.neg_risk else EXCHANGE_V2

        maker_amount, taker_amount = self._compute_amounts(
            params.price, params.size, params.side,
        )

        domain = {
            "name": EXCHANGE_DOMAIN_NAME,
            "version": EXCHANGE_DOMAIN_VERSION,
            "chainId": POLY_CHAIN_ID,
            "verifyingContract": verifying_contract,
        }

        order_struct = {
            "salt":          salt,
            "maker":         self._account.address,
            "signer":        self._account.address,
            "tokenId":       int(params.token_id),
            "makerAmount":   maker_amount,
            "takerAmount":   taker_amount,
            "side":          int(params.side),
            "signatureType": self._sig_type,
            "timestamp":     timestamp_ms,
            "metadata":      params.metadata,
            "builder":       params.builder_code,
        }

        signable = encode_typed_data(
            domain_data=domain,
            message_types={"Order": ORDER_TYPES["Order"]},
            message_data=order_struct,
        )

        signed = self._account.sign_message(signable)

        self._log.debug(
            "order_signed",
            token_id=params.token_id[:8] + "...",
            side=params.side.name,
            price=round(params.price, 4),
            size=params.size,
            neg_risk=params.neg_risk,
            maker_amount=maker_amount,
            taker_amount=taker_amount,
        )

        sig_hex = signed.signature.hex()
        if not sig_hex.startswith("0x"):
            sig_hex = "0x" + sig_hex

        return SignedOrder(
            salt=str(salt),
            maker=self._account.address,
            signer=self._account.address,
            token_id=params.token_id,
            maker_amount=str(maker_amount),
            taker_amount=str(taker_amount),
            side=int(params.side),
            signature_type=self._sig_type,
            timestamp=str(timestamp_ms),
            metadata=params.metadata,
            builder=params.builder_code,
            expiration=str(params.expiration or 0),
            signature=sig_hex,
        )

    # Private helpers
    def _compute_amounts(
        self,
        price: float,
        size: float,
        side: OrderSide,
    ) -> tuple[int, int]:
        """
        Both amounts are 6-decimal under V2 (pUSD and outcome tokens match
        now, V1's 18-decimal token convention is gone).
        BUY:  makerAmount = pUSD (price*size), takerAmount = tokens (size)
        SELL: makerAmount = tokens (size), takerAmount = pUSD (price*size)
        """
        usdc_notional  = price * size
        token_notional = size

        usdc_raw  = int(round(usdc_notional  * 10 ** USDC_DECIMALS))
        token_raw = int(round(token_notional * 10 ** TOKEN_DECIMALS))

        if side == OrderSide.BUY:
            return usdc_raw, token_raw
        else:
            return token_raw, usdc_raw

    @staticmethod
    def _generate_salt() -> int:
        """Random 32-byte salt, don't be tempted to use a timestamp here."""
        return int.from_bytes(secrets.token_bytes(32), "big")
