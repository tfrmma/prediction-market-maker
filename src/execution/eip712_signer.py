"""
src/execution/eip712_signer.py
───────────────────────────────
EIP-712 typed data signing for Polymarket CLOB orders.

Polymarket uses a custom EIP-712 domain on Polygon (chainId=137).
Every order sent to the CLOB must include a valid EIP-712 signature
over the order struct to prove wallet ownership and prevent replay attacks.

Order struct (from Polymarket CTF Exchange contract):
  Order {
    salt:           uint256
    maker:          address
    signer:         address
    taker:          address
    tokenId:        uint256    ← YES or NO token ID
    makerAmount:    uint256    ← collateral amount (USDC, 6 decimals)
    takerAmount:    uint256    ← outcome token amount (18 decimals)
    expiration:     uint256    ← unix timestamp
    nonce:          uint256
    feeRateBps:     uint256
    side:           uint8      ← 0=BUY, 1=SELL
    signatureType:  uint8      ← 0=EOA, 2=Poly proxy
  }
"""
from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional

from eth_account import Account
from eth_account.messages import encode_typed_data

import structlog

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

POLY_CHAIN_ID = 137                        # Polygon Mainnet
CTF_EXCHANGE  = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982e"  # Polymarket CTF Exchange

USDC_DECIMALS = 6
TOKEN_DECIMALS = 18

# EIP-712 domain
DOMAIN = {
    "name": "ClobAuthDomain",
    "version": "1",
    "chainId": POLY_CHAIN_ID,
    "verifyingContract": CTF_EXCHANGE,
}

# Typed data type definitions
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
        {"name": "taker",          "type": "address"},
        {"name": "tokenId",        "type": "uint256"},
        {"name": "makerAmount",    "type": "uint256"},
        {"name": "takerAmount",    "type": "uint256"},
        {"name": "expiration",     "type": "uint256"},
        {"name": "nonce",          "type": "uint256"},
        {"name": "feeRateBps",     "type": "uint256"},
        {"name": "side",           "type": "uint8"},
        {"name": "signatureType",  "type": "uint8"},
    ],
}

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class OrderSide(IntEnum):
    BUY  = 0
    SELL = 1


class SignatureType(IntEnum):
    EOA   = 0   # standard wallet
    POLY  = 2   # Polymarket proxy wallet


# ──────────────────────────────────────────────
# Order Definition
# ──────────────────────────────────────────────

@dataclass
class OrderParams:
    """
    Human-readable order parameters (floating point).
    Signer converts to EVM raw integers before signing.
    """
    token_id: str          # YES or NO token ERC1155 token ID (string of uint256)
    side: OrderSide
    price: float           # probability [0, 1], i.e. price per YES token in USDC
    size: float            # number of outcome tokens (not collateral)
    expiration: Optional[int] = None  # unix ts; None = no expiration (GTC-ish)
    fee_rate_bps: int = 0


@dataclass
class SignedOrder:
    """Ready-to-submit order struct for Polymarket CLOB REST API."""
    salt:           str    # uint256 as string
    maker:          str    # checksum address
    signer:         str    # checksum address (same as maker for EOA)
    taker:          str    # zero address (open order)
    token_id:       str    # uint256 as string
    maker_amount:   str    # uint256 as string (USDC, 6 dec)
    taker_amount:   str    # uint256 as string (outcome tokens, 18 dec)
    expiration:     str    # uint256 as string
    nonce:          str    # uint256 as string
    fee_rate_bps:   str    # uint256 as string
    side:           int    # 0 or 1
    signature_type: int    # 0 or 2
    signature:      str    # hex string "0x..."

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize for Polymarket CLOB REST endpoint."""
        return {
            "salt":          self.salt,
            "maker":         self.maker,
            "signer":        self.signer,
            "taker":         self.taker,
            "tokenId":       self.token_id,
            "makerAmount":   self.maker_amount,
            "takerAmount":   self.taker_amount,
            "expiration":    self.expiration,
            "nonce":         self.nonce,
            "feeRateBps":    self.fee_rate_bps,
            "side":          str(self.side),
            "signatureType": self.signature_type,
            "signature":     self.signature,
        }


# ──────────────────────────────────────────────
# Signer
# ──────────────────────────────────────────────

class EIP712Signer:
    """
    Stateless EIP-712 order signer.

    Usage:
        signer = EIP712Signer(private_key="0x...")
        signed = signer.sign_order(params)
        payload = signed.to_api_dict()
    """

    def __init__(
        self,
        private_key: str,
        signature_type: SignatureType = SignatureType.EOA,
    ):
        """
        Parameters
        ----------
        private_key     : Hex private key, with or without "0x" prefix.
        signature_type  : EOA (0) for standard wallets, POLY (2) for proxy.
        """
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        self._account = Account.from_key(private_key)
        self._sig_type = int(signature_type)
        self._nonce    = 0  # monotonically increasing per session
        self._log = logger.bind(
            address=self._account.address[:10] + "...",
            sig_type=int(signature_type),
        )
        self._log.info("eip712_signer_ready", chain_id=POLY_CHAIN_ID)

    @property
    def address(self) -> str:
        return self._account.address

    def sign_order(self, params: OrderParams) -> SignedOrder:
        """
        Construct, encode, and sign an EIP-712 order.
        Returns a SignedOrder ready to submit to the CLOB.
        """
        salt        = self._generate_salt()
        nonce       = self._next_nonce()
        expiration  = params.expiration or 0

        # Convert floats to on-chain integers
        maker_amount, taker_amount = self._compute_amounts(
            params.price,
            params.size,
            params.side,
        )

        order_struct = {
            "salt":          salt,
            "maker":         self._account.address,
            "signer":        self._account.address,
            "taker":         ZERO_ADDRESS,
            "tokenId":       int(params.token_id),
            "makerAmount":   maker_amount,
            "takerAmount":   taker_amount,
            "expiration":    expiration,
            "nonce":         nonce,
            "feeRateBps":    params.fee_rate_bps,
            "side":          int(params.side),
            "signatureType": self._sig_type,
        }

        # EIP-712 encode and sign
        signable = encode_typed_data(
            domain_data=DOMAIN,
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
            maker_amount=maker_amount,
            taker_amount=taker_amount,
        )

        return SignedOrder(
            salt=str(salt),
            maker=self._account.address,
            signer=self._account.address,
            taker=ZERO_ADDRESS,
            token_id=params.token_id,
            maker_amount=str(maker_amount),
            taker_amount=str(taker_amount),
            expiration=str(expiration),
            nonce=str(nonce),
            fee_rate_bps=str(params.fee_rate_bps),
            side=int(params.side),
            signature_type=self._sig_type,
            signature=signed.signature.hex(),
        )

    # ── Private helpers ───────────────────────

    def _compute_amounts(
        self,
        price: float,
        size: float,
        side: OrderSide,
    ) -> tuple[int, int]:
        """
        Polymarket CLOB amount semantics:
          BUY:  makerAmount = USDC collateral  (price × size)
                takerAmount = outcome tokens    (size)
          SELL: makerAmount = outcome tokens    (size)
                takerAmount = USDC collateral  (price × size)

        USDC has 6 decimals; outcome tokens have 18 decimals.
        Price is in [0, 1]; we treat it as cents per 100 (USDC per token).
        """
        # 1 outcome token = 1 USDC at $1.00 payout
        # price = probability = USDC per outcome token
        usdc_notional    = price * size   # USDC
        token_notional   = size           # outcome tokens

        usdc_raw  = int(round(usdc_notional  * 10 ** USDC_DECIMALS))
        token_raw = int(round(token_notional * 10 ** TOKEN_DECIMALS))

        if side == OrderSide.BUY:
            return usdc_raw, token_raw   # makerAmount=USDC, takerAmount=tokens
        else:
            return token_raw, usdc_raw   # makerAmount=tokens, takerAmount=USDC

    @staticmethod
    def _generate_salt() -> int:
        """
        Cryptographically random 32-byte salt.
        Do NOT use timestamp-based salts in production.
        """
        return int.from_bytes(secrets.token_bytes(32), "big")

    def _next_nonce(self) -> int:
        """
        Simple incrementing nonce.
        In production, should be persisted to survive restarts.
        """
        self._nonce += 1
        return self._nonce
