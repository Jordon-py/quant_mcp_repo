"""Kraken private execution adapter.

Only signed order-management calls live here; risk decisions happen before this
adapter is reached.
"""

from __future__ import annotations

import time

import httpx

from quant_mcp.adapters.kraken.signer import KrakenSigner


class KrakenPrivateClient:
    """Private execution adapter. Keep isolated from public data to reduce accidental coupling."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.kraken.com") -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")

    async def add_order(self, payload: dict[str, str]) -> dict:
        path = "/0/private/AddOrder"
        nonce = str(int(time.time() * 1000))
        body = {"nonce": nonce, **payload}
        # Kraken signs the path plus nonce/form body, not a JSON request body.
        signature = KrakenSigner.sign(path, nonce, body, self.api_secret)
        headers = {"API-Key": self.api_key, "API-Sign": signature}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(f"{self.base_url}{path}", data=body, headers=headers)
            response.raise_for_status()
            return response.json()

    async def cancel_all(self) -> dict:
        path = "/0/private/CancelAll"
        nonce = str(int(time.time() * 1000))
        body = {"nonce": nonce}
        # Recompute nonce/signature per request; Kraken rejects reused nonces.
        signature = KrakenSigner.sign(path, nonce, body, self.api_secret)
        headers = {"API-Key": self.api_key, "API-Sign": signature}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(f"{self.base_url}{path}", data=body, headers=headers)
            response.raise_for_status()
            return response.json()
