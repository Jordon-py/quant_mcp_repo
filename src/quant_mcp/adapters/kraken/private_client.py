from __future__ import annotations

import time

import httpx

from quant_mcp.adapters.kraken.signer import KrakenSigner


class KrakenPrivateClient:
    """Private execution adapter. Keep isolated from public data to reduce accidental coupling."""

    base_url = "https://api.kraken.com"

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.api_key = api_key
        self.api_secret = api_secret

    async def add_order(self, payload: dict[str, str]) -> dict:
        path = "/0/private/AddOrder"
        nonce = str(int(time.time() * 1000))
        body = {"nonce": nonce, **payload}
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
        signature = KrakenSigner.sign(path, nonce, body, self.api_secret)
        headers = {"API-Key": self.api_key, "API-Sign": signature}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(f"{self.base_url}{path}", data=body, headers=headers)
            response.raise_for_status()
            return response.json()
