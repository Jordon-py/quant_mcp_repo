"""Kraken REST signature helper for private API calls."""

from __future__ import annotations

import base64
import hashlib
import hmac
import urllib.parse


class KrakenSigner:
    @staticmethod
    def sign(path: str, nonce: str, body: dict[str, str], api_secret: str) -> str:
        post_data = urllib.parse.urlencode(body)
        encoded = (nonce + post_data).encode()
        # Kraken's API-Sign is HMAC-SHA512(path + SHA256(nonce + postdata)).
        message = path.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()
