# -*- coding: utf-8 -*-
"""HTTP client configuration."""

import httpx
from app.config import PROXY_URL, SSL_CERT_FILE


def build_http_client() -> httpx.Client:
    """Build configured HTTP client with proxy support."""
    verify = SSL_CERT_FILE
    if PROXY_URL:
        try:
            # Use 'all://' to catch all protocols
            # Use separate transports for proxy to ensure retries work
            proxy_transport = httpx.HTTPTransport(proxy=PROXY_URL, retries=3)
            return httpx.Client(
                transport=proxy_transport,
                timeout=60,
                verify=verify,
            )
        except (TypeError, ValueError) as e:
            print(f"[WARNING] Failed to configure proxy: {e}")
            transport = httpx.HTTPTransport(retries=3)
            return httpx.Client(timeout=60, verify=verify, transport=transport)
    transport = httpx.HTTPTransport(retries=3)
    return httpx.Client(timeout=60, verify=verify, transport=transport)


# Global HTTP client instance
_HTTP_CLIENT = build_http_client()


def get_http_client() -> httpx.Client:
    """Get the global HTTP client instance."""
    return _HTTP_CLIENT
