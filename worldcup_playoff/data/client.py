"""Rate-limited football-data.org API client with circuit-breaker retry logic."""

from __future__ import annotations

import logging
import os
import random
import time
from http.client import RemoteDisconnected
from typing import Any

import requests
import requests.exceptions

from worldcup_playoff.config import ClientConfig

logger = logging.getLogger(__name__)

# football-data.org v4 base URL
_BASE_URL = "https://api.football-data.org/v4"

# HTTP status codes that warrant a retry with backoff
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Browser-style User-Agent to avoid being blocked by WAF rules
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

CUSTOM_HEADERS: dict[str, str] = {
    "User-Agent": _DEFAULT_USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


class FootballClient:
    """Wraps football-data.org v4 REST calls with rate limiting and retries.

    Implements a circuit-breaker pattern: after ``_circuit_breaker_threshold``
    consecutive failures the client pauses for ``_circuit_breaker_cooldown``
    seconds before allowing the next attempt, giving the remote server time
    to recover.

    The API key is read from the ``FOOTBALL_DATA_API_KEY`` environment variable.
    If the variable is absent or empty the client still works but is subject to
    the unauthenticated rate limit (10 req/min → default delay of 6 s).
    """

    _SESSION_RESET_INTERVAL: int = 50

    def __init__(self, config: ClientConfig | None = None) -> None:
        self._config = config or ClientConfig()
        self._session: requests.Session = self._build_session()
        self._call_count: int = 0
        self._consecutive_failures: int = 0
        self._circuit_breaker_threshold: int = 3
        self._circuit_breaker_cooldown: float = 60.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue a GET request against the v4 base URL and return parsed JSON.

        Args:
            path: Endpoint path, e.g. ``"/competitions/WC/matches"``.
            params: Optional query-string parameters.

        Returns:
            Parsed JSON response as a Python dict.

        Raises:
            requests.exceptions.HTTPError: On non-retryable 4xx responses.
            RuntimeError: When all retry attempts are exhausted.
        """
        # Circuit breaker: if many consecutive failures, pause before retrying
        if self._consecutive_failures >= self._circuit_breaker_threshold:
            logger.warning(
                "Circuit breaker open (%d consecutive failures) — "
                "cooling down %.0fs before next attempt",
                self._consecutive_failures,
                self._circuit_breaker_cooldown,
            )
            time.sleep(self._circuit_breaker_cooldown)
            self._reset_session()

        # Periodic session recycling to avoid stale connections
        self._call_count += 1
        if self._call_count % self._SESSION_RESET_INTERVAL == 0:
            self._reset_session()

        result = self._get_with_retry(path, params)
        self._consecutive_failures = 0
        time.sleep(self._config.delay)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_with_retry(
        self,
        path: str,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Attempt the GET request up to ``max_retries + 1`` times.

        Uses exponential backoff with jitter on retryable errors:
        ``wait = backoff_base ^ attempt + random(0, 1)``
        """
        last_exception: Exception | None = None
        url = f"{_BASE_URL}{path}"

        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._config.timeout,
                )
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in _RETRYABLE_STATUS_CODES:
                    raise
                last_exception = exc
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                RemoteDisconnected,
                ConnectionResetError,
                OSError,
            ) as exc:
                last_exception = exc
            except Exception:
                raise

            self._consecutive_failures += 1

            if attempt < self._config.max_retries:
                jitter = random.uniform(0, 1)  # noqa: S311
                wait = self._config.backoff_base**attempt + jitter
                logger.warning(
                    "Retry %d/%d for %s (error: %s), sleeping %.1fs",
                    attempt + 1,
                    self._config.max_retries,
                    url,
                    last_exception,
                    wait,
                )
                time.sleep(wait)

                # Fresh TCP connection on connection-level errors
                if isinstance(
                    last_exception,
                    (
                        requests.exceptions.ConnectionError,
                        RemoteDisconnected,
                        ConnectionResetError,
                    ),
                ):
                    self._reset_session()

        msg = f"All {self._config.max_retries + 1} attempts failed for {url}"
        raise RuntimeError(msg) from last_exception

    def _reset_session(self) -> None:
        """Replace the requests.Session with a fresh instance."""
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass
        self._session = self._build_session()
        logger.debug("Reset HTTP session")

    def _build_session(self) -> requests.Session:
        """Create a new requests.Session with appropriate headers."""
        session = requests.Session()

        if self._config.use_custom_headers:
            session.headers.update(CUSTOM_HEADERS)

        # Attach API key if available — unauthenticated access still works at lower rate
        api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
        if api_key:
            session.headers["X-Auth-Token"] = api_key
            logger.debug("Authenticated session created (API key present)")
        else:
            logger.debug("Unauthenticated session created (no API key found)")

        return session
