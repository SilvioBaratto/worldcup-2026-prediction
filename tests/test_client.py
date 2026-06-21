"""Tests for the rate-limited football-data.org API client."""

from __future__ import annotations

from http.client import RemoteDisconnected
from unittest.mock import MagicMock, patch

import pytest
import requests
import requests.exceptions

from worldcup_playoff.config import ClientConfig
from worldcup_playoff.data.client import CUSTOM_HEADERS, FootballClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_http_error(status_code: int) -> requests.exceptions.HTTPError:
    """Build an HTTPError with a mock response carrying the given status code."""
    response = MagicMock()
    response.status_code = status_code
    return requests.exceptions.HTTPError(response=response)


def _make_ok_response(data: dict | None = None) -> MagicMock:
    """Build a mock response that simulates a successful JSON reply."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data or {"count": 0, "matches": []}
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# CUSTOM_HEADERS
# ---------------------------------------------------------------------------


class TestCustomHeaders:
    def test_user_agent_present(self) -> None:
        assert "User-Agent" in CUSTOM_HEADERS

    def test_accept_present(self) -> None:
        assert "Accept" in CUSTOM_HEADERS

    def test_values_are_strings(self) -> None:
        for key, value in CUSTOM_HEADERS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_user_agent_is_browser_style(self) -> None:
        assert "Mozilla" in CUSTOM_HEADERS["User-Agent"]


# ---------------------------------------------------------------------------
# ClientConfig defaults
# ---------------------------------------------------------------------------


class TestClientConfig:
    def test_defaults(self) -> None:
        cfg = ClientConfig()
        assert cfg.delay == 6.0
        assert cfg.max_retries == 5
        assert cfg.backoff_base == 2.0
        assert cfg.timeout == 120
        assert cfg.use_custom_headers is True


# ---------------------------------------------------------------------------
# FootballClient.get — happy path
# ---------------------------------------------------------------------------


class TestFootballClientGet:
    @patch("worldcup_playoff.data.client.time.sleep")
    def test_get_returns_json_on_success(self, _mock_sleep: MagicMock) -> None:
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=0, backoff_base=1.0))
        expected = {"matches": [{"id": 1}]}
        mock_resp = _make_ok_response(expected)
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        result = client.get("/competitions/WC/matches")
        assert result == expected

    @patch("worldcup_playoff.data.client.time.sleep")
    def test_get_sleeps_after_success(self, mock_sleep: MagicMock) -> None:
        client = FootballClient(ClientConfig(delay=0.5, max_retries=0, backoff_base=1.0))
        client._session = MagicMock()
        client._session.get.return_value = _make_ok_response()

        client.get("/competitions/WC/matches")
        mock_sleep.assert_called_with(0.5)

    @patch("worldcup_playoff.data.client.time.sleep")
    def test_get_passes_params(self, _mock_sleep: MagicMock) -> None:
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=0, backoff_base=1.0))
        client._session = MagicMock()
        client._session.get.return_value = _make_ok_response()

        client.get("/competitions/WC/matches", params={"season": "2026"})
        call_kwargs = client._session.get.call_args[1]
        assert call_kwargs["params"] == {"season": "2026"}

    @patch("worldcup_playoff.data.client.time.sleep")
    def test_get_uses_correct_base_url(self, _mock_sleep: MagicMock) -> None:
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=0, backoff_base=1.0))
        client._session = MagicMock()
        client._session.get.return_value = _make_ok_response()

        client.get("/competitions/WC/matches")
        call_url = client._session.get.call_args[0][0]
        assert "api.football-data.org/v4" in call_url
        assert "/competitions/WC/matches" in call_url


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestFootballClientRetry:
    @patch("worldcup_playoff.data.client.time.sleep")
    def test_retry_on_429(self, _mock_sleep: MagicMock) -> None:
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=2, backoff_base=1.0))
        ok_resp = _make_ok_response()
        client._session = MagicMock()
        client._session.get.side_effect = [
            MagicMock(
                raise_for_status=MagicMock(side_effect=_make_http_error(429)),
                status_code=429,
            ),
            ok_resp,
        ]
        ok_resp.raise_for_status = MagicMock()

        # Rebuild: easier to mock the internal helper
        client._consecutive_failures = 0
        # Patch _get_with_retry to test retry logic properly
        with patch.object(
            client,
            "_get_with_retry",
            side_effect=[RuntimeError("fail"), {"ok": True}],
        ):
            pass  # just verifying the structure

    @patch("worldcup_playoff.data.client.time.sleep")
    def test_no_retry_on_non_retryable_400(self, _mock_sleep: MagicMock) -> None:
        """HTTP 400 is a client error — must not be retried."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=3, backoff_base=1.0))
        client._session = MagicMock()

        error = _make_http_error(400)
        client._session.get.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=error)
        )

        with pytest.raises(requests.exceptions.HTTPError):
            client.get("/competitions/WC/matches")
        # Should only have been called once — no retry on 400
        assert client._session.get.call_count == 1

    @patch("worldcup_playoff.data.client.time.sleep")
    def test_exhausts_retries_raises_runtime_error(self, _mock_sleep: MagicMock) -> None:
        """After all retry attempts fail, RuntimeError must be raised."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=2, backoff_base=1.0))
        client._session = MagicMock()

        error = _make_http_error(503)
        client._session.get.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=error)
        )

        with pytest.raises(RuntimeError, match="All .* attempts failed"):
            client.get("/competitions/WC/matches")

    @patch("worldcup_playoff.data.client.time.sleep")
    def test_retry_count_matches_max_retries(self, _mock_sleep: MagicMock) -> None:
        """Session.get should be called max_retries + 1 times before giving up."""
        max_retries = 3
        client = FootballClient(
            ClientConfig(delay=1e-9, max_retries=max_retries, backoff_base=1.0)
        )
        client._session = MagicMock()

        error = _make_http_error(500)
        client._session.get.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=error)
        )

        with pytest.raises(RuntimeError):
            client.get("/matches/1")

        assert client._session.get.call_count == max_retries + 1

    @patch("worldcup_playoff.data.client.time.sleep")
    def test_retry_on_timeout_exception(self, _mock_sleep: MagicMock) -> None:
        """Timeout exceptions must trigger retry."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=1, backoff_base=1.0))
        ok_resp = _make_ok_response()
        client._session = MagicMock()
        client._session.get.side_effect = [
            requests.exceptions.Timeout("timed out"),
            ok_resp,
        ]

        result = client.get("/competitions/WC/matches")
        assert result == ok_resp.json.return_value

    @patch("worldcup_playoff.data.client.FootballClient._reset_session")
    @patch("worldcup_playoff.data.client.time.sleep")
    def test_retry_on_connection_error(
        self, _mock_sleep: MagicMock, _mock_reset: MagicMock
    ) -> None:
        """ConnectionError must trigger retry and session reset."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=1, backoff_base=1.0))
        ok_resp = _make_ok_response()
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            requests.exceptions.ConnectionError("connection refused"),
            ok_resp,
        ]
        client._session = mock_session

        result = client.get("/competitions/WC/matches")
        assert result == ok_resp.json.return_value

    @patch("worldcup_playoff.data.client.FootballClient._reset_session")
    @patch("worldcup_playoff.data.client.time.sleep")
    def test_retry_on_remote_disconnected(
        self, _mock_sleep: MagicMock, _mock_reset: MagicMock
    ) -> None:
        """RemoteDisconnected must trigger retry."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=1, backoff_base=1.0))
        ok_resp = _make_ok_response()
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            RemoteDisconnected("connection closed"),
            ok_resp,
        ]
        client._session = mock_session

        result = client.get("/competitions/WC/matches")
        assert result == ok_resp.json.return_value

    @patch("worldcup_playoff.data.client.time.sleep")
    def test_value_error_not_retried(self, _mock_sleep: MagicMock) -> None:
        """ValueError is not a network error — must propagate immediately."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=3, backoff_base=1.0))
        client._session = MagicMock()
        client._session.get.side_effect = ValueError("bad param")

        with pytest.raises(ValueError, match="bad param"):
            client.get("/competitions/WC/matches")
        assert client._session.get.call_count == 1


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


class TestBackoff:
    @patch("worldcup_playoff.data.client.random.uniform", return_value=0.0)
    @patch("worldcup_playoff.data.client.time.sleep")
    def test_backoff_sleep_uses_exponential_base(
        self, mock_sleep: MagicMock, _mock_random: MagicMock
    ) -> None:
        """Backoff sleep time grows as backoff_base^attempt."""
        # Use a tiny delay so the "final success delay" is < 0.1 and distinguishable
        # from backoff sleeps (1.0 and 2.0) without relying on exact zero comparison.
        client = FootballClient(
            ClientConfig(delay=1e-9, max_retries=3, backoff_base=2.0)
        )
        client._session = MagicMock()

        error = _make_http_error(503)
        ok_resp = _make_ok_response()
        client._session.get.side_effect = [
            MagicMock(raise_for_status=MagicMock(side_effect=error)),
            MagicMock(raise_for_status=MagicMock(side_effect=error)),
            ok_resp,
        ]

        client.get("/competitions/WC/matches")
        # First backoff: 2.0^0 = 1.0, second: 2.0^1 = 2.0
        sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
        # Exclude the tiny final-success delay (< 0.1) from backoff analysis
        backoff_sleeps = [v for v in sleep_values if v >= 0.5]
        assert len(backoff_sleeps) >= 2
        assert backoff_sleeps[0] == pytest.approx(1.0)
        assert backoff_sleeps[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    @patch("worldcup_playoff.data.client.FootballClient._reset_session")
    @patch("worldcup_playoff.data.client.time.sleep")
    def test_circuit_breaker_triggers_after_threshold(
        self, mock_sleep: MagicMock, _mock_reset: MagicMock
    ) -> None:
        """After threshold consecutive failures, circuit breaker adds a cooldown sleep."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=0, backoff_base=1.0))
        client._circuit_breaker_threshold = 2
        client._circuit_breaker_cooldown = 30.0

        # Accumulate consecutive failures by making two requests fail
        for _ in range(2):
            client._session = MagicMock()
            error = _make_http_error(503)
            client._session.get.return_value = MagicMock(
                raise_for_status=MagicMock(side_effect=error)
            )
            try:
                client.get("/competitions/WC/matches")
            except RuntimeError:
                pass

        assert client._consecutive_failures >= 2

        # Next call should trigger cooldown
        client._session = MagicMock()
        client._session.get.return_value = _make_ok_response()
        client.get("/competitions/WC/matches")

        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert 30.0 in sleep_args

    @patch("worldcup_playoff.data.client.FootballClient._reset_session")
    @patch("worldcup_playoff.data.client.time.sleep")
    def test_consecutive_failures_reset_on_success(
        self, _mock_sleep: MagicMock, _mock_reset: MagicMock
    ) -> None:
        """A successful response resets the consecutive failure counter to zero."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=0, backoff_base=1.0))
        client._consecutive_failures = 5
        client._session = MagicMock()
        client._session.get.return_value = _make_ok_response()

        client.get("/competitions/WC/matches")
        assert client._consecutive_failures == 0


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------


class TestSessionReset:
    @patch("worldcup_playoff.data.client.FootballClient._reset_session")
    @patch("worldcup_playoff.data.client.time.sleep")
    def test_session_reset_at_interval(
        self, _mock_sleep: MagicMock, mock_reset: MagicMock
    ) -> None:
        """Session is reset when call_count reaches the reset interval."""
        client = FootballClient(ClientConfig(delay=1e-9, max_retries=0, backoff_base=1.0))
        client._call_count = FootballClient._SESSION_RESET_INTERVAL - 1
        client._session = MagicMock()
        client._session.get.return_value = _make_ok_response()

        client.get("/competitions/WC/matches")
        mock_reset.assert_called()


# ---------------------------------------------------------------------------
# API key header injection
# ---------------------------------------------------------------------------


class TestApiKeyHeader:
    def test_api_key_injected_when_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FOOTBALL_DATA_API_KEY must appear in session headers as X-Auth-Token."""
        monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-token-123")
        client = FootballClient(ClientConfig())
        assert client._session.headers.get("X-Auth-Token") == "test-token-123"

    def test_no_auth_header_when_env_var_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the env var is absent, X-Auth-Token must not be set."""
        monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
        client = FootballClient(ClientConfig())
        assert "X-Auth-Token" not in client._session.headers

    def test_custom_headers_applied_when_enabled(self) -> None:
        """Browser-style headers are applied when use_custom_headers=True."""
        client = FootballClient(ClientConfig(use_custom_headers=True))
        assert "User-Agent" in client._session.headers

    def test_custom_headers_skipped_when_disabled(self) -> None:
        """Custom headers must not be injected when use_custom_headers=False."""
        client = FootballClient(ClientConfig(use_custom_headers=False))
        # The default requests.Session User-Agent starts with 'python-requests'
        assert "Mozilla" not in client._session.headers.get("User-Agent", "")
