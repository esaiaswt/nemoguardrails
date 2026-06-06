"""Unit tests for get_health_status() function."""

from unittest.mock import MagicMock

from api_handlers import get_health_status


def test_healthy_when_both_components_initialized():
    """Return 200 with healthy status when both rails and direct_client are available."""
    rails = MagicMock()
    direct_client = MagicMock()

    status_code, body = get_health_status(
        rails=rails,
        direct_client=direct_client,
        default_mode="guarded",
        model_name="nvidia/llama-3.1-nemotron-nano-8b-v1",
    )

    assert status_code == 200
    assert body["status"] == "healthy"
    assert body["guardrails_mode"] == "guarded"
    assert body["model"] == "nvidia/llama-3.1-nemotron-nano-8b-v1"


def test_healthy_with_unguarded_mode():
    """Return correct mode when default_mode is unguarded."""
    rails = MagicMock()
    direct_client = MagicMock()

    status_code, body = get_health_status(
        rails=rails,
        direct_client=direct_client,
        default_mode="unguarded",
        model_name="test-model",
    )

    assert status_code == 200
    assert body["guardrails_mode"] == "unguarded"


def test_unhealthy_when_rails_is_none():
    """Return 503 with reason when rails is not initialized."""
    direct_client = MagicMock()

    status_code, body = get_health_status(
        rails=None,
        direct_client=direct_client,
        default_mode="guarded",
        model_name="test-model",
    )

    assert status_code == 503
    assert body["status"] == "unhealthy"
    assert "LLMRails not initialized" in body["reason"]


def test_unhealthy_when_direct_client_is_none():
    """Return 503 with reason when direct_client is not initialized."""
    rails = MagicMock()

    status_code, body = get_health_status(
        rails=rails,
        direct_client=None,
        default_mode="guarded",
        model_name="test-model",
    )

    assert status_code == 503
    assert body["status"] == "unhealthy"
    assert "Direct client not initialized" in body["reason"]


def test_unhealthy_when_both_components_are_none():
    """Return 503 with reason mentioning both components when neither is initialized."""
    status_code, body = get_health_status(
        rails=None,
        direct_client=None,
        default_mode="guarded",
        model_name="test-model",
    )

    assert status_code == 503
    assert body["status"] == "unhealthy"
    assert "LLMRails not initialized" in body["reason"]
    assert "Direct client not initialized" in body["reason"]
