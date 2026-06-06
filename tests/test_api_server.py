"""Unit tests for server initialization and configuration loading.

Tests startup with valid environment variables and validates that the server
exits appropriately on invalid configurations.

Requirements: 1.1, 1.4, 1.5, 3.7, 8.5
"""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from api_config import load_server_config


class TestLoadConfigWithValidEnvVars:
    """Test that load_server_config returns correct ServerConfig with valid env."""

    @patch("api_config.load_dotenv")
    def test_load_config_with_valid_env_vars(self, mock_load_dotenv):
        """Verify ServerConfig is populated correctly from environment variables."""
        env = {
            "NVIDIA_API_KEY": "test-key-123",
            "GUARDRAILS_MODE": "guarded",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_server_config()

        assert config.nvidia_api_key == "test-key-123"
        assert config.guardrails_mode == "guarded"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.config_dir.endswith("config")
        mock_load_dotenv.assert_called_once()


class TestExitOnMissingApiKey:
    """Test that missing NVIDIA_API_KEY causes sys.exit(1)."""

    @patch("api_config.load_dotenv")
    def test_exit_on_missing_api_key(self, mock_load_dotenv):
        """Verify sys.exit(1) when NVIDIA_API_KEY is not set."""
        env = {
            "GUARDRAILS_MODE": "guarded",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                load_server_config()
        assert exc_info.value.code == 1


class TestExitOnWhitespaceOnlyApiKey:
    """Test that whitespace-only NVIDIA_API_KEY causes sys.exit(1)."""

    @patch("api_config.load_dotenv")
    def test_exit_on_whitespace_only_api_key(self, mock_load_dotenv):
        """Verify sys.exit(1) when NVIDIA_API_KEY contains only whitespace."""
        env = {
            "NVIDIA_API_KEY": "   ",
            "GUARDRAILS_MODE": "guarded",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                load_server_config()
        assert exc_info.value.code == 1


class TestExitOnInvalidGuardrailsMode:
    """Test that invalid GUARDRAILS_MODE causes sys.exit(1)."""

    @patch("api_config.load_dotenv")
    def test_exit_on_invalid_guardrails_mode(self, mock_load_dotenv):
        """Verify sys.exit(1) when GUARDRAILS_MODE is not 'guarded' or 'unguarded'."""
        env = {
            "NVIDIA_API_KEY": "test-key-123",
            "GUARDRAILS_MODE": "invalid_mode",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                load_server_config()
        assert exc_info.value.code == 1


class TestExitOnInvalidPort:
    """Test that non-numeric API_PORT causes sys.exit(1)."""

    @patch("api_config.load_dotenv")
    def test_exit_on_invalid_port(self, mock_load_dotenv):
        """Verify sys.exit(1) when API_PORT is not a valid integer."""
        env = {
            "NVIDIA_API_KEY": "test-key-123",
            "GUARDRAILS_MODE": "guarded",
            "API_HOST": "0.0.0.0",
            "API_PORT": "not_a_number",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                load_server_config()
        assert exc_info.value.code == 1


class TestExitOnPortOutOfRange:
    """Test that out-of-range API_PORT causes sys.exit(1)."""

    @patch("api_config.load_dotenv")
    def test_exit_on_port_out_of_range(self, mock_load_dotenv):
        """Verify sys.exit(1) when API_PORT is outside [1, 65535]."""
        env = {
            "NVIDIA_API_KEY": "test-key-123",
            "GUARDRAILS_MODE": "guarded",
            "API_HOST": "0.0.0.0",
            "API_PORT": "99999",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                load_server_config()
        assert exc_info.value.code == 1


class TestCreateAppReturnsFastApiInstance:
    """Test that create_app() returns a FastAPI instance."""

    def test_create_app_returns_fastapi_instance(self):
        """Verify create_app() produces a FastAPI app object."""
        from api_server import create_app

        app = create_app()
        assert isinstance(app, FastAPI)


class TestConfigDefaults:
    """Test that defaults are applied when optional env vars are missing."""

    @patch("api_config.load_dotenv")
    def test_config_defaults(self, mock_load_dotenv):
        """Verify host defaults to '0.0.0.0', port to 8000, mode to 'guarded'."""
        env = {
            "NVIDIA_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_server_config()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.guardrails_mode == "guarded"
