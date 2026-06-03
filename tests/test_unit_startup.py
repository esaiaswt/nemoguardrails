"""
Unit tests for application startup, API key validation, config loading from disk,
and session state initialization.

Validates: Requirements 7.1, 7.5, 7.6
"""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestAPIKeyValidation:
    """Test API key validation behavior on startup."""

    def test_missing_api_key_shows_error(self):
        """When NVIDIA_API_KEY is missing, st.error and st.stop should be called.

        Validates: Requirements 7.5
        """
        # Remove the API key from environment
        env_without_key = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}

        # We need to reimport app.py with the key missing.
        # Reset the mock call counts and remove app from sys.modules.
        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()

        # Remove cached app module so it re-executes on import
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch.dict(os.environ, env_without_key, clear=True):
            import app  # noqa: F401

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "NVIDIA_API_KEY" in error_msg
        mock_st.stop.assert_called_once()

    def test_present_api_key_proceeds(self):
        """When NVIDIA_API_KEY is present, st.error and st.stop should NOT be called.

        Validates: Requirements 7.5
        """
        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()

        # Remove cached app module so it re-executes on import
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app  # noqa: F401

        mock_st.error.assert_not_called()
        mock_st.stop.assert_not_called()


class TestConfigLoadedFromDisk:
    """Test that config is loaded from disk files (config.yml and main.co)."""

    @pytest.fixture(autouse=True)
    def _import_app(self):
        """Ensure app module is imported with a valid API key."""
        if "app" in sys.modules:
            del sys.modules["app"]

        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app
            self.app = app

    def test_config_yml_loaded_from_disk(self):
        """config_yml session state should be loaded from config/config.yml on disk.

        Validates: Requirements 7.1
        """
        # load_config_from_disk should successfully read files
        config_yml, config_co = self.app.load_config_from_disk()

        # config.yml should contain expected NeMo Guardrails YAML elements
        assert 'colang_version' in config_yml
        assert '2.x' in config_yml
        assert 'models' in config_yml

    def test_config_co_loaded_from_disk(self):
        """config_co session state should be loaded from config/main.co on disk.

        Validates: Requirements 7.1
        """
        config_yml, config_co = self.app.load_config_from_disk()

        # main.co should contain expected Colang 2.x flows
        assert 'greeting' in config_co or 'flow' in config_co


class TestSessionStateInitialization:
    """Test that session state is initialized with all expected keys and defaults."""

    def test_session_state_keys_initialized(self):
        """Session state should have messages, guardrails_enabled, config_yml, config_co.

        Validates: Requirements 7.1
        """
        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()

        # Create a fresh MockSessionState without the keys to test initialization
        from tests.conftest import MockSessionState
        mock_st.session_state = MockSessionState()

        # Ensure chat_input returns None so message routing doesn't execute
        mock_st.chat_input.return_value = None

        # Remove cached app module so it re-executes on import
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app  # noqa: F401

        # Verify all expected keys are present
        assert "messages" in mock_st.session_state
        assert "guardrails_enabled" in mock_st.session_state
        assert "config_yml" in mock_st.session_state
        assert "config_co" in mock_st.session_state

        # Verify default values
        assert mock_st.session_state["messages"] == []
        assert mock_st.session_state["guardrails_enabled"] is True
        assert len(mock_st.session_state["config_yml"]) > 0
        assert len(mock_st.session_state["config_co"]) > 0
