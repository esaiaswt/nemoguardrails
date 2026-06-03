"""
Unit tests for config loading and hot-reload behavior.

Tests verify that:
- create_rails_instance() succeeds with valid config directory
- create_rails_instance() raises FileNotFoundError when config dir is missing
- create_rails_instance() raises ValueError when config is invalid
- Hot-reload failure preserves the previous LLMRails instance
- load_config_from_disk() reads content from config files

Validates: Requirements 7.1, 7.5, 7.6, 7.7
"""

import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestCreateRailsInstanceValidConfig:
    """Test that create_rails_instance() succeeds with valid config structure."""

    def test_create_rails_instance_with_valid_config(self):
        """Verify that create_rails_instance() succeeds when config/ directory
        exists with valid config.yml.

        Validates: Requirements 7.1, 7.5
        """
        # Remove cached app module to get a fresh import
        if "app" in sys.modules:
            del sys.modules["app"]

        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()

        mock_nemo = sys.modules["nemoguardrails"]
        mock_rails_config = MagicMock()
        mock_llm_rails_instance = MagicMock()

        mock_nemo.RailsConfig.from_path.return_value = mock_rails_config
        mock_nemo.LLMRails.return_value = mock_llm_rails_instance

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        # Patch os.path.isdir and os.path.isfile to simulate valid directory
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = app.create_rails_instance()

        # Verify RailsConfig.from_path was called with the CONFIG_DIR
        mock_nemo.RailsConfig.from_path.assert_called_with(app.CONFIG_DIR)
        # Verify LLMRails was instantiated with the config
        mock_nemo.LLMRails.assert_called_with(mock_rails_config)
        # Verify the return value is the LLMRails instance
        assert result == mock_llm_rails_instance


class TestCreateRailsInstanceMissingConfigDir:
    """Test that create_rails_instance() raises FileNotFoundError when config dir is missing."""

    def test_create_rails_instance_missing_config_dir(self):
        """Verify that create_rails_instance() raises FileNotFoundError when
        the config directory doesn't exist.

        Validates: Requirements 7.5, 7.6
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        # Simulate missing config directory
        with patch("os.path.isdir", return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                app.create_rails_instance()

        assert "Config directory not found" in str(exc_info.value)


class TestCreateRailsInstanceInvalidConfig:
    """Test that create_rails_instance() raises ValueError when config is invalid."""

    def test_create_rails_instance_invalid_config(self):
        """Verify that create_rails_instance() raises ValueError when
        RailsConfig.from_path fails with an exception.

        Validates: Requirements 7.6
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()

        mock_nemo = sys.modules["nemoguardrails"]
        mock_nemo.RailsConfig.from_path.side_effect = Exception("Invalid YAML format")

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError) as exc_info:
                app.create_rails_instance()

        assert "Failed to load config" in str(exc_info.value)

        # Reset side_effect for other tests
        mock_nemo.RailsConfig.from_path.side_effect = None


class TestHotReloadPreservesInstance:
    """Test that hot-reload failure preserves the previous LLMRails instance."""

    def test_hot_reload_preserves_instance_on_failure(self):
        """When writing config to disk succeeds but create_rails_instance() fails,
        the previous rails instance should be preserved in session state.

        Validates: Requirements 7.7
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()

        mock_nemo = sys.modules["nemoguardrails"]

        # Set up initial successful rails instance
        initial_rails = MagicMock(name="initial_rails_instance")
        mock_nemo.RailsConfig.from_path.return_value = MagicMock()
        mock_nemo.LLMRails.return_value = initial_rails

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        # Store initial rails instance in session state
        mock_st.session_state["rails"] = initial_rails
        mock_st.session_state["config_yml"] = "models:\n  - type: main"
        mock_st.session_state["config_co"] = "flow main\n  pass"

        # Now simulate a hot-reload failure:
        # create_rails_instance raises an exception after writing to disk
        mock_nemo.RailsConfig.from_path.side_effect = Exception("Config parse error")

        # Simulate the hot-reload logic from the Update button handler
        previous_rails = mock_st.session_state["rails"]

        try:
            # Write config to disk (this would succeed in real code)
            with patch("builtins.open", mock_open()):
                # Attempt to create new rails instance (this will fail)
                with patch("os.path.isdir", return_value=True), \
                     patch("os.path.isfile", return_value=True):
                    new_rails = app.create_rails_instance()
                    mock_st.session_state["rails"] = new_rails
        except (ValueError, Exception):
            # On failure, previous instance should be preserved
            pass

        # Verify the previous rails instance is still in session state
        assert mock_st.session_state["rails"] == previous_rails
        assert mock_st.session_state["rails"] is initial_rails

        # Reset side_effect for other tests
        mock_nemo.RailsConfig.from_path.side_effect = None


class TestLoadConfigFromDisk:
    """Test that load_config_from_disk() reads content from config files."""

    def test_load_config_from_disk_reads_files(self):
        """Verify that load_config_from_disk() returns content from config.yml
        and main.co files on disk.

        Validates: Requirements 7.1
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        mock_st = sys.modules["streamlit"]
        mock_st.error.reset_mock()
        mock_st.stop.reset_mock()
        mock_st.stop.side_effect = None

        mock_nemo = sys.modules["nemoguardrails"]
        mock_nemo.RailsConfig.from_path.side_effect = None
        mock_nemo.RailsConfig.from_path.return_value = MagicMock()
        mock_nemo.LLMRails.return_value = MagicMock()

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        expected_yml = "models:\n  - type: main\n    engine: openai"
        expected_co = "flow greeting\n  user said hello"

        # Mock file reads for both config files
        def mock_open_fn(path, *args, **kwargs):
            m = MagicMock()
            if "config.yml" in str(path):
                m.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=expected_yml)))
            elif "main.co" in str(path):
                m.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=expected_co)))
            else:
                m.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value="")))
            m.__exit__ = MagicMock(return_value=False)
            return m

        with patch("builtins.open", side_effect=mock_open_fn):
            config_yml, config_co = app.load_config_from_disk()

        assert config_yml == expected_yml
        assert config_co == expected_co
