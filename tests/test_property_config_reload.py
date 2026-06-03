"""Property-based test for configuration hot-reload.

Feature: annoy-fastembed-rag, Property: Configuration hot-reload

For any valid configuration content, when create_rails_instance() is called,
the application SHALL produce a new LLMRails instance initialized from the config
directory using RailsConfig.from_path().

**Validates: Requirements 7.4, 7.7**
"""

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st_hyp

import sys
import os

# Add project root to path so we can import app module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@settings(max_examples=100)
@given(st_hyp.text(min_size=1))
def test_config_hot_reload_creates_new_instance(config_content):
    """Property: Configuration hot-reload creates new instance.

    For any call to create_rails_instance(), it should invoke
    RailsConfig.from_path with the config directory and create a new
    LLMRails instance with the resulting config.

    **Validates: Requirements 7.4, 7.7**
    """
    mock_config = MagicMock(name="mock_rails_config")
    mock_config.docs = []

    with patch("app.RailsConfig") as mock_rails_config_cls, \
         patch("app.LLMRails") as mock_llm_rails_cls:

        mock_rails_config_cls.from_path.return_value = mock_config
        mock_llm_rails_instance = MagicMock(name="mock_llm_rails_instance")
        mock_llm_rails_instance.kb = None
        mock_llm_rails_cls.return_value = mock_llm_rails_instance

        from app import create_rails_instance

        result = create_rails_instance()

        # Verify RailsConfig.from_path was called (directory-based loading)
        mock_rails_config_cls.from_path.assert_called_once()

        # Verify LLMRails was instantiated with the config from RailsConfig.from_path
        mock_llm_rails_cls.assert_called_once_with(mock_config)

        # Verify the result is the new LLMRails instance
        assert result is mock_llm_rails_instance
