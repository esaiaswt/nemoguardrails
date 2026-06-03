"""
Conftest for property-based tests.

Mocks heavy dependencies (streamlit, nemoguardrails, keyboard, psutil) so that
app.py can be imported without actually running the Streamlit application.
"""

import os
import sys
from unittest.mock import MagicMock

# Set environment variable before any imports
os.environ.setdefault("NVIDIA_API_KEY", "nvapi-test1234key5678")

# Create a session state that supports both dict and attribute access
class MockSessionState(dict):
    """Mock Streamlit session_state supporting both dict-style and attribute-style access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

    def __contains__(self, key):
        return dict.__contains__(self, key)


# Create comprehensive streamlit mock
mock_st = MagicMock()
mock_st.session_state = MockSessionState({
    "messages": [],
    "guardrails_enabled": True,
    "config_yml": "models: []",
    "config_co": "flow main\n  pass",
})
mock_st.set_page_config = MagicMock()
mock_st.error = MagicMock()
mock_st.stop = MagicMock()

# Make st.sidebar work as a context manager
mock_sidebar = MagicMock()
mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
mock_sidebar.__exit__ = MagicMock(return_value=False)
mock_st.sidebar = mock_sidebar

mock_st.toggle = MagicMock(return_value=True)
mock_st.success = MagicMock()
mock_st.warning = MagicMock()
mock_st.divider = MagicMock()
mock_st.text_area = MagicMock(return_value="")
mock_st.button = MagicMock(return_value=False)
mock_st.chat_message = MagicMock()
mock_st.chat_input = MagicMock(return_value=None)
mock_st.markdown = MagicMock()
mock_st.spinner = MagicMock()

# Make chat_message work as a context manager
mock_chat_msg = MagicMock()
mock_chat_msg.__enter__ = MagicMock(return_value=mock_chat_msg)
mock_chat_msg.__exit__ = MagicMock(return_value=False)
mock_st.chat_message.return_value = mock_chat_msg

# Make expander work as a context manager
mock_expander = MagicMock()
mock_expander.__enter__ = MagicMock(return_value=mock_expander)
mock_expander.__exit__ = MagicMock(return_value=False)
mock_st.expander = MagicMock(return_value=mock_expander)

mock_st.json = MagicMock()

# Mock nemoguardrails module and submodules
mock_nemoguardrails = MagicMock()
mock_nemoguardrails_utils = MagicMock()

# Mock keyboard and psutil
mock_keyboard = MagicMock()
mock_psutil = MagicMock()

# Mock dotenv
mock_dotenv = MagicMock()
mock_dotenv.load_dotenv = MagicMock()

# Mock openai - need the real structure for OpenAI client
mock_openai_module = MagicMock()
mock_openai_module.OpenAI = MagicMock()
mock_openai_module.APIError = Exception
mock_openai_module.APIConnectionError = Exception

# Insert mocks into sys.modules BEFORE app.py is imported
sys.modules["streamlit"] = mock_st
sys.modules["nemoguardrails"] = mock_nemoguardrails
sys.modules["nemoguardrails.utils"] = mock_nemoguardrails_utils
sys.modules["keyboard"] = mock_keyboard
sys.modules["psutil"] = mock_psutil
sys.modules["dotenv"] = mock_dotenv
sys.modules["openai"] = mock_openai_module
