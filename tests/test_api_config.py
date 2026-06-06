"""Property-based tests for api_config module."""

import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from api_config import resolve_guardrails_mode, validate_api_key


# Feature: garak-endpoint-integration, Property 5: Mode Resolution
# **Validates: Requirements 3.2, 3.5, 3.6**
@settings(max_examples=100, deadline=5000)
@given(
    default_mode=st.sampled_from(["guarded", "unguarded"]),
    header_value=st.one_of(
        st.none(),
        st.just(""),
        st.text(),
        st.just("guarded"),
        st.just("unguarded"),
    ),
)
def test_mode_resolution(default_mode: str, header_value):
    """Property 5: Mode Resolution.

    For any combination of default guardrails mode and an X-Guardrails-Mode
    header value, the resolved mode SHALL equal the header value if it is
    exactly "guarded" or "unguarded", and SHALL equal the default mode
    otherwise (including when the header is absent, empty, or any other string).
    """
    result = resolve_guardrails_mode(header_value, default_mode)

    if header_value in ("guarded", "unguarded"):
        assert result == header_value, (
            f"Expected header value '{header_value}' to override default '{default_mode}', "
            f"but got '{result}'"
        )
    else:
        assert result == default_mode, (
            f"Expected default mode '{default_mode}' when header is '{header_value}', "
            f"but got '{result}'"
        )


import pytest
from hypothesis import assume

from api_config import validate_port


# Feature: garak-endpoint-integration, Property 12: Port Validation
# **Validates: Requirements 8.1, 8.5**


@settings(max_examples=100, deadline=5000)
@given(port=st.integers(min_value=1, max_value=65535))
def test_valid_port_strings_are_accepted(port):
    """For any string that parses to an integer within [1, 65535], validate_port SHALL accept it."""
    result = validate_port(str(port))
    assert result == port


@settings(max_examples=100, deadline=5000)
@given(port_str=st.text().filter(lambda s: s != ""))
def test_non_numeric_port_strings_are_rejected(port_str):
    """For any non-numeric string, validate_port SHALL reject it with sys.exit(1)."""
    # Ensure it's truly non-parseable as int
    try:
        int(port_str)
        assume(False)  # Skip if it accidentally parses as int
    except (ValueError, TypeError):
        pass

    with pytest.raises(SystemExit) as exc_info:
        validate_port(port_str)
    assert exc_info.value.code == 1


@settings(max_examples=100, deadline=5000)
@given(port=st.integers(max_value=0))
def test_port_below_range_is_rejected(port):
    """For any integer <= 0, validate_port SHALL reject it with sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        validate_port(str(port))
    assert exc_info.value.code == 1


@settings(max_examples=100, deadline=5000)
@given(port=st.integers(min_value=65536))
def test_port_above_range_is_rejected(port):
    """For any integer > 65535, validate_port SHALL reject it with sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        validate_port(str(port))
    assert exc_info.value.code == 1


class TestPortValidationEdgeCases:
    """Edge case tests for port validation."""

    def test_port_zero_rejected(self):
        """Port '0' is out of valid range and SHALL be rejected."""
        with pytest.raises(SystemExit) as exc_info:
            validate_port("0")
        assert exc_info.value.code == 1

    def test_port_65536_rejected(self):
        """Port '65536' is out of valid range and SHALL be rejected."""
        with pytest.raises(SystemExit) as exc_info:
            validate_port("65536")
        assert exc_info.value.code == 1

    def test_port_negative_one_rejected(self):
        """Port '-1' is out of valid range and SHALL be rejected."""
        with pytest.raises(SystemExit) as exc_info:
            validate_port("-1")
        assert exc_info.value.code == 1

    def test_port_non_numeric_rejected(self):
        """Port 'abc' is not parseable as integer and SHALL be rejected."""
        with pytest.raises(SystemExit) as exc_info:
            validate_port("abc")
        assert exc_info.value.code == 1

    def test_port_float_string_rejected(self):
        """Port '3.14' is not parseable as integer and SHALL be rejected."""
        with pytest.raises(SystemExit) as exc_info:
            validate_port("3.14")
        assert exc_info.value.code == 1

    def test_port_empty_string_rejected(self):
        """Port '' is not parseable as integer and SHALL be rejected."""
        with pytest.raises(SystemExit) as exc_info:
            validate_port("")
        assert exc_info.value.code == 1


# Feature: garak-endpoint-integration, Property 1: API Key Validation
# **Validates: Requirements 1.1, 1.4**
#
# For any string value provided as NVIDIA_API_KEY, if the string is empty or
# composed entirely of whitespace characters, the validation function SHALL
# reject it; if the string contains at least one non-whitespace character,
# the validation function SHALL accept it.


# Strategy: generate whitespace-only strings (empty, spaces, tabs, newlines, mixed)
whitespace_only = st.from_regex(r"^[\s]*$", fullmatch=True)


@settings(max_examples=100, deadline=5000)
@given(key=whitespace_only)
def test_whitespace_only_keys_are_rejected(key):
    """Whitespace-only strings (including empty) must cause sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        validate_api_key(key)
    assert exc_info.value.code == 1


# Strategy: generate strings with at least one non-whitespace character
valid_keys = st.text(min_size=1).filter(lambda s: s.strip() != "")


@settings(max_examples=100, deadline=5000)
@given(key=valid_keys)
def test_non_whitespace_keys_are_accepted(key):
    """Strings with at least one non-whitespace character must be accepted."""
    result = validate_api_key(key)
    assert result == key


def test_none_input_is_rejected():
    """None input must cause sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        validate_api_key(None)
    assert exc_info.value.code == 1
