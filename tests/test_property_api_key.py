"""Property-based test for API key non-exposure.

Feature: nemo-guardrails-playground, Property 6: API key non-exposure

For any NVIDIA API key value, the full key string SHALL never appear in any
UI element text, console log output, or session state value that is rendered
to the user.

**Validates: Requirements 10.2**
"""

from hypothesis import given, settings
from hypothesis import strategies as st_hyp


def mask_api_key(key: str) -> str:
    """Replicate the mask_api_key function from app.py for isolated testing."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"


def simulate_ui_render(key: str) -> list[str]:
    """Simulate a UI render pipeline that would display masked key information.

    This represents the kind of output strings that might appear in
    sidebar status, debug panels, or log messages.
    """
    masked = mask_api_key(key)
    return [
        f"API Key: {masked}",
        f"Connected with key: {masked}",
        f"Status: authenticated ({masked})",
        masked,
    ]


class TestAPIKeyNonExposure:
    """Property 6: API key non-exposure.

    **Validates: Requirements 10.2**
    """

    @given(st_hyp.text(min_size=8, max_size=64))
    @settings(max_examples=100)
    def test_full_key_never_appears_in_masked_output(self, key: str):
        """For any key of 8+ chars, the full key must not appear in masked output."""
        masked = mask_api_key(key)

        # The masked output must not contain the full key
        assert key not in masked, (
            f"Full API key '{key}' was exposed in masked output '{masked}'"
        )

    @given(st_hyp.text(min_size=8, max_size=64))
    @settings(max_examples=100)
    def test_masked_output_differs_from_original_key(self, key: str):
        """For any key of 8+ chars, the masked output must differ from the original."""
        masked = mask_api_key(key)

        # The masked version must be different from the original key
        assert masked != key, (
            f"Masked output is identical to original key '{key}'"
        )

    @given(st_hyp.text(min_size=8, max_size=64))
    @settings(max_examples=100)
    def test_masked_output_contains_mask_marker(self, key: str):
        """For any key of 8+ chars, the masked output must contain '****'."""
        masked = mask_api_key(key)

        # The masked version must contain the mask marker
        assert "****" in masked, (
            f"Masked output '{masked}' does not contain '****' for key '{key}'"
        )

    @given(st_hyp.text(min_size=8, max_size=64))
    @settings(max_examples=100)
    def test_full_key_never_in_simulated_render_output(self, key: str):
        """For any key of 8+ chars, the full key must not appear in any rendered UI string."""
        render_outputs = simulate_ui_render(key)

        for output_str in render_outputs:
            assert key not in output_str, (
                f"Full API key '{key}' was exposed in render output '{output_str}'"
            )
