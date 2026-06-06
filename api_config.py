"""Server configuration loading and validation for the API server."""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Configure logger with the same format used by app.py
logger = logging.getLogger("api_server.config")


@dataclass
class ServerConfig:
    """Configuration for the API server loaded from environment variables."""

    host: str  # From API_HOST env var, default "0.0.0.0"
    port: int  # From API_PORT env var, default 8000
    guardrails_mode: str  # From GUARDRAILS_MODE env var, default "guarded"
    nvidia_api_key: str  # From NVIDIA_API_KEY env var
    config_dir: str  # Path to config/ directory


def validate_api_key(api_key: Optional[str]) -> str:
    """Validate that the NVIDIA API key is present and non-whitespace.

    Args:
        api_key: The raw value from the environment variable.

    Returns:
        The validated API key string.

    Raises:
        SystemExit: If the key is missing, empty, or whitespace-only.
    """
    if api_key is None or api_key.strip() == "":
        logger.error(
            "NVIDIA_API_KEY is missing or contains only whitespace. "
            "Please set a valid API key in your .env file or environment."
        )
        sys.exit(1)
    return api_key


def validate_guardrails_mode(mode: str) -> str:
    """Validate that the guardrails mode is either 'guarded' or 'unguarded'.

    Args:
        mode: The raw value from the environment variable.

    Returns:
        The validated mode string.

    Raises:
        SystemExit: If the mode is not 'guarded' or 'unguarded'.
    """
    valid_modes = ("guarded", "unguarded")
    if mode not in valid_modes:
        logger.error(
            f"Invalid GUARDRAILS_MODE: '{mode}'. "
            f"Must be one of: {', '.join(valid_modes)}"
        )
        sys.exit(1)
    return mode


def validate_port(port_str: str) -> int:
    """Validate that the port string is an integer in [1, 65535].

    Args:
        port_str: The raw value from the environment variable.

    Returns:
        The validated port as an integer.

    Raises:
        SystemExit: If the port is not a valid integer or out of range.
    """
    try:
        port = int(port_str)
    except (ValueError, TypeError):
        logger.error(
            f"Invalid API_PORT: '{port_str}'. Must be an integer in the range 1-65535."
        )
        sys.exit(1)

    if port < 1 or port > 65535:
        logger.error(
            f"Invalid API_PORT: {port}. Must be in the range 1-65535."
        )
        sys.exit(1)

    return port


def resolve_guardrails_mode(header_value: Optional[str], default_mode: str) -> str:
    """Determine the effective guardrails mode for a request.

    - If header_value is "guarded" or "unguarded", use it.
    - Otherwise, fall back to default_mode.

    Args:
        header_value: The value of the X-Guardrails-Mode header, or None if absent.
        default_mode: The server's configured default mode.

    Returns:
        The resolved guardrails mode string.
    """
    if header_value in ("guarded", "unguarded"):
        return header_value
    return default_mode


def load_server_config() -> ServerConfig:
    """Load and validate server configuration from environment variables.

    Loads the .env file, reads configuration values, validates them,
    and returns a ServerConfig instance. Exits with error if any
    configuration is invalid.

    Returns:
        A validated ServerConfig instance.
    """
    # Load .env file
    load_dotenv()

    # Load and validate NVIDIA API key
    nvidia_api_key = validate_api_key(os.environ.get("NVIDIA_API_KEY"))

    # Load and validate guardrails mode
    guardrails_mode_raw = os.environ.get("GUARDRAILS_MODE", "guarded")
    guardrails_mode = validate_guardrails_mode(guardrails_mode_raw)

    # Load host
    host = os.environ.get("API_HOST", "0.0.0.0")

    # Load and validate port
    port_str = os.environ.get("API_PORT", "8000")
    port = validate_port(port_str)

    # Determine config directory (relative to this file's location)
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

    return ServerConfig(
        host=host,
        port=port,
        guardrails_mode=guardrails_mode,
        nvidia_api_key=nvidia_api_key,
        config_dir=config_dir,
    )
