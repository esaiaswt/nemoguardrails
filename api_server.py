"""FastAPI server entry point for the OpenAI-compatible Chat Completions API.

Exposes POST /v1/chat/completions and GET /health endpoints for nvidia_garak
vulnerability scanning. Supports guarded (NeMo Guardrails) and unguarded
(direct passthrough) modes.
"""

import logging
import os
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api_config import ServerConfig, load_server_config, resolve_guardrails_mode
from api_handlers import (
    DEFAULT_MODEL,
    configure_logging,
    get_health_status,
    handle_chat_completion,
)
from api_models import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse

logger = logging.getLogger("api_server")

# Module-level state
_rails = None  # LLMRails instance
_direct_client = None  # OpenAI client
_config: Optional[ServerConfig] = None


VALID_ROLES = {"system", "user", "assistant"}


def _validate_messages(body: dict) -> Optional[JSONResponse]:
    """Validate the messages field in the request body.

    Checks for:
    - messages field presence
    - messages is a non-empty list
    - Each message has role and content fields
    - Each message role is valid

    Args:
        body: The parsed JSON request body.

    Returns:
        A JSONResponse with 422 status if validation fails, or None if valid.
    """
    # Check messages field exists
    if "messages" not in body:
        return JSONResponse(
            status_code=422,
            content={"error": {"type": "validation_error", "message": "Field 'messages' is required."}},
        )

    messages = body["messages"]

    # Check messages is a list
    if not isinstance(messages, list):
        return JSONResponse(
            status_code=422,
            content={"error": {"type": "validation_error", "message": "Field 'messages' must be a list."}},
        )

    # Check messages is non-empty
    if len(messages) == 0:
        return JSONResponse(
            status_code=422,
            content={"error": {"type": "validation_error", "message": "messages must contain at least one message"}},
        )

    # Validate each message
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return JSONResponse(
                status_code=422,
                content={"error": {"type": "validation_error", "message": f"Message at index {i} must be an object."}},
            )

        # Check role field
        if "role" not in msg:
            return JSONResponse(
                status_code=422,
                content={"error": {"type": "validation_error", "message": f"Message at index {i} is missing the 'role' field."}},
            )

        # Check content field
        if "content" not in msg:
            return JSONResponse(
                status_code=422,
                content={"error": {"type": "validation_error", "message": f"Message at index {i} is missing the 'content' field."}},
            )

        # Validate role value
        role = msg["role"]
        if role not in VALID_ROLES:
            return JSONResponse(
                status_code=422,
                content={"error": {"type": "validation_error", "message": f"Message at index {i} has invalid role '{role}'. Must be one of: system, user, assistant."}},
            )

    return None


def create_app() -> FastAPI:
    """Factory function that creates and configures the FastAPI app.

    Registers routes for chat completions and health check,
    plus a global exception handler for unhandled errors.

    Returns:
        A configured FastAPI application instance.
    """
    app = FastAPI(title="NeMo Guardrails API", version="1.0.0")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint."""
        # Parse JSON body
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=422,
                content={"error": {"type": "validation_error", "message": "Request body must be valid JSON."}},
            )

        # Custom message validation with index reporting
        validation_error = _validate_messages(body)
        if validation_error is not None:
            return validation_error

        # Parse into Pydantic model (handles optional field range validation)
        try:
            chat_request = ChatCompletionRequest(**body)
        except Exception as exc:
            # Extract meaningful validation error message
            error_msg = str(exc)
            return JSONResponse(
                status_code=422,
                content={"error": {"type": "validation_error", "message": error_msg}},
            )

        # Resolve guardrails mode from header or default
        header_value = request.headers.get("X-Guardrails-Mode")
        mode = resolve_guardrails_mode(header_value, _config.guardrails_mode)

        # Process the request
        result = await handle_chat_completion(
            request=chat_request,
            mode=mode,
            rails=_rails,
            direct_client=_direct_client,
        )

        # Handle response
        if isinstance(result, tuple):
            # Error response: (status_code, error_dict)
            status_code, error_body = result
            return JSONResponse(status_code=status_code, content=error_body)

        # Success response
        return JSONResponse(status_code=200, content=result.model_dump())

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        status_code, body = get_health_status(
            rails=_rails,
            direct_client=_direct_client,
            default_mode=_config.guardrails_mode if _config else "guarded",
            model_name=DEFAULT_MODEL,
        )
        return JSONResponse(status_code=status_code, content=body)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler for unhandled exceptions.

        Returns a sanitized 500 error response without exposing internals.
        """
        from api_handlers import sanitize_message

        logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "message": sanitize_message(str(exc)) if str(exc) else "An internal error occurred.",
                }
            },
        )

    return app


def main():
    """Main entry point: load config, initialize components, start server."""
    global _rails, _direct_client, _config

    # Configure logging first
    configure_logging()

    # Load and validate server configuration (exits on failure)
    _config = load_server_config()

    # Check config directory and config.yml exist
    config_dir = _config.config_dir
    config_yml_path = os.path.join(config_dir, "config.yml")

    if not os.path.isdir(config_dir):
        logger.error(f"Configuration directory not found: {config_dir}")
        sys.exit(1)

    if not os.path.isfile(config_yml_path):
        logger.error(f"config.yml not found in configuration directory: {config_dir}")
        sys.exit(1)

    # Initialize LLMRails from config directory
    try:
        from nemoguardrails import LLMRails, RailsConfig

        rails_config = RailsConfig.from_path(config_dir)
        _rails = LLMRails(rails_config)
        logger.info("LLMRails initialized successfully.")
    except ImportError:
        logger.error("nemoguardrails package is not installed.")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to initialize LLMRails: {exc}")
        sys.exit(1)

    # Initialize OpenAI direct client with NVIDIA NIM base URL
    try:
        from openai import OpenAI

        _direct_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=_config.nvidia_api_key,
        )
        logger.info("OpenAI direct client initialized successfully.")
    except ImportError:
        logger.error("openai package is not installed.")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to initialize OpenAI client: {exc}")
        sys.exit(1)

    # Create the FastAPI app
    app = create_app()

    # Start uvicorn
    logger.info(f"Starting API server on {_config.host}:{_config.port}")
    uvicorn.run(app, host=_config.host, port=_config.port)


if __name__ == "__main__":
    main()
