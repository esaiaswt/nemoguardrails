"""Pydantic models for the OpenAI-compatible Chat Completions API.

Defines request validation and response formatting models used by the
FastAPI server (api_server.py) to serve nvidia_garak's OpenAICompatGenerator.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator


# --- Request Models ---


class ChatMessage(BaseModel):
    """A single message in the chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request payload."""

    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None

    class Config:
        extra = "ignore"

    @validator("messages")
    def messages_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("messages must contain at least one message")
        return v


# --- Response Models ---


class ResponseMessage(BaseModel):
    """The assistant's response message."""

    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage statistics for the completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response payload."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# --- Error Models ---


class ErrorDetail(BaseModel):
    """Details of an API error."""

    type: str
    message: str


class ErrorResponse(BaseModel):
    """Structured error response envelope."""

    error: ErrorDetail
