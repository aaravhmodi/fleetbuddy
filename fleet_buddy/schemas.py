"""Shapes of the JSON request bodies. FastAPI checks incoming requests against these automatically."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Shapes of the JSON request bodies. FastAPI checks incoming requests against these automatically.
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    previous_trace_id: str | None = None


class EvalCase(BaseModel):
    messages: list[Message] = Field(min_length=1)
    expected: str
