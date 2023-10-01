from typing import List, Optional

from pydantic import BaseModel


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class Parameters(BaseModel):
    type: str = "object"
    properties: dict
    required: list


class Message(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class Function(BaseModel):
    name: str
    description: str
    parameters: Parameters


class ChatInput(BaseModel):
    model: str
    messages: List[Message]
    functions: Optional[List[Function]] = None
    temperature: float = 0.5
    top_p: float = 1.0
    stream: bool = False


class Choice(BaseModel):
    message: Message
    finish_reason: str = "stop"


class StreamChoice(BaseModel):
    delta: Message
    finish_reason: Optional[str]


class ChatOutput(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[Choice]


class ChatStreamOutput(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    choices: List[StreamChoice]
