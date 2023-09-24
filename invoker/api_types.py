from typing import List, Optional

from pydantic import BaseModel


class FunctionCall(BaseModel):
    name: str
    arguments: str
    
    
class Parameters(BaseModel):
    type: str = "object"
    properties: dict
    required: list


class Message(BaseModel):
    role: str
    content: Optional[str]
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
    temperature: float = 0.7
    top_p: float = 1.0
    

class Choice(BaseModel):
    message: Message
    finish_reason: str = "stop"


class ChatOutput(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[Choice]