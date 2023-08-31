from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    message: str
    

class Function(BaseModel):
    function: str


class ChatInput(BaseModel):
    model: str
    messages: List[Message]
    functions: List[Function]
    temperature: float = 0.5
    top_p: float = 1.0
    

class ChatOutput(BaseModel):
    response: str