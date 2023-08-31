from typing import List, Dict
import uuid

from fastapi import FastAPI

from invoker.api_types import ChatInput, ChatOutput


app = FastAPI(title="Invoker")


@app.post('/chat', response_model=ChatOutput)
def chat(req: ChatInput):
    id = str(uuid.uuid4())
    message = {"role": "assistant", "content": "Endpoint called"}
    return {"id": id, "choices": [{"message": message, "finish_reason": "stop"}]}