from typing import List, Dict

from fastapi import FastAPI

from invoker.api_types import ChatInput, ChatOutput


app = FastAPI(title="Invoker")


@app.post('/chat', response_model=ChatOutput)
def chat(req: ChatInput):
    return {"response": "Endpoint called"}