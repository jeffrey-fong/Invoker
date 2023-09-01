from typing import List, Dict
from typing_extensions import Annotated
import uuid

from fastapi import FastAPI, Depends

from invoker.api_types import ChatInput, ChatOutput
from invoker.model import InvokerPipeline


app = FastAPI(title="Invoker")


async def get_pipeline(model_path: str):
    return await InvokerPipeline.maybe_init(model_path=model_path)


@app.post('/chat', response_model=ChatOutput)
def chat(req: ChatInput, invoker_pipeline: InvokerPipeline = Depends(get_pipeline)):
    id = str(uuid.uuid4())
    # output = invoker_pipeline.generate(input_text=req.)
    message = {"role": "assistant", "content": "Endpoint called"}
    return {"id": id, "choices": [{"message": message, "finish_reason": "stop"}]}