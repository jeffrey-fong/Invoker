from typing import List, Dict
from typing_extensions import Annotated
import uuid

from fastapi import FastAPI, Depends
from pydantic import Field
from pydantic_settings import BaseSettings

from invoker.api_types import ChatInput, ChatOutput
from invoker.model import InvokerPipeline


class Settings(BaseSettings):
    model_path: str = Field("jeffrey-fong/invoker-13b", env="MODEL_PATH")


app = FastAPI(title="Invoker")
settings = Settings()


async def get_pipeline(model_path: str):
    return await InvokerPipeline.maybe_init(model_path=model_path)


@app.post('/chat', response_model=ChatOutput)
async def chat(req: ChatInput, invoker_pipeline: InvokerPipeline = Depends(get_pipeline)):
    id = str(uuid.uuid4())
    # prompt = invoker_pipeline.format_message()
    output = await invoker_pipeline.generate(input_text=req.messages[0].content, params=[])
    message = {"role": "assistant", "content": output}
    return {"id": id, "choices": [{"message": message, "finish_reason": "stop"}]}


@app.on_event("startup")
async def startup():
    _ = await get_pipeline(model_path=settings.model_path)