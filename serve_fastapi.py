from typing import List, Dict
from typing_extensions import Annotated
import uuid
import time

from pydantic_settings import BaseSettings
from pydantic import Field
from fastapi import FastAPI, Depends

from invoker.api_types import ChatInput, ChatOutput
from invoker.model import InvokerPipeline


async def get_pipeline(model_path: str):
    return await InvokerPipeline.maybe_init(model_path=model_path)


class Settings(BaseSettings):
    invoker_model_name_or_path: str = Field("jeffrey-fong/invoker-13b", env="INVOKER_MODEL_NAME_OR_PATH")


app = FastAPI(title="Invoker")
settings = Settings()


@app.post('/chat/completions', response_model=ChatOutput)
def chat(req: ChatInput):
    breakpoint()
    invoker_pipeline: InvokerPipeline = Depends(get_pipeline(model_path=settings.invoker_model_name_or_path))
    id = str(uuid.uuid4())
    message = {"role": "assistant", "content": "Endpoint called"}
    created = int(time.time())
    return {"id": id}
    # return {"id": id, "object": "chat.completion", "created": created, "choices": [{"message": message, "finish_reason": "stop"}]}
    

@app.on_event("startup")
async def startup():
    _ = await get_pipeline(model_path=settings.invoker_model_name_or_path)