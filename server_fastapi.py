import time
import uuid

from fastapi import FastAPI
from pydantic import Field
from pydantic_settings import BaseSettings

from invoker.api_types import ChatInput, ChatOutput
from invoker.model import InvokerPipeline
from invoker.utils.enum_tags import ModelType


class Settings(BaseSettings):
    invoker_model_type: ModelType = Field("hf", env="INVOKER_MODEL_TYPE")
    invoker_model_name_or_path: str = Field("jeffrey-fong/invoker-13b", env="INVOKER_MODEL_NAME_OR_PATH")


async def get_pipeline(model_path: str, model_type: ModelType):
    return await InvokerPipeline.maybe_init(model_path=model_path, model_type=model_type)


app = FastAPI(title="Invoker")
settings = Settings()


@app.post("/chat/completions", response_model=ChatOutput)
async def chat(req: ChatInput):
    id = str(uuid.uuid4())
    invoker_pipeline: InvokerPipeline = await get_pipeline(
        model_path=settings.invoker_model_name_or_path, model_type=settings.invoker_model_type
    )
    prompt = invoker_pipeline.format_message(messages=req.messages, functions=req.functions)
    choices = invoker_pipeline.generate(input_text=prompt, params={"temperature": req.temperature, "top_p": req.top_p})
    created = int(time.time())
    return {"id": id, "created": created, "choices": choices}


@app.on_event("startup")
async def startup():
    _ = await get_pipeline(model_path=settings.invoker_model_name_or_path, model_type=settings.invoker_model_type)
