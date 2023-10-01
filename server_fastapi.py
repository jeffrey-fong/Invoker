import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import Field
from pydantic_settings import BaseSettings

from invoker.api_types import ChatInput, ChatOutput, ChatStreamOutput, StreamChoice
from invoker.model import InvokerPipeline
from invoker.utils.enum_tags import ModelType


class Settings(BaseSettings):
    invoker_model_type: ModelType = Field("hf", env="INVOKER_MODEL_TYPE")
    invoker_model_name_or_path: str = Field("jeffrey-fong/invoker-13b", env="INVOKER_MODEL_NAME_OR_PATH")


async def get_pipeline(model_path: str, model_type: ModelType):
    return await InvokerPipeline.maybe_init(model_path=model_path, model_type=model_type)


app = FastAPI(title="Invoker")
settings = Settings()


@app.post("/chat/completions")
async def chat(req: ChatInput):
    id = str(uuid.uuid4())
    invoker_pipeline: InvokerPipeline = await get_pipeline(
        model_path=settings.invoker_model_name_or_path, model_type=settings.invoker_model_type
    )
    prompt = invoker_pipeline.format_message(messages=req.messages, functions=req.functions)
    created = int(time.time())
    if not req.stream:
        choices = invoker_pipeline.generate(
            input_text=prompt, params={"temperature": req.temperature, "top_p": req.top_p}
        )
        return ChatOutput(id=id, created=created, choices=choices)
    else:
        response_generator = invoker_pipeline.generate_stream(
            input_text=prompt, params={"temperature": req.temperature, "top_p": req.top_p}
        )

        def get_streaming_response():
            i = 0
            for chunk in response_generator:
                choices = [StreamChoice(delta={"role": "assistant", "content": f"test{i}"}, finish_reason=None)]
                i += 1
                yield "data: " + ChatStreamOutput(id=id, created=created, choices=choices).model_dump_json(
                    exclude_unset=True
                ) + "\n\n"

        return StreamingResponse(content=get_streaming_response(), media_type="text/event-stream")


@app.on_event("startup")
async def startup():
    _ = await get_pipeline(model_path=settings.invoker_model_name_or_path, model_type=settings.invoker_model_type)
