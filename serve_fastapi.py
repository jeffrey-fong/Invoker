from typing import List, Dict
from typing_extensions import Annotated
import uuid

from fastapi import FastAPI, Depends
import uvicorn
import typer

from invoker.api_types import ChatInput, ChatOutput
from invoker.model import InvokerPipeline


app = FastAPI(title="Invoker")


async def get_pipeline(model_path: str):
    return await InvokerPipeline.maybe_init(model_path=model_path)


@app.post('/chat', response_model=ChatOutput)
def chat(req: ChatInput, invoker_pipeline: InvokerPipeline = Depends(get_pipeline)):
    id = str(uuid.uuid4())
    message = {"role": "assistant", "content": "Endpoint called"}
    return {"id": id, "choices": [{"message": message, "finish_reason": "stop"}]}


def main(
    model_path: Annotated[str, typer.Argument(help="The path to the model to use")] = "jeffrey-fong/invoker-13b"
):
    invoker_pipeline = get_pipeline(model_path=model_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    typer.run(main)