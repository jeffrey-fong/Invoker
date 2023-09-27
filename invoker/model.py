from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from transformers import LlamaForCausalLM, LlamaTokenizer

from invoker.api_types import Function, Message
from invoker.utils.enum_tags import ModelType


class InvokerPipeline:
    # Singleton instance
    _pipeline = None

    def __init__(self, model_path: str, model_type: ModelType):
        # Load model
        self._model_type = model_type
        if model_type == ModelType.exllamav2:
            config = ExLlamaV2Config()
            config.model_dir = model_path
            config.prepare()
            model = ExLlamaV2(config)
            model.load()
            self._tokenizer = ExLlamaV2Tokenizer(config)
            cache = ExLlamaV2Cache(model)
            self._generator = ExLlamaV2BaseGenerator(model, cache, self._tokenizer)
            self._generator.warmup()
            self._settings = ExLlamaV2Sampler.Settings()
            self._settings.token_repetition_penalty = 1.0
        else:
            self._tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
            self._model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    def format_message(self, messages: List[Message], functions: Optional[List[Function]]):
        prompt = "Available Functions:"
        if functions is not None:
            for function in functions:
                prompt += f"\n```json\n{json.dumps(function.model_dump(mode='json'))}\n```"
        else:
            prompt += "\nNone"
        prompt += (
            "\n\nA chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "The assistant calls functions with appropriate input when necessary."
        )
        for message in messages:
            if message.role == "assistant":
                prompt += f"\n{message.role.upper()}: ```" + "{"
                if message.content is None:
                    prompt += (
                        '"content": null, "function_call": {'
                        + f'"name": "{message.function_call.name}", "arguments": {message.function_call.arguments}'
                        + "}```"
                    )
                else:
                    prompt += f'"content": {message.content}' + ', "function_call": None}```'
            elif message.role == "function":
                prompt += (
                    f"\n{message.role.upper()}: ```"
                    + "{"
                    + f'"name": "{message.name}", "content": {message.content}'
                    + "}```"
                )
            else:
                prompt += f"\n{message.role.upper()}: {message.content}"
        prompt += "\nASSISTANT:"
        return prompt

    def generate(self, input_text: str, params: Dict[str, Any]) -> str:
        temperature, top_p = params.get("temperature"), params.get("top_p")
        if self._model_type == ModelType.exllamav2:
            self._settings.temperature, self._settings.top_p = temperature, top_p
            raw_output = self._generator.generate_simple(input_text, self._settings, num_tokens=512)
        else:
            input_ids = self._tokenizer(input_text, return_tensors="pt").input_ids.cuda()
            do_sample = True if temperature > 0.0 else False
            output_ids = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
            )
            raw_output = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output = raw_output[len(input_text) :]
        choices = self._postprocess(text=output)
        return choices

    def _postprocess(self, text):
        output_json = json.loads(re.search(r"```(.*?)```?", text, re.DOTALL).group(1))
        if output_json["function_call"] is not None:
            choices = [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": output_json["function_call"]["name"],
                            "arguments": output_json["function_call"]["arguments"]
                            if isinstance(output_json["function_call"]["arguments"], str)
                            else json.dumps(output_json["function_call"]["arguments"]),
                        },
                    },
                    "finish_reason": "function_call",
                }
            ]
        else:
            choices = [
                {
                    "message": {"role": "assistant", "content": output_json["content"]},
                    "finish_reason": "stop",
                }
            ]
        return choices

    @classmethod
    async def maybe_init(cls, model_path: str, model_type: ModelType) -> InvokerPipeline:
        if cls._pipeline is None:
            cls._pipeline = InvokerPipeline(model_path=model_path, model_type=model_type)
        if cls._pipeline is not None:
            return cls._pipeline
        else:
            raise ValueError("Pipeline could not be initialized!")
