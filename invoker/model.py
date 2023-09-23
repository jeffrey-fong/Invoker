from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import re

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, LlamaForCausalLM, LlamaTokenizer

from invoker.api_types import Message, Function


JSON_TO_PY_MAPPING = {
    "string": "str", 
    "integer": "int", 
    "boolean": "bool", 
    "array": "List", 
    "number": "float", 
    "object": "Dict", 
    "null": None
}


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stops=[]):
        StoppingCriteria.__init__(self)
        self.stops = stops
        # Check if stop word starts with "\" and remove first token (empty space)
        # This is only applicable to vicuna-13b and Nous-Hermes-13b tokenizers
        for i in range(len(self.stops)):
            if self.stops[i][0] == 29871:
                self.stops[i] = self.stops[i][1:]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        inputs = input_ids[0].tolist()
        for stop in self.stops:
            if len(inputs) >= len(stop) and inputs[-len(stop) :] == stop:
                return True
        return False
    
    
def convert_json_schema_to_py(json_schema: Dict[str, str]):
    if json_schema["type"] == "array":
        type_hint = f"{JSON_TO_PY_MAPPING['array']}[{JSON_TO_PY_MAPPING[json_schema['item']]}]"
    else:
        type_hint = JSON_TO_PY_MAPPING[json_schema["type"]]
    return type_hint


class InvokerPipeline:
    # Singleton instance
    _pipeline = None

    def __init__(self, model_path: str):
        self._tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        # Load model
        self._model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        
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
                    prompt += "\"content\": null, \"function_call\": {" + f"\"name\": \"{message.function_call.name}\", \"arguments\": {message.function_call.arguments}" + "}```"
                else:
                    prompt += f"\"content\": {message.content}" + ", \"function_call\": None}```"
            elif message.role == "function":
                prompt += f"\n{message.role.upper()}: ```" + "{" + f"\"name\": \"{message.name}\", \"content\": {message.content}" + "}```"
            else:
                prompt += f"\n{message.role.upper()}: {message.content}"
        prompt += "\nASSISTANT:"
        return prompt

    def generate(
        self, input_text: str, params: Dict[str, Any]
    ) -> str:
        # Tokenize the input
        input_ids = self._tokenizer(input_text, return_tensors="pt").input_ids.cuda()
        # Run the model to infer an output
        temperature, top_p = params.get("temperature"), params.get("top_p")
        do_sample = True if temperature > 0.0 else False
        output_ids = self._model.generate(
            input_ids=input_ids, max_new_tokens=512, do_sample=do_sample, top_p=top_p, temperature=temperature
        )
        raw_output = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output = raw_output[len(input_text):]
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
                            "arguments": output_json["function_call"]["arguments"] if isinstance(output_json["function_call"]["arguments"], str) else json.dumps(output_json["function_call"]["arguments"])
                        }
                    },
                    "finish_reason": "function_call"
                }
            ]
        else:
            choices = [
                {
                    "message": {"role": "assistant", "content": output_json["content"]},
                    "finish_reason": "stop"
                }
            ]
        return choices

    @classmethod
    async def maybe_init(cls, model_path: str) -> InvokerPipeline:
        if cls._pipeline is None:
            cls._pipeline = InvokerPipeline(model_path=model_path)
        if cls._pipeline is not None:
            return cls._pipeline
        else:
            raise ValueError("Pipeline could not be initialized!")