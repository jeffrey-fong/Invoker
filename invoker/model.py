from __future__ import annotations
from typing import List, Dict, Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, LlamaForCausalLM, LlamaTokenizer


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


class InvokerPipeline:
    # Singleton instance
    _pipeline = None

    def __init__(self, model_path: str):
        self._tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        # Load model
        self._model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

    async def generate(
        self, input_text: str, params: List[Dict[str, Any]]
    ) -> str:
        prompt = f"""Available Function Headers:
```python
# Plan a trip based on user's interests
def plan_trip(
	# The destination of the trip
	destination: str,
	# The duration of the trip in days
	duration: int,
    # The interests based on which the trip will be planned
    interests: List[str]
) -> Any:
```

A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary. 

USER: I want to plan a 7-day trip to Paris with a focus on art and culture
ASSISTANT:"""
        # Tokenize the input
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        # Run the model to infer an output
        stop_words_ids = [self._tokenizer.encode(stop_word, add_special_tokens=False) for stop_word in ["\nUSER:", "\nASSISTANT:", "\nFUNCTION:"]]
        stopping_criteria = StoppingCriteriaList([StopWordsCriteria(stops=stop_words_ids)])
        output_ids = self._model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.9,temperature=0.001, stopping_criteria=stopping_criteria)
        output = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output

    @classmethod
    async def maybe_init(cls, model_path: str) -> InvokerPipeline:
        if cls._pipeline is None:
            cls._pipeline = InvokerPipeline(model_path=model_path)
        if cls._pipeline is not None:
            return cls._pipeline
        else:
            raise ValueError("Pipeline could not be initialized!")