from collections import OrderedDict
from typing import Dict, Tuple

from deid.llm_api import VLLMChat


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.O = OrderedDict()

    def get(self, key: str) -> str | bool | None:
        if key in self.O:
            self.O.move_to_end(key)
            return self.O[key]
        return None
    
    def put(self, key: str, value: str | bool) -> None:
        if key not in self.O and len(self.O) == self.cap:
            self.O.popitem(last=False)
        self.O[key] = value
        self.O.move_to_end(key)


class BaseDeid:
    raw2deid_cache = LRUCache(capacity=128)  # raw_text: str -> deid_text: str
    deid2eval_cache = LRUCache(capacity=128)  # deid_text: str -> deid_passed: bool

    def __init__(self):
        raise NotImplementedError

    def deid(self, raw_text: str, **kwargs) -> str:
        raise NotImplementedError

    def eval(self, deid_text: str, **kwargs) -> bool:
        raise NotImplementedError


class ExampleDeid(BaseDeid):
    def __init__(self):
        self.llm = VLLMChat(
            model_name="qwen3:0.6b",
            base_url="http://vllm:8106/v1"
        )
        self.prompt_deid = open(f"/app/prompts/example/deid.txt", "r").read()
        self.prompt_eval = open(f"/app/prompts/example/eval.txt", "r").read()
    
    def deid(self, raw_text: str) -> str:
        out = self.raw2deid_cache.get(raw_text)
        if out is not None:
            return out
        else:
            prompt = self.prompt_deid.replace("{{ input_text }}", raw_text)
            deid_text = self.llm.run(prompt)
            self.raw2deid_cache.put(raw_text, deid_text)
            return deid_text

    def eval(self, deid_text: str) -> bool:
        deid_passed = self.deid2eval_cache.get(deid_text)
        if deid_passed is not None:
            return deid_passed
        else:
            prompt = self.prompt_eval.replace("{{ input_text }}", deid_text)
            out = self.llm.run(prompt, response_format='{"has_pii": bool, "explanation": str}')
            deid_passed = isinstance(out, Dict) and "has_pii" in out and not out["has_pii"]
            self.deid2eval_cache.put(deid_text, deid_passed)
            return deid_passed
        