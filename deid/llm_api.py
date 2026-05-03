from abc import abstractmethod
import ast
import json
import os
import re
from typing import Dict, List, Optional, Union, Type
import yaml

import httpx
from openai import OpenAI
import requests

from deid.llm_tokens import get_token_count, get_approx_token_count


class AzureOpenAIChatAPI:
    """
    support pydantic response
    """
    def __init__(
            self,
            api_key: str,
            model_name: str = "gpt-4.1-mini"
        ):
        self.api_key = api_key
        self.azure_endpoint = f"https://project-emc-llm-foundry.openai.azure.com/openai/deployments/{model_name}/chat/completions?api-version=2024-10-21"
    
    def run(self, prompt: Union[str, List], response_format: Optional[Type] = None) -> str:
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        data = {"messages": messages, "max_tokens": None}
        if response_format:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema()
                }
            }    
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        response = requests.post(self.azure_endpoint, headers=headers, json=data)
        return response.json()["choices"][0]['message']['content']

    async def arun(self, prompt: Union[str, List], response_format: Optional[Type] = None) -> str:
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        data = {"messages": messages, "max_tokens": None}
        if response_format:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema()
                }
            }
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self.azure_endpoint,
                headers=headers,
                json=data
            )
        return response.json()["choices"][0]["message"]["content"]



class OllamaChat:
    """
    Does not support pydantic response, use prompt hint instead
    """
    def __init__(
            self,
            api_key: str = "",
            model_name: str = "qwen3:0.6b",
            base_url: str = f"{os.environ.get('OLLAMA_CHAT_BASE_URL', '')}/api/generate",
        ):
        self.model_name = model_name
        self.base_url = base_url
    
    def _postprocess(self, out: str) -> Dict:
        out = out.replace("```json", "").replace("```", "")
        try:
            out_ = out.replace("None", "null")
            return json.loads(out_)
        except:
            pass
        try:
            out_ = out.replace("null", "None")
            return ast.literal_eval(out_)
        except:
            print(f"Failed to parse LLM output: {out}")
            return {}

    def run(self, prompt: str, response_format: Optional[str] = None) -> Union[Dict, str]:
        if response_format:
            prompt_ = f"{prompt}\n**Response Format**: output must strictly follow the valid json as below - {response_format}"
        else:
            prompt_ = prompt
        payload = {"model": self.model_name, "prompt": prompt_, "stream": False}
        response = requests.post(self.base_url, json=payload)
        out = response.json()["response"]
        if response_format:
            out = self._postprocess(out)
        return out
    

class VLLMChat:
    """
    Does not support pydantic response, use prompt hint instead
    """
    token_limit = 4096

    def __init__(
            self,
            api_key: str = "",
            model_name: str = "qwen3:0.6b",
            base_url: str = os.environ.get('VLLM_CHAT_BASE_URL', '')
        ):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _prepare_args(self, prompt: Union[str, List], temperature: float) -> Dict:
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }

    def _postprocess(self, out: str) -> Dict:
        out = out.replace("```json", "").replace("```", "")
        try:
            out_ = out.replace("None", "null")
            return json.loads(out_)
        except:
            pass
        try:
            out_ = out.replace("null", "None")
            return ast.literal_eval(out_)
        except:
            print(f"Failed to parse LLM output: {out}")
            return {}
    
    def run(self, prompt: str, response_format: Optional[str] = None, temperature: float = 0.7) -> Union[Dict, str]:
        tokens = get_token_count(prompt)
        if tokens >= int(self.token_limit * 0.9):
            return f"[Error] Input text {tokens} tokens is too long for the model to process (limit: {self.token_limit} tokens)."
        
        prompt_ = prompt #recursive_summarization(self, prompt, tag="prompt_before_input_llm")
        if response_format:
            prompt_ += f"{prompt}\n**Response Format**: without any further explanation, also Do NOT include <think> tags. Only output a valid json format as below:\n{response_format}"
        args = self._prepare_args(prompt_, temperature)
        response = self.client.chat.completions.create(**args)
        out = response.choices[0].message.content
        out = re.sub(r"<think>[\s\S]*?</think>", "", out)
        if response_format:
            out = self._postprocess(out)
        elif "```markdown" in prompt:
            out = out.replace("```markdown", "").replace("```", "")
        return out


if __name__ == "__main__":
    api_keys = yaml.safe_load(open("/app/cfgs/api_keys.yaml", "r"))

    # Normal Chat
    if 1:
        # llm = AzureOpenAIChatAPI(api_keys["azure_openai"], "gpt-4.1-mini")
        # print(llm.run("How are you"))
        
        # llm = OllamaChat("http://ollama:11434/api/generate", "qwen3:8b")
        # print(llm.run("How are you"))

        # llm = VLLMChat()
        # print(llm.run("How are you"))

        pass