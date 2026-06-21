import ast
import json
import os
import re
import sys
from typing import Dict, List, Optional, Union, Type

import httpx
from openai import OpenAI
import requests
import yaml


class LLMAPI:
    def __init__(self, **kwargs):
        raise NotImplementedError

    def run(self, **kwargs) -> Union[Dict, str]:
        raise NotImplementedError


def llm_postprocess(out: str, to_dict: bool = False) -> Union[Dict, str]:
    out = re.sub(r"<think>[\s\S]*?</think>", "", out)
    if to_dict:
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
            return out
    else:
        return out


class VLLMChat(LLMAPI):
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.client = OpenAI(api_key="", base_url=base_url)

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

    def run(self, prompt: str, response_format: str = "", temperature: float = 0.7) -> Union[Dict, str]:
        """Pydantic response requires <json_str> specified in response_format e.g. '{"score": int}'"""
        if response_format:
            prompt_ = f"{prompt}\n**Response Format**: Only output a valid json format as below:\n{response_format}"
        else:
            prompt_ = prompt
        args = self._prepare_args(prompt_, temperature)
        response = self.client.chat.completions.create(**args)
        out = response.choices[0].message.content
        out = llm_postprocess(out, to_dict=bool(response_format))
        return out


class OllamaChat(LLMAPI):
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url

    def run(self, prompt: str, response_format: str = "") -> Union[Dict, str]:
        """Pydantic response requires <json_str> specified in response_format e.g. '{"score": int}'"""
        if response_format:
            prompt_ = f"{prompt}\n**Response Format**: Only output a valid json format as below:\n{response_format}"
        else:
            prompt_ = prompt
        payload = {"model": self.model_name, "prompt": prompt_, "stream": False}
        response = requests.post(self.base_url, json=payload)
        out = response.json()["response"]
        out = llm_postprocess(out, to_dict=bool(response_format))
        return out


class AzureOpenAIChatAPI(LLMAPI):
    """
    All support pydantic response in server side
    """
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-mini"):
        if os.path.isfile(api_key):
            with open(api_key, "r") as f:
                self.api_key = yaml.safe_load(f)["azure_openai"]
        else:
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


def init_llm(llm_cfg: Dict) -> LLMAPI:
    llm_cls = getattr(sys.modules[__name__], llm_cfg["cls_name"])
    llm = llm_cls(**llm_cfg["args"])
    return llm
