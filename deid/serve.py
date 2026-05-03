import os
from typing import Dict, List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel
import yaml

from deid.async_funcs import async_executor
from deid.llm_api import AzureOpenAIChatAPI, OllamaChat, VLLMChat
from deid.llm_pydantic import schema_to_model, schema_to_json_str


cfg = yaml.safe_load(open(os.getenv("API_KEYS_PATH"), "r"))
llm_cloud = AzureOpenAIChatAPI(
    api_key=cfg["azure_openai"],
    model_name="gpt-4.1-mini"
)
llm_local_cpu = OllamaChat(
    model_name="qwen3:0.6b"
)
llm_local_gpu = VLLMChat(
    model_name="qwen3:0.6b"
)
app = FastAPI()


class APIRequest(BaseModel):
    prompt: Union[str, List]
    response_format_dict: Optional[Dict] = None


@app.post("/cloud_api")
def cloud_api(r: APIRequest):
    response_format = schema_to_model("custom", r.response_format_dict) \
        if r.response_format_dict else None
    return llm_cloud.run(prompt=r.prompt, response_format=response_format)


@app.post("/async_cloud_api")
def async_cloud_api(r: List[APIRequest]):
    return async_executor(
        llm_cloud.arun,
        [
            {
                "prompt": ri.prompt,
                "response_format": schema_to_model("custom", ri.response_format_dict) \
                    if ri.response_format_dict else None
            }
            for ri in r
        ]
    )


@app.post("/local_api_cpu")
def local_api_cpu(r: APIRequest):
    """Ollama"""
    response_format = schema_to_json_str("custom", r.response_format_dict) \
        if r.response_format_dict else None
    return llm_local_cpu.run(prompt=r.prompt, response_format=response_format)


@app.post("/local_api_gpu")
def local_api_gpu(r: APIRequest):
    """VLLM"""
    response_format = schema_to_json_str("custom", r.response_format_dict) \
        if r.response_format_dict else None
    return llm_local_gpu.run(prompt=r.prompt, response_format=response_format)
