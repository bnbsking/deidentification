from pydantic import BaseModel
import yaml

from deid.llm_api import AzureOpenAIChatAPI, VLLMChat, OllamaChat


class TestAzureOpenAIChatAPI:
    def __init__(self):
        api_keys = yaml.safe_load(open("/app/cfgs/api_keys.yaml", "r"))
        self.llm = AzureOpenAIChatAPI(api_keys["azure_openai"], "gpt-4.1-mini")

    def test_run(self):
        out = self.llm.run("How are you")
        print(out)
        # I'm doing well, thank you! How can I assist you today?
    
    def test_run_with_response_format(self):
        class ResponseModel(BaseModel):
            answer: int
            explanation: str
        out = self.llm.run("What is 1+1? Answer with answer and explanation.", response_format=ResponseModel)
        print(out)
        # {"answer":2, "explanation": "Adding 1 and 1 together results in 2 because 1 plus 1 equals 2."}


class TestVLLMChat:
    def __init__(self):
        self.llm = VLLMChat()

    def test_run(self):
        out = self.llm.run("How are you")
        print(out)
        # I'm doing well, thank you! How can I assist you today?

    def test_run_with_response_format(self):
        response_format = '{"answer": int, "explanation": str}'
        out = self.llm.run("What is 1+1? Answer with answer and explanation.", response_format=response_format)
        print(out)
        # {'answer': 2, 'explanation': '1 plus 1 equals 2 because you are adding two equal numbers.'}


class TestOllamaChat:
    def __init__(self):
        self.llm = OllamaChat()

    def test_run(self):
        out = self.llm.run("How are you")
        print(out)
        # I'm doing well, thank you! How can I assist you today?

    def test_run_with_response_format(self):
        response_format = '{"answer": int, "explanation": str}'
        out = self.llm.run("What is 1+1? Answer with answer and explanation.", response_format=response_format)
        print(out)
        # {'answer': 2, 'explanation': '1+1=2'}


if __name__ == "__main__":
    obj = TestAzureOpenAIChatAPI()
    obj.test_run()
    obj.test_run_with_response_format()

    obj = TestVLLMChat()
    obj.test_run()
    obj.test_run_with_response_format()

    # obj = TestOllamaChat()
    # obj.test_run()
    # obj.test_run_with_response_format()
