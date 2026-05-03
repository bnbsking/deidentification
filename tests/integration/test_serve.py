import requests


def test_cloud_api():
    response = requests.post(
        "http://localhost:8006/cloud_api",
        json={"prompt": "What is the capital of France?"}
    )
    print(response.json())  # The capital of France is Paris.


def test_cloud_api_pydantic():
    response_format_dict = {
        "name": "str",
        "age": "int",
        "hobbies": ["str"]
    }
    response = requests.post(
        "http://localhost:8006/cloud_api",
        json={
            "prompt": "Generate a fake person information",
            "response_format_dict": response_format_dict
        }
    )
    print(response.json())
    # {"name":"Sarah Thompson","age":34,"hobbies":["painting","cycling","cooking"]}


def test_async_cloud_api():
    response = requests.post(
        "http://localhost:8006/async_cloud_api",
        json=[
            {"prompt": "What is the next day of Sunday?"},
            {"prompt": "How much is 15 * 12"}
        ]
    )
    print(response.json())
    # ['The next day after Sunday is Monday.', '15 * 12 = 180']


def test_async_cloud_api_pydantic():
    response_format_dict = {
        "name": "str",
        "age": "int",
        "hobbies": ["str"]
    }
    response = requests.post(
        "http://localhost:8006/async_cloud_api",
        json=[
            {
                "prompt": "Generate a fake person information",
                "response_format_dict": response_format_dict
            },
            {
                "prompt": "Generate another fake person information",
                "response_format_dict": response_format_dict
            }
        ]
    )
    print(response.json())
    # [{"name":"Sarah Thompson","age":34,"hobbies":["painting","cycling","cooking"]},
    #  {"name":"Michael Johnson","age":28,"hobbies":["hiking","photography","gaming"]}]


if __name__ == "__main__":
    test_cloud_api()
    test_cloud_api_pydantic()
    test_async_cloud_api()
    test_async_cloud_api_pydantic()
