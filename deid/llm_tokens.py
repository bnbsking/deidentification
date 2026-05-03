import re

import tiktoken


def get_token_count(text: str, model: str = "gpt-4.1") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def get_approx_token_count(text: str) -> int:
    """
    Estimate token count for mixed Chinese/English text.
    Heuristic tuned for modern BPE tokenizers (Qwen/LLaMA/GPT style).
    """
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'[A-Za-z]+', text)
    numbers = re.findall(r'\d+', text)
    punctuation = re.findall(r'[^\w\s\u4e00-\u9fff]', text)

    tokens = len(chinese_chars) * 1\
        + int(len(english_words) * 1.3)\
        + len(numbers) * 1\
        + len(punctuation) * 1

    return tokens
