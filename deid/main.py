from typing import Dict

from deid import deid_collections


deid_registry = {}  # str -> deid_collections.BaseDeid object


def register(key: str, deid_cls_name: str, deid_cls_args: Dict):
    deid_cls = getattr(deid_collections, deid_cls_name)
    deid_registry[key] = deid_cls(**deid_cls_args)


def unregister(key: str):
    if key in deid_registry:
        del deid_registry[key]


def run(key: str, raw_text: str) -> str:
    if key not in deid_registry:
        raise ValueError(f"Deid class for key {key} is not registered.")
    else:
        obj = deid_registry[key]
    
    deid_text = obj.deid(raw_text)
    if obj.eval(deid_text):
        return deid_text
    else:
        return "[Deid_failed]"
        