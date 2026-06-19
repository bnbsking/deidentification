from deid import deid_collections


deid_registry = {}  # name -> deid.BaseDeid


def register(key: str, deid_cls_name: str, **kwargs):
    cls = getattr(deid_collections, deid_cls_name)
    deid_registry[key] = cls(**kwargs)


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
        raise ValueError("Deidentified content did not pass evaluation.")
