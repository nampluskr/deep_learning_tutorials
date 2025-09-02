```python
XXXX_REGISTRY = {"type1": Type1Xxxx, "type2": Type2Xxxx, "type3": Type3Xxxx}

def build_xxxx(xxxx_type, **xxxx_params):
    xxxx_type = xxxx_type.lower()
    if xxxx_type not in XXXX_REGISTRY:
        available_xxxxs = list(XXXX_REGISTRY.keys())
        raise ValueError(f"Unknown xxxx: {xxxx_type}. Available xxxxs: {available_xxxxs}")

    xxxx = XXXX_REGISTRY.get(xxxx_type)
    params = {'lr': 0.001}
    params.update(xxxx_params)
    return xxxx(**params)
```
