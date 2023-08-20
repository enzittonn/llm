from huggingface_hub import HfApi, list_models

# Use root method
models = list_models()

# Or configure a HfApi client
hf_api = HfApi(
    endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token="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", # Token is not persisted on the machine.
)
models = hf_api.list_models()
print(models)