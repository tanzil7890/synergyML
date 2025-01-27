import warnings
from synergyml.llm.gpt.clients.openai.completion import (
    get_chat_completion as _oai_get_chat_completion,
)
from synergyml.llm.gpt.clients.llama_cpp.completion import (
    get_chat_completion as _llamacpp_get_chat_completion,
)
from synergyml.llm.gpt.utils import split_to_api_and_model
from synergyml.config import SynergyMLConfig as _Config


def get_chat_completion(
    messages: dict,
    openai_key: str = None,
    openai_org: str = None,
    model: str = "gpt-3.5-turbo",
    json_response: bool = False,
):
    """Gets a chat completion from the OpenAI compatible API."""
    api, model = split_to_api_and_model(model)
    if api == "gguf":
        return _llamacpp_get_chat_completion(messages, model)
    else:
        url = _Config.get_gpt_url()
        if api == "openai" and url is not None:
            warnings.warn(
                f"You are using the OpenAI backend with a custom URL: {url}; did you mean to use the `custom_url` backend?\nTo use the OpenAI backend, please remove the custom URL using `synergymlConfig.reset_gpt_url()`."
            )
        elif api == "custom_url" and url is None:
            raise ValueError(
                "You are using the `custom_url` backend but no custom URL was provided. Please set it using `synergymlConfig.set_gpt_url(<url>)`."
            )
        # Only use json_response for models that support it (gpt-4-turbo and gpt-3.5-turbo)
        use_json = json_response and model in ["gpt-4-turbo-preview", "gpt-3.5-turbo"]
        return _oai_get_chat_completion(
            messages,
            openai_key,
            openai_org,
            model,
            api=api,
            json_response=use_json,
        )
