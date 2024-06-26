import openai
import os
from omegaconf import OmegaConf
from retry import retry
from openai.error import Timeout

class OpenAIChat:
    def __init__(self, conf):
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        self.client = openai.ChatCompletion()
        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.verbose = True
        self.message_history = []

    def _validate_conf(self):
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        except Exception:
            raise ValueError("No API keys provided")
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")

    @retry(Timeout, tries=3)
    def generate(
        self,
        prompt,
        stop=None,
        max_length=None,
        keep_message_history=True,
        request_timeout=40,
        temperature=None,
    ):
        params = OmegaConf.to_object(self.generation_params)

        # Override stop if provided
        if stop is None:
            stop = self.generation_params.stop
        params["stop"] = stop

        # Override max_length if provided
        if max_length is not None:
            params["max_tokens"] = max_length
        
        # Override temperature if provided
        if temperature is not None:
            params["temperature"] = temperature

        messages = self.message_history.copy()
        # Add system message if no messages
        if len(messages) == 0:
            messages.append({"role": "system", "content": self.llm_conf.system_message})


        # Add current message
        messages.append({"role": "user", "content": prompt})
        params["messages"] = messages

        params["request_timeout"] = request_timeout
        
        self.response = self.client.create(**params)
        text_response = self.response.choices[0].message.content

        # Update message history
        if keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})
        return text_response
