import openai
import os
from omegaconf import OmegaConf
from retry import retry

from openai.error import Timeout

class OpenAI:
    def __init__(self, conf):
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        self.client = openai.Completion()
        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.verbose = True

    def _validate_conf(self):
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        except Exception:
            raise ValueError("No API keys provided")
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")
        if self.generation_params.n > 1 and self.generation_params.stream:
            raise ValueError("Cannot stream results with n > 1")
        if self.generation_params.best_of > 1 and self.generation_params.stream:
            raise ValueError("Cannot stream results with best_of > 1")

    @retry(Timeout, tries=3)
    def generate(self, prompt, stop=None, max_length=None):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        params = OmegaConf.to_object(self.generation_params)
        params["prompt"] = prompt

        if stop is None:
            stop = self.generation_params.stop
        params["stop"] = stop

        if max_length is not None:
            params["max_tokens"] = max_length

        # if self.verbose:
        #     print(f"Prompt: {prompt}")

        self.response = self.client.create(**params)
        return self.response.choices[0].text
