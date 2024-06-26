import os

from omegaconf import OmegaConf
from hydra.utils import instantiate

from llm.llm.openai import OpenAI 
from llm.llm.openai_chat import OpenAIChat  

def instantiate_llm(llm_name, generation_params={}, **kwargs):
    llm_config_path = f'llm/conf/llm/{llm_name}.yaml'
    assert os.path.exists(llm_config_path), f'LLM config file not found at {llm_config_path}'

    # Load the LLM config file
    llm_config = OmegaConf.load(llm_config_path)

    # Update the config with the kwargs
    if generation_params:
        llm_config.generation_params = OmegaConf.merge(llm_config.generation_params, OmegaConf.create(generation_params))

    if kwargs:
        llm_config = OmegaConf.merge(llm_config, OmegaConf.create(kwargs))

    llm = instantiate(llm_config.llm)(llm_config)

    return llm
