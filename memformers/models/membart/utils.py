import os
from omegaconf import OmegaConf

def get_model_config(name="membart-large.yaml"):
    file = os.path.join(os.path.dirname(__file__), "config", name)
    return OmegaConf.load(file)