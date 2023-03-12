import os
import torch
import numpy as np
import scipy.special
import logging

from torchfly.flylogger import FlyLogger
from torchfly.flyconfig import FlyConfig
from torchfly.utilities import set_random_seed
from omegaconf import OmegaConf

from memformers.models.membart import MemBartFlyModel
from memformers.recurrent_training.recurrent_trainer import RecurrentTrainer

from dataloader import DataLoaderHelper

logger = logging.getLogger(__name__)


def main():
    # load config
    config = FlyConfig.load("config/membart.yaml")
    set_random_seed(config.training.random_seed)


    print("Loading Example Data")
    data_helper = DataLoaderHelper(config)
    train_dataloader = data_helper.init_train_loader("./data/msc_dialog_samples.json")
    valid_dataloader = data_helper.init_valid_loader("./data/msc_dialog_samples.json")

    
    print("Loading Model")
    model = MemBartFlyModel(config)
    model.configure_metrics()

    trainer = RecurrentTrainer(config.training, model)

    print("Start Training")
    with FlyLogger(config.flylogger) as flylogger:
        # save_config
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f)
            logger.info(config)

        trainer.train(config.training, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()
