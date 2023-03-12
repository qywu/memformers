from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
from torchfly.training import FlyModel

from .recurrent_memory import RecurrentMemory
from .recurrent_training_model import RecurrentTrainingModel

# pylint:disable=no-member


class RecurrentTrainingFlyModel(FlyModel):
    def __init__(self, config):
        super().__init__(config)
        self.model: RecurrentTrainingModel = None

    def forward(self, rollout, memory):
        model_output = self.rt_model(
            rollout, memory, output_history_memories=True, output_rng_states=True
        )
        return model_output, model_output.memory

    @abstractmethod
    def compute_step_loss(self, step_input, step_output):
        raise NotImplementedError

    def compute_rollout_loss(self, rollout_inputs, rollout_outputs):
        loss = 0.0
        for idx in range(len(rollout_inputs)):
            step_inputs = rollout_inputs[idx]
            step_outputs = rollout_outputs[idx]
            loss += self.compute_step_loss(step_inputs, step_outputs)
        return loss

    def construct_memory(self, batch_size) -> RecurrentMemory:
        return self.recurrent_model.construct_memory(batch_size)

    def set_memory_params(self, memory: RecurrentMemory):
        pass
