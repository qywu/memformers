from typing import Any, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from .recurrent_memory import RecurrentMemory


class RecurrentTrainingCell(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, inputs: Any, memory: RecurrentMemory
    ) -> Tuple[torch.Tensor, RecurrentMemory]:
        """ Recurrent Cell takes one input and memory_0 at a time and output memory_1
        Args:
            inputs: the input for cell 
            memory_0: the input memory state 
        Returns:
            output: the output from the cell
            memory_1: the next memory state
        """
        raise NotImplementedError

    @abstractmethod
    def construct_memory(self, batch_size) -> RecurrentMemory:
        raise NotImplementedError