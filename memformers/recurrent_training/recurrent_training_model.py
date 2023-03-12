from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn

from .recurrent_memory import RecurrentMemory
from .recurrent_training_cell import RecurrentTrainingCell

# pylint:disable=no-member


def get_device_rng_state(device):
    with torch.cuda.device(device):
        cuda_rng_state = torch.cuda.get_rng_state()
    return cuda_rng_state


@dataclass
class RecurrentTrainingModelOutput:
    outputs: List = None
    memory: RecurrentMemory = None
    history_memories: List[RecurrentMemory] = None
    rng_states: Dict = None


class RecurrentTrainingModel(nn.Module):
    """ RecurrentModel should warp all the network weights including the memory initialization
    """

    def __init__(self, recurrent_training_cell: RecurrentTrainingCell):
        super().__init__()
        self.recurrent_training_cell = recurrent_training_cell

    def forward(
        self,
        rollout: List,
        memory: RecurrentMemory,
        output_history_memories: bool = False,
        output_rng_states: bool = False,
    ):
        """ This is an example of passing a list of inputs
        Args:
            rollout (list): list of inputs
            memory_0 (RecurrentMemory): initial memory state
        Returns:
            outputs (list): list of outputs
            memory_n: the final memory state
        """
        device = next(self.parameters()).device
        time_horizon = len(rollout)
        outputs = []
        history_memories = []
        rng_states = {"fwd_cpu_state": [], "fwd_gpu_states": []}

        for time in range(time_horizon):
            input = rollout[time]

            if output_history_memories:
                history_memories.append(memory)

            if output_rng_states:
                rng_states["fwd_cpu_state"].append(torch.get_rng_state())
                if str(device) != "cpu":
                    rng_states["fwd_gpu_states"].append(get_device_rng_state(device))

            output, memory = self.recurrent_training_cell(input, memory)
            outputs.append(output)

        return RecurrentTrainingModelOutput(
            outputs=outputs,
            memory=memory,
            history_memories=history_memories,
            rng_states=rng_states,
        )

    def construct_memory(self, batch_size) -> RecurrentMemory:
        return self.recurrent_training_cell.construct_memory(batch_size)
