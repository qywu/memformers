from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import torch.nn as nn


class RecurrentMemory(ABC):
    """ This is a abstract class for defining general type of memory
        in a recurrent network.
    """
    def __init__(self, memory_states, batch_size: int):
        self.memory_states = memory_states
        self.batch_size = batch_size

    @abstractmethod
    def to(self, device):
        self.memory_states.to(device)

    @abstractmethod
    def update(self, new_memory_states) -> 'RecurrentMemory':
        self.memory_states += new_memory_states

    @abstractmethod
    def chunk(self, chunks: int) -> List['RecurrentMemory']:
        """
        chunks (int): number of chunks to return 
        """
        split_size = self.batch_size // chunks
        memories = []

        for i in range(chunks):
            begin_idx = i * split_size
            end_idx = min((i + 1) * split_size, self.batch_size)
            chunk_size = end_idx - begin_idx

            memory_chunk = self.memory_states[begin_idx:end_idx]
            memories.append(RecurrentMemory(memory_chunk, chunk_size))

        return memories

    def __repr__(self) -> str:
        return "RecurrentMemory:" + repr(self.memory_states)