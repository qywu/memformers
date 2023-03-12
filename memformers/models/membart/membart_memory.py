from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...recurrent_training.recurrent_memory import RecurrentMemory

# pylint:disable=no-member


class MemBartMemory(RecurrentMemory):
    def __init__(self, memory_states, batch_size: int):
        super().__init__(memory_states, batch_size)

    def to(self, device: torch.device):
        self.memory_states.to(device)

    def update(self, new_memory_states) -> RecurrentMemory:
        self.memory_states = new_memory_states

    def chunk(self, chunks: int) -> List[RecurrentMemory]:
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
            memories.append(MemBartMemory(memory_chunk, chunk_size))

        return memories

    def retain_grad(self):
        if not self.memory_states.requires_grad:
            self.memory_states.requires_grad = True
        self.memory_states.retain_grad()

    def backward(self, grad):
        self.memory_states.backward(grad, retain_graph=True)

    def detach_(self):
        self.memory_states = self.memory_states.detach()

    @property
    def grad(self):
        return self.memory_states.grad
