from typing import Any, List, Dict, Iterator, Callable, Iterable
import os
import random
import numpy as np
from dataclasses import dataclass
import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig

# from apex.parallel import Reducer
import socket

# local imports
import torchfly.distributed as distributed
from torchfly.training.callbacks import Callback, CallbackHandler, Events
from torchfly.training.callbacks import Checkpoint, Evaluation
from torchfly.utilities import move_to_device
from torchfly.training import Trainer, FlyModel

from torchfly.flylogger.train_logger import TrainLogger

import logging

logger = logging.getLogger(__name__)

# pylint: disable=no-member


class RecurrentTrainer(Trainer):
    def __init__(self, config: DictConfig, model: FlyModel, *args, **kwargs):
        super().__init__(config, model, *args, **kwargs)
        # exponential discounted return
        self.loss_weights = np.power(2, np.arange(1, self.config.time_horizon + 1))
        self.loss_weights = self.loss_weights / np.linalg.norm(self.loss_weights, 1)

    def init_training_constants(self, config):
        super().init_training_constants(config)
        if self.training_in_epoch:
            if self.epoch_num_batches is not None:
                self.epoch_num_batches = self.epoch_num_batches // (self.config.time_horizon - self.config.time_overlap)
                self.total_num_batches = self.epoch_num_batches * self.total_num_epochs 
                self.total_num_update_steps = (
                    self.total_num_batches // self.gradient_accumulation_batches
                )
                self.epoch_num_update_steps = (
                    self.epoch_num_batches // self.gradient_accumulation_batches
                )

    def train_epoch(self):
        self.optimizer = self.optimizers[0]
        self.scheduler = self.schedulers[0]

        self.local_step_count = 0
        batch_size = self.config.batch_size
        num_batch_chunks = self.config.num_batch_chunks
        chunk_size = batch_size // num_batch_chunks
        assert batch_size % num_batch_chunks == 0

        # Initialize memory
        memory = self.model.construct_memory(batch_size)
        memories = memory.chunk(num_batch_chunks)
        history_inputs = [[] for _ in range(num_batch_chunks)]
        self.last_memories = memories

        # Iterate all batches in train dataloader

        for batch in self.train_dataloader:
            for chunk in range(num_batch_chunks):
                history_inputs[chunk].append(batch[chunk])

            if len(history_inputs[0]) == self.config.time_horizon:
                self.callback_handler.fire_event(Events.BATCH_BEGIN)

                # Train the model
                for chunk in range(num_batch_chunks):
                    output, memories[chunk] = self.train_iteration(
                        history_inputs[chunk], self.last_memories[chunk]
                    )
                    # update each chunk's last memory
                    if self.config.time_overlap != 0:
                        self.last_memories[chunk] = output.history_memories[-self.config.time_overlap]
                        history_inputs[chunk] = history_inputs[chunk][-self.config.time_overlap :]                        

                # keep the most recent history inputs
                if self.config.time_overlap == 0:
                    history_inputs = [[] for _ in range(num_batch_chunks)]
                    self.last_memories = memories                    

                # Update the model with optimizer
                if (self.global_batch_count + 1) % self.gradient_accumulation_batches == 0:
                    self.step_update(self.model, self.optimizer, self.scheduler)
                    self.global_step_count += 1
                    self.local_step_count += 1

                self.model.set_memory_params(memories)

                self.callback_handler.fire_event(Events.BATCH_END)

                if self.global_step_count >= self.total_num_update_steps:
                    break

                self.global_batch_count += 1
            else:
                continue

    def train_iteration_efficient(self, rollout, memory_0):
        self.model.train()

        rollout = move_to_device(rollout, self.device)

        backward_memory_grad = None
        prev_memory = memory_0
        prev_memory.detach_()

        # Memory Replay Back-Propagation Forward
        with torch.cuda.amp.autocast(self.fp16):
            with torch.no_grad():
                model_output, memory_n = self.model(rollout, prev_memory)

        # check if gradient is inf
        if (not self.fp16) or (self.loss_scaler._scale is None) or (not sum(
                v.item() for v in self.loss_scaler._check_inf_per_device(self.optimizer).values())):
            for time in reversed(range(len(rollout))):
                with torch.random.fork_rng(devices=[self.device]):
                    prev_memory = model_output.history_memories[time]
                    prev_memory.retain_grad()

                    torch.set_rng_state(model_output.rng_states["fwd_cpu_state"][time])
                    torch.cuda.set_rng_state(model_output.rng_states["fwd_gpu_states"][time])

                    # recompute the step
                    with torch.cuda.amp.autocast(self.fp16):
                        step_input = rollout[time]
                        step_output, current_memory = self.model.recurrent_model.recurrent_cell(step_input, prev_memory)
                        loss = self.model.compute_loss(step_input, step_output)

                # backward order must be perserved
                # memory direction first
                if backward_memory_grad is not None:
                    current_memory.backward(backward_memory_grad)
                # loss direction second
                self.loss_backward(loss * self.loss_weights[time])

                backward_memory_grad = prev_memory.grad

            self.model.training_metrics["mem_grad_std"](backward_memory_grad.std().item())
            self.model.training_metrics["mem_grad_max"](backward_memory_grad.max().item())

        return model_output, memory_n

    def train_iteration(self, rollout, memory_0):
        self.model.train()
        rollout = move_to_device(rollout, self.device)

        total_loss = 0.0
        memory_0.detach_()
        prev_memory = memory_0
        all_memories = []

        # Memory Replay Back-Propagation Forward

        with torch.cuda.amp.autocast(self.fp16):
            for time in range(len(rollout)):
                step_input = rollout[time]
                prev_memory.retain_grad()
                all_memories.append(prev_memory)
                step_output, current_memory = self.model.model.recurrent_training_cell(step_input, prev_memory)
                loss = self.model.compute_step_loss(step_input, step_output)
                #  for the last one
                prev_memory = current_memory
                total_loss = total_loss + loss * self.loss_weights[time]

        self.loss_backward(total_loss)

        if memory_0.grad is not None:
            self.model.training_metrics["mem_grad_std"](memory_0.grad.std().item())
            self.model.training_metrics["mem_grad_max"](memory_0.grad.max().item())

        step_output.history_memories = all_memories

        return step_output, current_memory

    def backward_batch(self, batch, memory):
        batch = move_to_device(batch, self.device)
        with torch.cuda.amp.autocast(self.fp16):
            output, memory = self.model(batch, memory)
        self.loss_backward(output.loss)
        return output, memory

    # def validate(self, dataloader):
    #     self.callback_handler.fire_event(Events.VALIDATE_BEGIN)
    #     self.model.reset_evaluation_metrics()

    #     # Validation
    #     self.model.eval()
    #     # No gradient is needed for validation
    #     with torch.no_grad():
    #         # Progress bar
    #         pbar = tqdm.tqdm(dataloader) if self.rank == 0 else self.validation_dataloader
    #         pbar.mininterval = 2.0
    #         # Initialize memory
    #         memory = self.model.construct_memory(self.config.evaluation.batch_size)

    #         for batch in pbar:
    #             batch = move_to_device(batch, self.device)
    #             _, memory = self.model.predict([batch], memory)

    #     # END
    #     self.model.train()
    #     self.callback_handler.fire_event(Events.VALIDATE_END)

    # def test(self):
    #     self.callback_handler.fire_event(Events.TEST_BEGIN)
    #     self.model.reset_evaluation_metrics()
    #     # Validation
    #     self.model.eval()
    #     # No gradient is needed for test
    #     with torch.no_grad():
    #         # Progress bar
    #         pbar = tqdm.tqdm(self.test_dataloader) if self.rank == 0 else self.test_dataloader
    #         pbar.mininterval = 2.0
    #         # Initialize memory
    #         memory = self.model.construct_memory(self.config.evaluation.batch_size)

    #         for rollout in pbar:
    #             # send to cuda device
    #             for batch in rollout:
    #                 batch = move_to_device(batch, self.device)
    #                 _, memory = self.model.predict([batch], memory)
    #     # END
    #     self.model.train()
    #     self.callback_handler.fire_event(Events.TEST_END)
