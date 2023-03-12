from typing import Any, Dict, Tuple, List, Union
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfly.metrics import Average, MovingAverage, Speed
from torchfly.training.schedulers import WarmupWarpper, WarmupLinearSchedule
from torchfly.utilities import move_to_device
from transformers import BartModel, PretrainedConfig

from .modeling_membart import MemBartModel
from .membart_memory import MemBartMemory
from .utils import get_model_config
from ...recurrent_training.recurrent_training_cell import RecurrentTrainingCell
from ...recurrent_training.recurrent_training_model import RecurrentTrainingModel
from ...recurrent_training.recurrent_training_flymodel import RecurrentTrainingFlyModel


def sync_linear(a, b):
    a.weight.data.copy_(b.weight.data)
    a.bias.data.copy_(b.bias.data)


def sync_ln(a, b):
    a.weight.data.copy_(b.weight.data)
    a.bias.data.copy_(b.bias.data)


class MemBartTrainingCell(RecurrentTrainingCell):
    def __init__(self, config):
        super().__init__()
        self.cell = MemBartModel.from_pretrained(config.task.model_name_or_path)

        self.cell.shared.weight.requires_grad = False
        self.cell.encoder.embed_positions.weight.requires_grad = False
        self.cell.decoder.embed_positions.weight.requires_grad = False

    def forward(self, inputs: Any, memory: MemBartMemory) -> Tuple[Any, MemBartMemory]:
        memory_states = memory.memory_states

        encoder_outputs = self.cell.encoder(
            input_ids=inputs["encoder_input_ids"],
            memory_states=memory_states,
            memory_resets=inputs["reset"],
            attention_mask=inputs["encoder_attention_mask"],
        )

        return (
            encoder_outputs,
            MemBartMemory(encoder_outputs.memory_states, len(inputs["reset"])),
        )

    def construct_memory(self, batch_size):
        return MemBartMemory(self.cell.construct_memory(batch_size), batch_size)


class MemBartFlyModel(RecurrentTrainingFlyModel):
    def __init__(self, config):
        super().__init__(config)
        recurrent_training_cell = MemBartTrainingCell(config)
        recurrent_training_model = RecurrentTrainingModel(recurrent_training_cell)
        self.model = recurrent_training_model
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.training = True

    def forward(self, rollout, memory):
        model_outputs = self.model(rollout, memory, output_history_memories=True, output_rng_states=True)
        return model_outputs, model_outputs.memory

    def predict_step(self, batch_idx, batch, memory):
        step_output, new_memory = self(batch, memory)
        loss = self.compute_step_loss(batch[0], step_output.outputs[0], training=False)
        return loss, new_memory

    def compute_step_loss(self, step_input, step_output, training=True):
        decoder_outputs = self.model.recurrent_training_cell.cell.decoder(
            input_ids=step_input["decoder_input_ids"],
            attention_mask=step_input["decoder_attention_mask"],
            encoder_hidden_states=step_output.last_hidden_state,
            encoder_attention_mask=step_output.encoder_attention_mask,
            return_dict=True,
        )
        lm_logits = F.linear(decoder_outputs.last_hidden_state, self.model.recurrent_training_cell.cell.shared.weight,)
        lm_loss = self.loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), step_input["target"].view(-1))

        # log
        if training:
            self.training_metrics["loss"](lm_loss.item())
            # seq_len x batch_size
            self.training_metrics["tok/s"](
                step_input["target"].shape[0] * step_input["target"].shape[1] * self.config.training.num_gpus_per_node
            )
        else:
            self.evaluation_metrics["loss"](lm_loss.item())

        return lm_loss

    def configure_metrics(self):
        self.training_metrics = {
            "mem_grad_std": MovingAverage(name="mem_grad_std"),
            "mem_grad_max": MovingAverage(name="mem_grad_max"),
            "loss": MovingAverage(name="loss"),
            "tok/s": Speed(),
        }
        self.evaluation_metrics = {"loss": Average()}

    def get_training_metrics(self) -> Dict[str, str]:
        loss = self.training_metrics["loss"].get_metric()
        ppl = np.exp(loss)
        tok_s = self.training_metrics["tok/s"].get_metric()
        lr = self.get_last_lr()[0]
        mem_grad_std = self.training_metrics["mem_grad_std"].get_metric()
        mem_grad_max = self.training_metrics["mem_grad_max"].get_metric()

        metrics = {
            "tok/s": f"{tok_s:5.0f}",
            "lr": f"{lr:3.2e}",
            "mem_grad_std": f"{mem_grad_std:4.2e}",
            "mem_grad_max": f"{mem_grad_max:4.2e}",
            "loss": f"{loss:8.4f}",
            "ppl": f"{ppl:8.4f}",
        }
        return metrics

    def get_evaluation_metrics(self) -> Dict[str, str]:
        loss = self.evaluation_metrics["loss"].get_metric()
        ppl = np.exp(loss)
        metrics = {"loss": f"{loss:8.4f}", "ppl": f"{ppl:8.4f}", "score": f"{-ppl:8.4f}"}
        return metrics

    def construct_memory(self, batch_size) -> MemBartMemory:
        return self.model.construct_memory(batch_size)

    def set_memory_params(self, memory: MemBartMemory):
        pass

    def configure_optimizers(self, config, total_num_update_steps) -> Union[List, List]:
        optimizer_grouped_parameters = [
            {
                "params": self.model.parameters(),
                "lr": config.optimization.learning_rate,
                "weight_decay": config.optimization.weight_decay,
            },
        ]

        betas = config.optimization.get("betas", (0.9, 0.999))
        warmup_steps = config.scheduler.warmup_steps

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1000, betas=betas)

        scheduler = scheduler = WarmupLinearSchedule(optimizer, warmup_steps, total_num_update_steps)

        self.get_last_lr = scheduler.get_last_lr

        return [optimizer], [scheduler]

    def validation_loop(self, dataloader):
        # No gradient is needed for validation
        self.training = False
        self.eval()
        self.reset_evaluation_metrics()

        with torch.no_grad():
            # Progress bar
            pbar = tqdm.tqdm(dataloader)
            pbar.mininterval = 2.0
            # Initialize memory
            memory = self.model.construct_memory(self.config.training.evaluation.batch_size)

            for batch_idx, batch in enumerate(pbar):
                batch = move_to_device(batch, self.device)
                _, memory = self.predict_step(batch_idx, [batch], memory)

        self.training = True

    def test_loop(self, dataloader):
        return self.validation_loop(dataloader)