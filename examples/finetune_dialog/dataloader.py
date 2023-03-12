import os
import json
import copy
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from collections import deque
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, T5Model
import logging

logger = logging.getLogger(__name__)


class DialogDataAgent:
    def __init__(self, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # shared queue
        self.iterator = None
        self.start_prompt = tokenizer.encode("<conversation starts>\n\n", add_special_tokens=False)
        self.output_role = tokenizer.encode("B", add_special_tokens=False)[0]

    def get(self, queue):
        if self.iterator is None:
            if len(queue) == 0:
                # if_empty, (reset, result)
                return True, (True, None)
            else:
                example = queue.popleft()
                self.iterator = iter(self.iter_example(example))

        output = next(self.iterator, None)
        if output is None:
            self.iterator = None
            return self.get(queue)
        else:
            return output

    def iter_example(self, example):
        history = copy.copy(self.start_prompt)
        reset = True

        for item in example:
            turn_token_ids = item["token_ids"]
            role_token = turn_token_ids[0]

            if self.output_role == role_token:
                result = {
                    "session": item["session"],
                    "dial_id": item["dial_id"],
                    "turn_id": item["turn_id"],
                    "history": [self.tokenizer.bos_token_id]
                    + history[-self.max_seq_length - 2 :]
                    + [self.tokenizer.eos_token_id],
                    "labels": [self.tokenizer.bos_token_id] + turn_token_ids,
                }
                yield False, (reset, result)
                reset = False

            history += turn_token_ids


class SeriesQueueDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        tokenizer,
        max_seq_length,
        shuffle=False,
        collate_fn=None,
        enable_length=True,
        output_if_empty=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.enable_length = enable_length
        self.output_if_empty = output_if_empty

        self._saved_length = None

        self.agents = [DialogDataAgent(tokenizer, max_seq_length) for i in range(self.batch_size)]

    def __iter__(self):
        queue = deque()  # full of datas
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            queue.append(self.dataset[idx])

        count = 0
        while True:
            batch = [self.agents[i].get(queue) for i in range(self.batch_size)]
            if_empty, batch = zip(*batch)

            if (isinstance(self._saved_length, int) and count >= self._saved_length) or all(if_empty) is True:
                break

            if self.collate_fn is not None:
                batch = self.collate_fn((if_empty, batch))
                yield batch
            else:
                yield if_empty, batch

            count += 1

    def __len__(self):
        if not self.enable_length:
            raise NotImplementedError

        if self.shuffle is False:
            count = 0
            for batch in self:
                count += 1
            return count
        else:
            if self._saved_length is None:
                count = 0
                for batch in self:
                    count += 1
                    # avoid over batching
                self._saved_length = (count // self.batch_size - 2) * self.batch_size
            return self._saved_length


class DataLoaderHelper:
    def __init__(self, config) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.task.model_name_or_path)

    def eval_collate_fn(self, batch):
        # assert num_batch_chunks == 1
        if_empty, batch = batch
        resets, batch = zip(*batch)
        batch = list(batch)
        # deal with None
        for i in range(len(batch)):
            if batch[i] is None:
                batch[i] = {"history": [], "labels": [], "session": "none", "dial_id": "empty", "turn_id": "empty"}
                # logger.warn("DetDialogDataAgent
        input_ids = [torch.LongTensor(item["history"]) for item in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.BoolTensor(input_ids != self.tokenizer.pad_token_id)
        labels = [torch.LongTensor(item["labels"]) for item in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_input_ids = labels[:, :-1].clone()
        labels = labels[:, 1:].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        decoder_attention_mask = torch.BoolTensor(decoder_input_ids != self.tokenizer.pad_token_id)

        session_id = [item["session"] for item in batch]
        dial_ids = [item["dial_id"] for item in batch]
        turn_ids = [item["turn_id"] for item in batch]

        output = {
            "session_id": session_id,
            "dial_ids": dial_ids,
            "turn_ids": turn_ids,
            "reset": torch.BoolTensor(resets),
            "encoder_input_ids": input_ids,
            "encoder_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "target": labels,
        }
        return output

    def train_collate_fn(self, batch):
        return [self.eval_collate_fn(batch)]

    def init_train_loader(self, file_path):
        with open(file_path) as f:
            data = json.load(f)

        dataset = [data[key] for key in data.keys()]

        dataloader = SeriesQueueDataLoader(
            dataset,
            self.config.training.batch_size,
            self.tokenizer,
            self.config.task.max_seq_length,
            collate_fn=self.train_collate_fn,
            shuffle=True,
        )
        print("Loading dataloader length: ", len(dataloader))
        return dataloader

    def init_valid_loader(self, file_path):
        with open(file_path) as f:
            data = json.load(f)

        dataset = [data[key] for key in data.keys()]

        dataloader = SeriesQueueDataLoader(
            dataset,
            self.config.training.evaluation.batch_size,
            self.tokenizer,
            self.config.task.max_seq_length,
            collate_fn=self.eval_collate_fn,
            shuffle=False,
        )
        print("Loading dataloader length: ", len(dataloader))
        return dataloader