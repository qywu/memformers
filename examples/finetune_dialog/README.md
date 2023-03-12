# Fine-tuning

This example code shows how to fine-tune MemBart for dialogue data.

First, the dialogue data needs to be processed into the format like `data/msc_dialog_samples.json`. 
The below shows the example.

```python
{
    "dialog_1": [
        {
            "dial_id": "dialog_1",
            "turn_id": 1,
            "text": "What kind of car do you own? I have a jeep.",
            "role": "bot_0",
            "session": 1,
            "token_ids": [250, 35, ...]
        },
        {
            "dial_id": "dialog_1",
            "turn_id": 2,
            "text": "I don't own my own car! I actually really enjoying walking and running, but then again, I live in a small town and semi-close to work.",
            "role": "bot_1",
            "session": 1,
            "token_ids": [387, 35, ...]
        },
        ...
    ]
}
```

In `dataloader.py`, there is a class `DialogDataAgent` manages the conversation history. 
You can customize it for different tasks.
```python
class DialogDataAgent:
    ...

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
```

In `train.py`, we use [TorchFly](https://github.com/qywu/TorchFly) for training.
It uses the `MemBartFlyModel` to handle the training.
Also, we write a custom class `RecurrentTrainer` to maintain the memory states between timesteps.
You can easily change the configuration in `config/membart.yaml` and `config/training/train_128.yaml`.

```python
from torchfly.flyconfig import FlyConfig
from memformers.recurrent_training.recurrent_trainer import RecurrentTrainer
from memformers.models.membart import MemBartFlyModel


config = FlyConfig.load("config/membart.yaml")
train_dataloader = ...
valid_dataloader = ...

model = MemBartFlyModel(config)
model.configure_metrics()


trainer = RecurrentTrainer(config.training, model)
trainer.train(config.training, train_dataloader, valid_dataloader)
```

For details, please refer to the code.
