# Memformers

Memformers utilize a external dynamic memory to store history information. 
This repo contains implementation of the pre-trained model MemBART and its training code.

Check the repo [memformers](https://github.com/qywu/memformers) for details.

## Install

Download this repo and install it with:
```bash
git clone https://github.com/qywu/memformers
cd memformers
pip install -e .
```

## Usage


### Inference and Generation

Our implementation is based on huggingface [transformers](https://github.com/huggingface/transformers). Currently, we provide two checkpoints `"qywu/membart-large"` [(checkpooint)](https://huggingface.co/qywu/membart-large) and `"qywu/membart-base"`[(checkpooint)](https://huggingface.co/qywu/membart-base). 
You can directly load the checkpoint with:

```python
import torch
from transformers import AutoTokenizer
from memformers.models.membart import MemBartForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# load the large model in huggingface way
membart = MemBartForConditionalGeneration.from_pretrained("qywu/membart-large")


text1 = "Barack Obama served as the 44th President of the United States."
text2 = "<mask> served as the 44th President of the United States."

# construct the initial memory
memory_states = membart.construct_memory(batch_size=1)

# t = 0
input_ids1 = torch.LongTensor([tokenizer.encode(text1)])
# only run the encoder to get memory states
encoder_outputs = membart.model.encoder(input_ids=input_ids1, memory_states=memory_states, attention_mask=None)
memory_states = encoder_outputs.memory_states


# t = 1
input_ids2 = torch.LongTensor([tokenizer.encode(text2)])

encoder_outputs2 = membart.model.encoder(input_ids=input_ids2, memory_states=memory_states, attention_mask=None)

outputs = membart.generate(
    encoder_outputs=encoder_outputs2,
    decoder_start_token_id=tokenizer.bos_token_id,
    max_length=64,
    num_beams=1,
    do_sample=False,
    return_dict_in_generate=True,
)

print(tokenizer.decode(outputs.sequences[0]))
# Barack Obama served as the 44th President of the United States.
```


Note that due to [BART](https://arxiv.org/abs/1910.13461) denosing pre-training, it needs to further fine-tune the model on the downstream tasks to get better performance.

### Training

Training requires to install [TorchFly](https://github.com/qywu/TorchFly).
```bash
git clone https://github.com/qywu/TorchFly
cd TorchFly
pip install -e .
```

Then, you can refer to the code in `examples/finetune_dialog` for details about finetuning or further pre-training MemBart on your tasks.

```python
python train.py
```

For details, see `examples/training_msc`.

## Citations

Memformer: A Memory-Augmented Transformer for Sequence Modeling
```bib
@inproceedings{DBLP:conf/ijcnlp/WuLQGGY22,
  author    = {Qingyang Wu and
               Zhenzhong Lan and
               Kun Qian and
               Jing Gu and
               Alborz Geramifard and
               Zhou Yu},
  title     = {Memformer: {A} Memory-Augmented Transformer for Sequence Modeling},
  booktitle = {Findings of the Association for Computational Linguistics: {AACL-IJCNLP}
               2022, Online only, November 20-23, 2022},
  pages     = {308--318},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.findings-aacl.29},
  timestamp = {Tue, 29 Nov 2022 14:53:03 +0100},
  biburl    = {https://dblp.org/rec/conf/ijcnlp/WuLQGGY22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Stateful Memory-Augmented Transformers for Dialogue Modeling
```bib
@article{DBLP:journals/corr/abs-2209-07634,
  author    = {Qingyang Wu and
               Zhou Yu},
  title     = {Stateful Memory-Augmented Transformers for Dialogue Modeling},
  journal   = {CoRR},
  volume    = {abs/2209.07634},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2209.07634},
  doi       = {10.48550/arXiv.2209.07634},
  eprinttype = {arXiv},
  eprint    = {2209.07634},
  timestamp = {Tue, 27 Sep 2022 16:29:43 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2209-07634.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
